import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from beartype import beartype
from beartype.typing import Optional, Callable, List, Tuple

from einops import rearrange

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True, eps = 1e-10):
    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, frac_num_tokens = 0.1, k: Optional[int] = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# decoding

@torch.no_grad()
@beartype
def sample(
    net: Module,
    prompts,
    seq_len: int,
    temperature = 1.,
    filter_fn: Callable = top_p,
    filter_kwargs: dict = dict(),
    pad_id: int = -1,
    eos_id: Optional[int] = None,
    output_keep_prompt = False
):
    device = next(net.parameters()).device
    net.eval()

    if isinstance(prompts, (tuple, list)):
        prompts = pad_sequence(prompts, batch_first = True, padding_value = pad_id)

    batch, prompts_tensor_len = prompts.shape

    batch_arange = torch.arange(batch, device = device)[..., None]

    prompt_lens = (prompts != pad_id).sum(dim = -1)
    curr_seq_indices = prompt_lens[..., None]

    out = prompts.clone()

    while (curr_seq_indices < seq_len).any():
        out = F.pad(out, (0, 1), value = pad_id)

        net_input = out.masked_fill(out == pad_id, 0)

        logits = net(net_input)

        logits = logits[batch_arange, curr_seq_indices]
        logits = rearrange(logits, 'b 1 d -> b d')

        logits = filter_fn(logits, **filter_kwargs)
        sampled_tokens = gumbel_sample(logits, temperature = temperature, dim = -1)

        out[batch_arange, curr_seq_indices] = sampled_tokens

        curr_seq_indices += 1
        curr_seq_indices.clamp_(max = seq_len)

        if not exists(eos_id):
            continue

        is_eos_mask = out == eos_id
        all_eos = is_eos_mask.any(dim = -1).all()

        if all_eos:
            break

    out = out[:, :seq_len]

    if exists(eos_id):
        is_eos_mask = out == eos_id
        after_eos_mask = F.pad(is_eos_mask.cumsum(dim = -1) > 0, (1, -1), value = False)
        out = out.masked_fill_(after_eos_mask, pad_id)

    if output_keep_prompt:
        return out

    prompt_mask = torch.arange(out.shape[-1], device = device) < prompt_lens[..., None]

    generated_seq_mask = out != pad_id & ~prompt_mask
    seq_lens = generated_seq_mask.sum(dim = -1).tolist()

    return out[generated_seq_mask].split(seq_lens)
