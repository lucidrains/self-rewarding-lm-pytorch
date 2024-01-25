import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from beartype import beartype
from beartype.typing import Optional, Callable

from tqdm import tqdm

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

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

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
@beartype()
def sample(
    net: Module,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_fn: Callable = top_p,
    filter_kwargs: dict = dict()
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in tqdm(range(sample_num_times)):
        logits = net(out)
        logits = logits[:, -1]

        logits = filter_fn(logits, **filter_kwargs)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]
