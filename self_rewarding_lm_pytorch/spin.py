from pathlib import Path
from copy import deepcopy

from beartype import beartype
from beartype.typing import Optional, Callable
from torchtyping import TensorType

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from accelerate import Accelerator

from einops import rearrange
from einx import get_at

from pytorch_custom_utils.utils import (
    masked_mean
)

from self_rewarding_lm_pytorch.dpo import (
    adam_optimizer_with_linear_decay
)

from self_rewarding_lm_pytorch.sampling_utils import (
    sample,
    top_p,
    top_k
)

from tqdm import tqdm

# helper functions

def exists(v):
    return v is not None

def freeze_all_layers_(module):
    for param in module.parameters():
        param.requires_grad = False

def log_prob_from_model_and_seq(model, seq):
    logits = model(seq)
    log_probs = logits.log_softmax(dim = -1)
    return get_at('b n [c], b n -> b n', log_probs, seq)

def prompt_mask_from_len(lengths, seq):
    seq_len, device = seq.shape[-1], seq.device
    return torch.arange(seq_len, device = device) < rearrange(lengths, '... -> ... 1')

def maybe_and_mask(*masks):
    masks = [*filter(exists, masks)]
    if len(masks) == 0:
        return None

    mask, *rest_masks = masks
    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask

# main class

class SPIN(Module):
    def __init__(
        self,
        model: Module,
        *,
        λ = 0.1,
        pad_id: Optional[int] = None
    ):
        super().__init__()
        self.policy_model = model

        self.ref_model = deepcopy(model)
        freeze_all_layers_(self.ref_model)

        self.λ = λ
        self.pad_id = pad_id

    def update_reference_model_with_policy(self):
        self.ref_model.load_state_dict(self.policy_model.state_dict())

    def parameters(self):
        return self.policy_model.parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    @autocast(enabled = False)
    def forward(
        self,
        generated_seq: TensorType['b', 'n', int],
        real_seq: TensorType['b', 'n', int],
        prompt_len: TensorType['b', int],
        generated_seq_mask: Optional[TensorType['b', 'n', bool]] = None,
        real_seq_mask: Optional[TensorType['b', 'n', bool]] = None
    ):
        self.policy_model.train()

        """
        b - batch
        n - sequence length
        """
        assert generated_seq.ndim == real_seq.ndim == 2

        real_prompt_mask = torch.arange(real_seq.shape[-1], device = self.device) < prompt_len[:, None]
        generated_prompt_mask = torch.arange(generated_seq.shape[-1], device = self.device) < prompt_len[:, None]

        """
        Equation 4.7 in https://arxiv.org/abs/2401.01335v1
        """

        if exists(self.pad_id):
            assert not exists(generated_seq_mask)
            assert not exists(real_seq_mask)
            generated_seq_mask = generated_seq != self.pad_id
            generated_seq.masked_fill_(~generated_seq_mask, 0)

            real_seq_mask = real_seq != self.pad_id
            real_seq.masked_fill_(~real_seq_mask, 0)

        with torch.no_grad():
            self.ref_model.eval()
            ref_generated_logprob = log_prob_from_model_and_seq(self.ref_model, generated_seq)
            ref_real_logprob = log_prob_from_model_and_seq(self.ref_model, real_seq)

        policy_generated_logprob = log_prob_from_model_and_seq(self.policy_model, generated_seq)
        policy_real_logprob = log_prob_from_model_and_seq(self.policy_model, real_seq)

        # masked mean for variable lengths

        policy_generated_logprob, ref_generated_logprob = [masked_mean(seq, maybe_and_mask(generated_seq_mask, ~generated_prompt_mask)) for seq in (policy_generated_logprob, ref_generated_logprob)]
        policy_real_logprob, ref_real_logprob = [masked_mean(seq, maybe_and_mask(real_seq_mask, ~real_prompt_mask)) for seq in (policy_real_logprob, ref_real_logprob)]

        # SPIN loss

        losses = -F.logsigmoid(self.λ * ((policy_real_logprob - ref_real_logprob) - (policy_generated_logprob - ref_generated_logprob)))

        return losses.mean()

class SPINTrainer(Module):
    def __init__(
        self,
        model: Module,
        *,
        sft_dataset: Dataset,
        max_seq_len: int,
        accelerator: Optional[Accelerator] = None,
        accelerator_kwargs: dict = dict(),
        batch_size = 16,
        epochs = 2,
        start_learning_rate = 1e-6,
        end_learning_rate = 1e-7,
        learning_rate_num_decay_steps = 1000,
        weight_decay = 0.,
        adam_kwargs: dict = dict(),
        temperature = 0.7,
        nucleus_p = 0.9,
        pad_id: int = -1,
        spin_λ = 0.1,
        checkpoint_every = None,
        checkpoint_folder = './spin-checkpoints'
    ):
        super().__init__()

        self.accelerator = accelerator
        if not exists(self.accelerator):
            self.accelerator = Accelerator(**accelerator_kwargs)

        self.model = model
        self.epochs = epochs
        self.train_dataloader = DataLoader(sft_dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        self.optimizer = adam_optimizer_with_linear_decay(
            model,
            start_learning_rate,
            end_learning_rate,
            num_decay_steps = learning_rate_num_decay_steps,
            accelerator = self.accelerator,
            weight_decay = weight_decay,
            adam_kwargs = adam_kwargs
        )

        (
            self.model,
            self.train_dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader
        )

        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.nucleus_p = nucleus_p
        self.pad_id = pad_id

        self.spin_λ = spin_λ

        # checkpointing

        self.should_checkpoint = exists(checkpoint_every)
        self.checkpoint_every = checkpoint_every

        if self.should_checkpoint:
            self.checkpoint_folder = Path(checkpoint_folder)
            self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

        self.steps = 0

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def print(self, *msg):
        self.accelerator.print(*msg)

    def log(self, **data):
        self.accelerator.log(data, step = self.steps)

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def save(self, path: str, overwrite: bool = False):
        if not self.accelerator.is_main_process:
            return

        path = self.checkpoint_folder / path

        assert not path.exists() or overwrite, f'file already exists'

        pkg = dict(
            model = self.unwrapped_model.state_dict()
        )

        torch.save(pkg, str(path))

    def forward(self, overwrite_checkpoints: bool = True):
        """
        Algorithm 1 - https://arxiv.org/abs/2401.01335v1
        """

        self.steps = 0
        self.model.train()

        spin = SPIN(
            self.model,
            pad_id = self.pad_id,
            λ = self.spin_λ
        )

        for epoch in tqdm(range(self.epochs), desc = 'spin epoch'):
            for real_seq, prompt_len in tqdm(self.train_dataloader, desc = 'spin finetuning'):

                prompt_mask = prompt_mask_from_len(prompt_len, real_seq)
                prompts = real_seq[prompt_mask].split(prompt_len.tolist())

                generated_seqs = sample(
                    self.model,
                    prompts = prompts,
                    seq_len = self.max_seq_len,
                    temperature = self.temperature,
                    filter_fn = top_p,
                    filter_kwargs = dict(
                        thres = self.nucleus_p
                    ),
                    output_keep_prompt = True
                )

                spin_loss = spin(
                    real_seq = real_seq,
                    generated_seq = generated_seqs,
                    prompt_len = prompt_len
                )

                self.print(f'spin loss: {spin_loss.item():.3f}')
                self.log(loss = spin_loss.item())

                self.accelerator.backward(spin_loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.steps += 1

                self.wait()

                if self.should_checkpoint and not (self.checkpoint_every % self.steps):
                    checkpoint_num = self.steps // self.checkpoint_every
                    self.save(f'spin.ckpt.{checkpoint_num}.pt', overwrite = overwrite_checkpoints)

                self.wait()

        print(f'self-play training complete')
