from pathlib import Path

from beartype import beartype
from beartype.typing import Optional, Callable, Union
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
    masked_mean,
    maybe_and_mask
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

from ema_pytorch import EMA

# helper functions

def exists(v):
    return v is not None

def log_prob_from_model_and_seq(model, seq):
    logits = model(seq)
    log_probs = logits.log_softmax(dim = -1)
    return get_at('b n [c], b n -> b n', log_probs, seq)

def prompt_mask_from_len(lengths, seq):
    seq_len, device = seq.shape[-1], seq.device
    return torch.arange(seq_len, device = device) < rearrange(lengths, '... -> ... 1')

# main class

class SPIN(Module):
    def __init__(
        self,
        model: Module,
        *,
        λ = 0.1,
        pad_id: Optional[int] = None,
        ref_model_ema_decay = 1.,
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.policy_model = model

        self.ref_model = EMA(
            model,
            beta = ref_model_ema_decay,
            **ema_kwargs
        )

        self.λ = λ
        self.pad_id = pad_id

    def update_reference_model_with_policy(self):
        self.ref_model.copy_params_from_model_to_ema()

    def update_ema(self):
        self.ref_model.update()

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

        real_prompt_mask = prompt_mask_from_len(prompt_len, real_seq)
        generated_prompt_mask = prompt_mask_from_len(prompt_len, generated_seq)

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
        model: Union[Module, SPIN],
        *,
        train_sft_dataset: Dataset,
        max_seq_len: int,
        valid_sft_dataset: Optional[Dataset] = None,
        valid_every = 100,
        accelerator: Optional[Accelerator] = None,
        accelerate_kwargs: dict = dict(),
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
        ref_model_ema_decay = 1.,
        checkpoint_every = None,
        checkpoint_folder = './spin-checkpoints',
        spin_kwargs: dict = dict(
            λ = 0.1,
        )
    ):
        super().__init__()

        self.accelerator = accelerator
        if not exists(self.accelerator):
            self.accelerator = Accelerator(**accelerate_kwargs)

        if not isinstance(model, SPIN):
            model = SPIN(
                model,
                pad_id = pad_id,
                ref_model_ema_decay = ref_model_ema_decay,
                **spin_kwargs
            )

        self.model = model
        self.epochs = epochs
        self.train_dataloader = DataLoader(train_sft_dataset, batch_size = batch_size, shuffle = True, drop_last = True)

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

        # validation

        self.valid_dataloader = None
        self.valid_every = valid_every

        if exists(valid_sft_dataset):
            self.valid_dataloader = DataLoader(valid_sft_dataset, batch_size = batch_size)

        # checkpointing

        self.should_checkpoint = exists(checkpoint_every)
        self.checkpoint_every = checkpoint_every

        if self.should_checkpoint:
            self.checkpoint_folder = Path(checkpoint_folder)
            self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

        self.steps = 0

    @property
    def is_main(self):
        return self.accelerator.is_main_process

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
        self.wait()

        if self.is_main:

            path = self.checkpoint_folder / path

            assert not path.exists() or overwrite, f'file already exists'

            pkg = dict(
                model = self.unwrapped_model.state_dict()
            )

            torch.save(pkg, str(path))

        self.wait()

    def calc_spin_loss(
        self,
        real_seq: TensorType['b', 'n', int],
        prompt_len: TensorType['b', int]
    ):
        prompt_mask = prompt_mask_from_len(prompt_len, real_seq)
        prompts = real_seq[prompt_mask].split(prompt_len.tolist())

        generated_seqs = sample(
            self.unwrapped_model.policy_model,
            prompts = prompts,
            seq_len = self.max_seq_len,
            temperature = self.temperature,
            filter_fn = top_p,
            filter_kwargs = dict(
                thres = self.nucleus_p
            ),
            output_keep_prompt = True
        )

        spin_loss = self.model(
            real_seq = real_seq,
            generated_seq = generated_seqs,
            prompt_len = prompt_len
        )

        return spin_loss

    def forward(self, overwrite_checkpoints: bool = True):
        """
        Algorithm 1 - https://arxiv.org/abs/2401.01335v1
        """

        self.model.update_reference_model_with_policy()

        self.steps = 0
        self.model.train()

        for epoch in tqdm(range(self.epochs), desc = 'spin epoch'):
            for real_seq, prompt_len in tqdm(self.train_dataloader, desc = 'spin finetuning'):

                self.model.train()

                train_loss = self.calc_spin_loss(real_seq, prompt_len)

                self.print(f'train spin loss: {train_loss.item():.3f}')
                self.log(loss = train_loss.item())

                self.accelerator.backward(train_loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.steps += 1

                self.wait()

                self.unwrapped_model.update_ema()

                self.wait()

                if exists(self.valid_dataloader) and not (self.valid_every % self.steps):
                    self.wait()

                    if self.is_main:
                        total_loss = 0.
                        total_batches = 0.

                        with torch.no_grad():
                            self.model.eval()

                            for valid_seq, prompt_len in tqdm(self.valid_dataloader, desc = 'valid spin'):
                                batch = valid_seq.shape[0]
                                valid_spin_loss = self.calc_spin_loss(valid_seq, prompt_len)

                                total_batches += batch
                                total_loss += valid_spin_loss * batch

                            valid_loss = total_loss / total_batches

                            self.print(f'valid spin loss: {valid_loss.item():.3f}')
                            self.log(valid_spin_loss = valid_loss.item())

                    self.wait()

                if self.should_checkpoint and not (self.checkpoint_every % self.steps):
                    checkpoint_num = self.steps // self.checkpoint_every
                    self.save(f'spin.ckpt.{checkpoint_num}.pt', overwrite = overwrite_checkpoints)

        self.print(f'self-play training complete')
