from pathlib import Path
from copy import deepcopy
from collections import namedtuple
from dataclasses import dataclass

from beartype import beartype
from beartype.typing import Optional, Callable
from torchtyping import TensorType

import torch
from torch.nn import Module, Dropout
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from accelerate import Accelerator

from einops import rearrange
from einx import get_at

from numpy.lib.format import open_memmap

from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule
)

from pytorch_custom_utils.accelerate_utils import (
    model_forward_contexts
)

from pytorch_custom_utils.utils import (
    masked_mean
)

from tqdm import tqdm

# helper functions

def exists(v):
    return v is not None

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

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

def set_dropout_(model: Module, prob: float):
    for module in model.modules():
        if isinstance(module, Dropout):
            module.p = prob

def adam_optimizer_with_linear_decay(
    model: Module,
    start_learning_rate: float,
    end_learning_rate: float,
    num_decay_steps: int,
    accelerator: Accelerator,
    weight_decay: float,
    adam_kwargs: dict = dict(),
) -> OptimizerWithWarmupSchedule:

    adam = get_adam_optimizer(
        model.parameters(),
        lr = start_learning_rate,
        wd = weight_decay
    )

    scheduler = None
    if start_learning_rate != end_learning_rate:
        scheduler = LinearLR

    return OptimizerWithWarmupSchedule(
        optimizer = adam,
        accelerator = accelerator,
        scheduler = LinearLR,
        scheduler_kwargs = dict(
            start_factor = 1.,
            end_factor = end_learning_rate / start_learning_rate,
            total_iters = num_decay_steps
        )
    )

# early stopping

@dataclass
class EarlyStopperReturn:
    should_stop: bool
    score: float

class EarlyStopper(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        calculate_should_stop: Callable[..., bool] = lambda past_scores, score: len(past_scores) > 0 and score < past_scores[-1],
        early_stop_checkpoint_folder: str = './early-stop-checkpoint'
    ):
        super().__init__()
        self.model = model
        self.scores = []
        self.calculate_should_stop = calculate_should_stop

        self.val_dl = DataLoader(dataset, batch_size  = batch_size, shuffle = True, drop_last = True)

        self.early_stop_checkpoint_folder = Path(early_stop_checkpoint_folder)
        self.early_stop_checkpoint_folder.mkdir(exist_ok = True, parents = True)

    def save(self, path: str, overwrite: bool = False):
        if not self.accelerator.is_main_process:
            return

        path = self.checkpoints_folder / path

        assert not path.exists() or overwrite, f'file already exists'

        pkg = dict(
            model = self.model.state_dict()
        )

        torch.save(pkg, str(path))

    @torch.no_grad()
    def forward(self) -> EarlyStopperReturn:
        self.model.eval()

        raise NotImplementedError

        should_stop = self.calculate_should_stop(self.scores, score)
        self.scores.append(score)

        if should_stop:
            prev_checkpoint_filename = f'model.ckpt.{len(self.scores) - 1}.pt'
            ckpt_path = self.early_stop_checkpoint_folder / prev_checkpoint_filename

            pkg = torch.load(str(ckpt_path))

            self.model.load_state_dict(pkg['model'])
        else:
            checkpoint_filename = f'model.ckpt.{len(self.scores)}.pt'
            ckpt_path = self.early_stop_checkpoint_folder / checkpoint_filename
            self.save(str(ckpt_path))

        return EarlyStopperReturn(score, should_stop)

# dataset from two memmap numpy file

# preferred and unpreferred sequences of shape - (<num samples>, <preference (2) - preferred followed by unpreferred>, <seq length>)
# prompt length (<num samples>,)

class DPODataset(Dataset):
    def __init__(
        self,
        data_folder: str = './',
        preference_seq_memmap_file: str = 'preference_seq.memmap.npy',
        prompt_len_memmap_file: str = 'prompt_len.memmap.npy',
    ):
        self.data_folder = Path(data_folder)
        assert self.data_folder.exists() and self.data_folder.is_dir()

        preference_seq_memmap_path = self.data_folder / preference_seq_memmap_file
        prompt_len_memmap_path = self.data_folder / prompt_len_memmap_file

        assert preference_seq_memmap_path.exists()
        assert prompt_len_memmap_path.exists()

        self.paired_sequences = open_memmap(str(preference_seq_memmap_path), dtype = 'int', mode = 'r')
        self.prompt_len = open_memmap(str(prompt_len_memmap_path), dtype = 'int', mode = 'r')

        self.seq_len = self.paired_sequences.shape[1]
        assert self.paired_sequences.shape[0] == self.prompt_len.shape[0]

    def __len__(self):
        return self.paired_sequences.shape[0]

    def __getitem__(self, idx):
        sequences = self.paired_sequences[idx].copy()
        prompt_lens = self.prompt_len[idx].copy()

        preferred_seq, unpreferred_seq = sequences

        return preferred_seq, unpreferred_seq, prompt_lens

# main class

class DPO(Module):
    def __init__(
        self,
        model: Module,
        *,
        beta = 0.1,
        pad_id: Optional[int] = None
    ):
        super().__init__()
        self.policy_model = model

        self.ref_model = deepcopy(model)
        freeze_all_layers_(self.ref_model)

        self.beta = beta
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
        preferred_seq: TensorType['b', 'n', int],
        unpreferred_seq: TensorType['b', 'n', int],
        prompt_len: TensorType['b', int],
        preferred_seq_mask: Optional[TensorType['b', 'n', bool]] = None,
        unpreferred_seq_mask: Optional[TensorType['b', 'n', bool]] = None
    ):
        self.policy_model.train()

        """
        b - batch
        n - sequence length
        """

        preferred_prompt_mask = prompt_mask_from_len(prompt_len, preferred_seq)
        unpreferred_prompt_mask = prompt_mask_from_len(prompt_len, unpreferred_seq)

        """
        Following Appendix B in https://arxiv.org/abs/2305.18290
        """

        if exists(self.pad_id):
            assert not exists(preferred_seq_mask)
            assert not exists(unpreferred_seq_mask)
            preferred_seq_mask = preferred_seq != self.pad_id
            preferred_seq.masked_fill_(~preferred_seq_mask, 0)

            unpreferred_seq_mask = unpreferred_seq != self.pad_id
            unpreferred_seq.masked_fill_(~unpreferred_seq_mask, 0)            

        with torch.no_grad():
            self.ref_model.eval()
            ref_preferred_logprob = log_prob_from_model_and_seq(self.ref_model, preferred_seq)
            ref_unpreferred_logprob = log_prob_from_model_and_seq(self.ref_model, unpreferred_seq)

        policy_preferred_logprob = log_prob_from_model_and_seq(self.policy_model, preferred_seq)
        policy_unpreferred_logprob = log_prob_from_model_and_seq(self.policy_model, unpreferred_seq)

        # masked mean

        policy_preferred_logprob, ref_preferred_logprob = [masked_mean(seq, maybe_and_mask(preferred_seq_mask, ~preferred_prompt_mask)) for seq in (policy_preferred_logprob, ref_preferred_logprob)]
        policy_unpreferred_logprob, ref_unpreferred_logprob = [masked_mean(seq, maybe_and_mask(unpreferred_seq_mask, ~unpreferred_prompt_mask)) for seq in (policy_unpreferred_logprob, ref_unpreferred_logprob)]

        # DPO loss

        policy_logratios = policy_preferred_logprob - policy_unpreferred_logprob
        ref_logratios = ref_preferred_logprob - ref_unpreferred_logprob

        losses = -F.logsigmoid(self.beta * (policy_logratios - ref_logratios))

        return losses.mean()

# trainer class

class DPOTrainer(Module):
    @beartype
    def __init__(
        self,
        dpo: DPO,
        *,
        accelerator: Optional[Accelerator],
        accelerate_kwargs: dict = dict(),
        batch_size: int = 16,
        num_decay_steps: int = 1000,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.,
        val_dataset: Optional[Dataset] = None,
        start_learning_rate: float = 1e-6,
        end_learning_rate: float = 1e-7,
        adam_kwargs: dict = dict(),
        early_stopper: Optional[EarlyStopper] = None,
        dropout: float = 0.1,
        check_early_stop_every: int = 200
    ):
        super().__init__()
        set_dropout_(dpo, dropout)

        if not exists(accelerator):
            accelerator = Accelerator(**accelerate_kwrags)

        self.accelerator = accelerator

        self.model = accelerator.prepare(dpo)

        self.batch_size = batch_size

        self.optimizer = adam_optimizer_with_linear_decay(
            dpo,
            start_learning_rate,
            end_learning_rate,
            num_decay_steps = num_decay_steps,
            accelerator = accelerator,
            weight_decay = weight_decay,
            adam_kwargs = adam_kwargs
        )

        self.early_stopper = None
        if self.is_main:
            self.early_stopper = early_stopper

        self.check_early_stop_every = check_early_stop_every

        self.val_dataloader = None
        if exists(val_dataset):
            self.val_dataloader = DataLoader(val_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        self.steps = 0
        self.register_buffer('break_signal', torch.tensor(0.))

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def log(self, **data):
        self.accelerator.log(data, step = self.steps)

    def forward(
        self,
        train_self_reward_dataset: Dataset
    ):
        train_dataloader = DataLoader(train_self_reward_dataset, batch_size = self.batch_size, drop_last = True, shuffle = True)
        train_dataloader = self.accelerator.prepare(train_dataloader)

        iter_dl = cycle(train_dataloader)

        pbar = tqdm(desc = 'dpo finetuning')

        while True:
            self.model.train()

            batch = next(iter_dl)

            dpo_loss = self.model(*batch)
            self.accelerator.backward(dpo_loss)

            self.log(loss = dpo_loss.item())

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1
            pbar.update(1)
            self.wait()

            if not (self.steps % self.check_early_stop_every) and exists(self.early_stopper):

                early_stop_return = self.early_stopper()

                self.log(dpo_valid_score = early_stop_return.score)

                if self.is_main and early_stop_return.should_stop:
                    self.break_signal.copy_(1.)
                    dist.all_reduce(self.break_signal)

                if self.break_signal.item() == 1:
                    break

            self.wait()

        pbar.close()
        print('dpo training finished')
