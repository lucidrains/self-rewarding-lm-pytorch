from copy import deepcopy
from collections import namedtuple

import torch
from torch.nn import Module, Dropout
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator

from beartype import beartype
from beartype.typing import Optional, Callable

from einx import get_at

from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule
)

from pytorch_custom_utils.accelerate_utils import (
    auto_unwrap_model,
    model_forward_contexts
)

from torchtyping import TensorType

# helper functions

def exists(v):
    return v is not None

def freeze_all_layers_(module):
    for param in module.parameters():
        param.requires_grad = False

def log_prob_from_model_and_seq(model, seq, eps = 1e-20):
    logits = model(seq)
    prob = logits.softmax(dim = -1)
    return get_at('b n [c], b n -> b n', prob, indices).clamp(min = eps).log()

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

EarlyStopperReturn = namedtuple('EarlyStopperReturn', ['should_stop', 'score'])

class EarlyStopper(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        calculate_should_stop: Callable[..., bool] = lambda past_scores, score: len(past_scores) > 0 and score < past_scores[-1]
    ):
        super().__init__()
        self.model = model
        self.scores = []
        self.calculate_should_stop = calculate_should_stop

        self.val_dl = DataLoader(dataset, batch_size  = batch_size, shuffle = True, drop_last = True)

    @torch.no_grad()
    def forward(self) -> EarlyStopperReturn:
        self.model.eval()

        raise NotImplementedError

        should_stop = self.calculate_should_stop(self.scores, score)
        self.scores.append(score)

        return EarlyStopperReturn(score, should_stop)

# main class

class DPO(Module):
    def __init__(
        self,
        model: Module,
        *,
        beta = 0.1
    ):
        super().__init__()
        self.policy_model = model

        self.ref_model = deepcopy(model)
        freeze_all_layers_(self.ref_model)

        self.beta = beta

    def update_reference_model_with_policy(self):
        self.ref_model.load_state_dict(self.policy_model.state_dict())

    def parameters(self):
        return self.policy_model.parameters()

    @autocast(enabled = False)
    def forward(
        self,
        preferred_seq: TensorType['b', 'n', int],
        unpreferred_seq: TensorType['b', 'n', int],
        prompt_mask: Optional[TensorType['b', 'n', bool]] = None,
        preferred_seq_mask: Optional[TensorType['b', 'n', bool]] = None,
        unpreferred_seq_mask: Optional[TensorType['b', 'n', bool]] = None
    ):
        """
        b - batch
        n - sequence length
        """

        assert preferred_seq.ndim == 2
        assert preferred_seq.shape == unpreferred_seq.shape

        """
        Following Appendix B in https://arxiv.org/abs/2305.18290
        """

        with torch.no_grad():
            self.ref_model.eval()
            ref_preferred_logprob = log_prob_from_model_and_seq(self.ref_model, preferred_seq)
            ref_unpreferred_logprob = log_prob_from_model_and_seq(self.ref_model, unpreferred_seq)

        policy_preferred_logprob = log_prob_from_model_and_seq(self.policy_model, preferred_seq)
        policy_unpreferred_logprob = log_prob_from_model_and_seq(self.policy_model, unpreferred_seq)

        policy_logratios = policy_preferred_logprob - policy_unpreferred_logprob
        ref_logratios = ref_preferred_logprob - ref_unpreferred_logprob

        losses = -F.logsigmoid(self.beta * (policy_logratios - ref_logratios))

        loss_mask = maybe_and_mask(preferred_seq_mask, unpreferred_seq, ~prompt_mask)

        if exists(loss_mask):
            losses = losses[loss_mask]

        return losses.mean()

# trainer class

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

class DPOTrainer(Module):
    @beartype
    def __init__(
        self,
        dpo: DPO,
        *,
        accelerator: Accelerator,
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

        self.early_stopper = early_stopper
        self.check_early_stop_every = check_early_stop_every

        self.val_dataloader = None
        if exists(val_dataset):
            self.val_dataloader = DataLoader(val_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        self.steps = 0

    def forward(
        self,
        train_self_reward_dataset: Dataset
    ):
        train_dataloader = DataLoader(train_self_reward_dataset, batch_size = self.batch_size, drop_last = True, shuffle = True)
        iter_dl = cycle(train_dataloader)

        while True:
            self.model.train()

            batch = next(iter_dl)

            dpo_loss = self.model(batch)
            self.accelerator.backward(dpo_loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1

            if not (self.steps % self.check_early_stop_every):
                if self.early_stopper():
                    break

        raise NotImplementedError
