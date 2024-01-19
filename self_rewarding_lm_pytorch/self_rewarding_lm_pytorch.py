import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from self_rewarding_lm_pytorch.dpo import DPO

from einops import rearrange

from accelerate import Accelerator

# helper

def exists(v):
    return v is not None

# fine tuning class

class SelfRewardingTrainer(Module):
    def __init__(
        self,
        model: Module,
        *,
        num_iterations = 3,
        beta = 0.1
    ):
        super().__init__()
        self.num_iterations = num_iterations

        self.model_with_dpo = DPO(model, beta = beta)

    def forward(self):
        raise NotImplementedError
