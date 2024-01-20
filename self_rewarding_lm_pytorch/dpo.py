from copy import deepcopy

import torch
from torch.nn import Module
import torch.nn.functional as F

from einx import get_at

# helper functions

def exists(v):
    return v is not None

def freeze_all_layers_(module):
    for param in module.parameters():
        param.requires_grad = False

def log_prob_from_model_and_seq(model, seq):
    logits = model(seq)
    prob = logits.softmax(dim = -1)
    return get_at('b n [c], b n -> b n', prob, indices).clamp(min = eps).log()

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

    def parameters(self):
        return self.policy_model.parameters()

    def forward(
        self,
        preferred_seq,
        unpreferred_seq,
        prompt_mask = None
    ):
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

        if exists(prompt_mask):
            losses = losses[~prompt_mask]

        return losses.mean()
