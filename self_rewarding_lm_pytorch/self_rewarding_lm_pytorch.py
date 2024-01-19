import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from self_rewarding_lm_pytorch.dpo import DPO

from einops import rearrange

from accelerate import Accelerator

# helper

def exists(v):
    return v is not None

# constants
# llm-as-judge prompt
# https://openreview.net/forum?id=uccHPGDlao

DEFAULT_REWARD_PROMPT = """
Review the user’s question and the corresponding response using the additive 5-point
scoring system described below. Points are accumulated based on the satisfaction of each
criterion:
- Add 1 point if the response is relevant and provides some information related to
the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question,
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a
useful way, regardless of whether it seems to have been written by an AI Assistant or if it
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective,
addressing the user’s question directly and comprehensively, and is well-organized and
helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question
by an AI Assistant, without extraneous information, reflecting expert knowledge, and
demonstrating a high-quality, engaging, and insightful answer.
User: <INSTRUCTION_HERE>
<response><RESPONSE_HERE></response>
After examining the user’s instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: “Score: <total points>”
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as
necessary. To evaluate the response in alignment with this additive scoring model, we’ll
systematically attribute points based on the outlined criteria.
"""

# fine tuning class

class SelfRewardingTrainer(Module):
    def __init__(
        self,
        model: Module,
        *,
        num_iterations = 3,
        beta = 0.1,
        reward_prompt = DEFAULT_REWARD_PROMPT
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.reward_prompt = reward_prompt

        self.model_with_dpo = DPO(model, beta = beta)

    def forward(self):
        raise NotImplementedError
