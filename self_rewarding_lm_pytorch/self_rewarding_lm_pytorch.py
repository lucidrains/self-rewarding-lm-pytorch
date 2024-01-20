from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset

from numpy.lib.format import open_memmap

from beartype import beartype
from beartype.typing import Optional

from self_rewarding_lm_pytorch.dpo import DPO

from einops import rearrange

from accelerate import Accelerator

# helper

def exists(v):
    return v is not None

# constants
# llm-as-judge prompt
# https://openreview.net/forum?id=uccHPGDlao

DEFAULT_LLM_AS_JUDGE_PROMPT = """
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

# config, allowing for different types of reward prompting
# colocate with functions for extracting the response and reward

REWARD_PROMPT_CONFIG = dict(
    default = dict(
        prompt = DEFAULT_LLM_AS_JUDGE_PROMPT
    )
)

# sft trainer

class SFTTrainer(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        accelerator: Accelerator,
        train_dataset: Dataset,
        val_dataset: Dataset
    ):
        super().__init__()
        self.model = model

    def forward(self):
        raise NotImplementedError

# reward generator class


class RewardGenerator(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        reward_config: dict
    ):
        raise NotImplementedError

    def forward(self) -> Dataset:
        raise NotImplementedError

# fine tuning class

class SelfRewardingTrainer(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        train_sft_dataset: Optional[Dataset] = None,
        valid_sft_dataset: Optional[Dataset] = None,
        beta = 0.1,
        self_reward_num_iterations = 2,
        reward_prompt_config: dict = REWARD_PROMPT_CONFIG,
        reward_iteration_type = ['default', 'default'],
        accelerate_kwargs: dict = dict(),
        checkpoints_folder: str = './checkpoints',
        sft_trainer_kwargs: dict = dict(),
        dpo_trainer_kwargs: dict = dict()
    ):
        super().__init__()
        assert all([key in reward_prompt_config for key in reward_iteration_type]), f'reward prompt must be one of {reward_prompt_config.keys()}'

        self.reward_prompt_configs = [reward_prompt_config[key] for key in reward_iteration_type]
        self.self_reward_num_iterations = self_reward_num_iterations

        self.model = model
        self.first_iterate_on_sft = exists(train_sft_dataset)

        self.accelerator = Accelerator(**accelerate_kwargs)

        self.sft_trainer = None
        if self.first_iterate_on_sft:
            self.sft_trainer = SFTTrainer(
                model,
                accelerator = self.accelerator,
                train_dataset = train_sft_dataset,
                valid_dataset = valid_sft_dataset,
                **sft_trainer_kwargs
            )

        self.reward_generators = [RewardGenerator(model = model, reward_config = reward_config) for reward_config in self.reward_prompt_configs]

        self.dpo = DPO(model)

        self.dpo_trainer = DPOTrainer(
            dpo = self.dpo,
            accelerator = self.accelerator,
            **dpo_trainer_kwargs
        )

        checkpoints_folder = Path(checkpoints_folder)
        checkpoints_folder.mkdir(parents = True, exist_ok = True)
        self.checkpoints_folder = checkpoints_folder

    def save(self, path: str, overwrite: bool = False):
        if not self.accelerator.is_main_process:
            return

        path = self.checkpoints_folder / path

        assert not path.exists() or overwrite, f'file already exists'

        pkg = dict(
            model = self.model.state_dict()
        )

        torch.save(pkg, str(path))

    def forward(self):

        if self.first_iterate_on_sft:
            self.sft_trainer()
            self.save('sft.ckpt.pt')

        for ind, reward_generator in enumerate(range(self.reward_generators)):
            iterate_num = ind + 1

            self_reward_dataset = reward_generator()

            self.dpo_trainer(self_reward_dataset)

            self.dpo.update_reference_model_with_policy()

            self.save(f'self-reward.{iterate_num}.ckpt.pt')

        print(f'done')
