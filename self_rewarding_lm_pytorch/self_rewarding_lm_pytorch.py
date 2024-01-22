from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, ConcatDataset

from numpy.lib.format import open_memmap

from beartype import beartype
from beartype.typing import Optional, List, Union, Callable

from self_rewarding_lm_pytorch.dpo import (
    DPO,
    DPODataset,
    EarlyStopper,
    DPOTrainer,
    set_dropout_,
    adam_optimizer_with_linear_decay
)

from einops import rearrange

from accelerate import Accelerator

from pytorch_custom_utils.utils import pad_or_slice_to

from torchtyping import TensorType

from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# helper

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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

def default_parse_reward_fn(llm_response: str):
    result = re.search(r"Score: ([0-9\.]+)", a)
    if result.groups == 0:
        return None

    return result.groups(1)

def default_construct_reward_prompt(instruction: str, response: str):
    return f"""

User: {instruction}
<response>{response}</response>
"""

# config, allowing for different types of reward prompting
# colocate with functions for extracting the response and reward

REWARD_PROMPT_CONFIG = dict(
    default = dict(
        prompt = DEFAULT_LLM_AS_JUDGE_PROMPT,
        parse_reward = default_parse_reward_fn,
        construct_reward_prompt = default_construct_reward_prompt
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
        train_dataset: Union[List[Dataset], Dataset],
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        num_epochs: int = 3,
        start_learning_rate: float = 5.5e-6,
        end_learning_rate: float = 1.1e-6,
        weight_decay: float = 0.,
        ignore_index: int = -1,
        adam_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = accelerator
        self.model = model

        self.num_epochs = num_epochs
        self.ignore_index = ignore_index

        if isinstance(train_dataset, list):
            train_dataset = ConcatDataset(train_dataset)

        self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        self.model, self.train_dataloader = self.accelerator.prepare(self.model, self.train_dataloader)

        self.optimizer = adam_optimizer_with_linear_decay(
            model,
            start_learning_rate,
            end_learning_rate,
            accelerator = accelerator,
            weight_decay = weight_decay,
            adam_kwargs = adam_kwargs
        )

        self.val_dataloader = None
        if exists(val_dataset):
            self.val_dataloader = DataLoader(val_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

    def forward(self):
        for epoch in self.num_epochs:
            for seq in self.train_dataloader:
                seq, labels = seq[: :-1], seq[:, 1:]

                logits = self.model(seq)

                ce_loss = F.cross_entropy(
                    rearrange(logits, 'b n l -> b l n'),
                    labels,
                    ignore_index = self.ignore_index
                )

                self.accelerator.backward(ce_loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

# reward generator class

class RewardGenerator(Module):
    @beartype
    def __init__(
        self,
        model_generate_prompt: Module,
        model: Module,
        num_preference_pairs: int,
        tokenizer_encode: Callable[[str], TensorType['seq_len', int]],
        tokenizer_decode: Callable[[TensorType['seq_len', int]], str],
        batch_size: int = 16,
        num_candidate_responses: int = 4,
        gen_temperature: float = 0.7,
        gen_nucleus_p: float = 0.9,
        eval_temperature: float = 0.7,
        eval_nucleus_p: float = 0.9,
        num_evals_to_average: int = 3,
        *,
        reward_config: dict,
        preference_seq_memmap_file: str,
        prompt_len_memmap_file: str,
        preference_max_seq_len: int = 4096,
        pad_id: int = -1
    ):
        super().__init__()
        self.model_generate_prompt = model_generate_prompt

        self.model = model
        self.num_candidate_responses = num_candidate_responses
        self.reward_config = reward_config

        self.batch_size = batch_size

        self.gen_nucleus_p = gen_nucleus_p
        self.gen_temperature = gen_temperature

        self.eval_nucleus_p = eval_nucleus_p
        self.eval_temperature = eval_temperature

        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

        self.num_evals_to_average = num_evals_to_average

        self.num_preference_pairs = num_preference_pairs

        self.preference_seq_memmap_file = preference_seq_memmap_file
        self.prompt_len_memmap_file = prompt_len_memmap_file

        self.preference_max_seq_len = preference_max_seq_len

        self.pad_id = pad_id

        memmap_shape = (num_preference_pairs, 2, preference_max_seq_len)

        self.preference_seq_memmap = open_memmap(preference_seq_memmap_file, dtype = 'int', mode = 'w+', shape = memmap_shape)
        self.prompt_len_memmap = open_memmap(prompt_len_memmap, dtype = 'int', mode = 'w+', shape = (num_preference_pairs,))

    def generate_reward(
        self,
        prompt: str,
        response: str
    ) -> float:

        """
        main contribution of the paper is the logic in this function
        in paper, they sample it 4 times and then average
        """

        device = next(self.model.parameters()).device

        parse_reward = self.reward_config['parse_reward']
        reward_base_prompt_str = self.reward_config['prompt']
        construct_reward_prompt = self.reward_config['construct_reward_prompt']

        reward_prompt = reward_base_prompt_str + construct_reward_prompt(prompt, response)
        reward_prompt = self.tokenizer_encode(reward_prompt_str).to(device)

        reward_prompt = repeat(reward_prompt, 'n -> b n', b = self.num_evals_to_average)

        wrapped_model = AutoregressiveWrapper(self.model)

        reward_responses = wrapped_model.generate(
            prompt = reward_prompt,
            temperature = self.eval_temperature,
            filter_kwargs = dict(
                thres = self.eval_nucleus_p
            )
        )

        reward_responses_as_str: List[str] = [self.tokenizer_decode(resp[resp != self.pad_id].cpu()) for resp in reward_responses]
        rewards: List[Optional[float]] = [parse_reward(resp_str) for resp_str in reward_responses_as_str]

        rewards = [*filter(exists, rewards)] # for now, just filter out any failed responses

        avg_reward = Tensor(rewards).mean()
        return avg_reward

    def forward(self) -> DPODataset:

        raise NotImplementedError

        self.prompt_len_memmap.flush()
        self.preference_seq_memmap.flush()

        dpo_dataset = DPODataset(self.preference_seq_memmap_file, self.prompt_len_memmap_file)
        return dpo_dataset

# fine tuning class

class SelfRewardingTrainer(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        train_sft_dataset: Optional[Union[List[Dataset], Dataset]] = None,
        valid_sft_dataset: Optional[Dataset] = None,
        beta = 0.1,
        self_reward_num_iterations = 2,
        reward_prompt_config: dict = REWARD_PROMPT_CONFIG,
        reward_iteration_type = ['default', 'default'],
        num_preference_pairs: List[int] = [3964, 6942],
        reward_generator_kwargs: dict = dict(
            num_candidate_responses = 4,
            gen_temperature = 0.7,
            gen_nucleus_p = 0.9,
            eval_temperature = 0.7,
            eval_nucleus_p = 0.9
        ),
        model_generate_prompt: Optional[Module] = None,
        early_stopper: Optional[EarlyStopper] = None,
        accelerate_kwargs: dict = dict(),
        sft_trainer_kwargs: dict = dict(),
        dpo_trainer_kwargs: dict = dict(),
        dropout = 0.1,
        checkpoints_folder: str = './checkpoints'
    ):
        super().__init__()
        assert all([key in reward_prompt_config for key in reward_iteration_type]), f'reward prompt must be one of {reward_prompt_config.keys()}'

        if not exists(model_generate_prompt):
            model_generate_prompt = deepcopy(model)

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

        self.reward_generators = [RewardGenerator(model = model, model_generate_prompt = model_generate_prompt, reward_config = reward_config, num_preference_pairs = one_stage_num_preference_pairs, **reward_generator_kwargs) for reward_config, one_stage_num_preference_pairs in zip(self.reward_prompt_configs, num_preference_pairs)]

        set_dropout_(model, dropout)
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
