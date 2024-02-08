import re
import sys
from functools import partial
from random import randrange
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field
from functools import wraps
from textwrap import dedent

from beartype import beartype
from beartype.typing import Optional, Dict, List, Tuple, Union, Callable
from torchtyping import TensorType

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Dropout
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from numpy.lib.format import open_memmap

from self_rewarding_lm_pytorch.dpo import (
    DPO,
    DPODataset,
    DPOTrainer,
    EarlyStopper,
    set_dropout_,
    adam_optimizer_with_linear_decay
)

from self_rewarding_lm_pytorch.spin import (
    SPIN,
    SPINTrainer
)

from einops import rearrange, repeat

from accelerate import Accelerator

from pytorch_custom_utils.utils import pad_or_slice_to

from pytorch_custom_utils.accelerate_utils import (
    model_forward_contexts
)

from self_rewarding_lm_pytorch.sampling_utils import (
    sample,
    top_p,
    top_k
)

from self_rewarding_lm_pytorch.mocks import always

from tqdm import tqdm

# warning

if sys.maxsize <= (2 ** 32):
    print('you need to be on 64 bit system to use memmapped files of > 2GB')

# basic templating engine

import jinja2
jinja2_env = jinja2.Environment()

def find_variables_from_jinja_template(template: str):
    from jinja2 import meta
    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)

# helper

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

def identity(t, *args, **kwargs):
    return t

def prompt_mask_from_len(length, seq):
    seq_len, device = seq.shape[-1], seq.device
    return torch.arange(seq_len, device = device) < rearrange(length, '... -> ... 1')

def cast_tuple(t, length = 1, validate = False):
    out = t if isinstance(t, tuple) else ((t,) * length)
    assert not validate or len(out) == length
    return out

def cast_input(cast_fn):
    def decorator(fn):
        @wraps(fn)
        def inner(t, *args, **kwargs):
            t = cast_fn(t)
            return fn(t, *args, **kwargs)
        return inner

    return decorator

def cast_output(cast_fn):
    def decorator(fn):
        @wraps(fn)
        def output(*args, **kwargs):
            out = fn(*args, **kwargs)
            out = cast_fn(out)
            return out
        return output

    return decorator

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
User: {{ prompt }}
<response>{{ response }}</response>
After examining the user’s instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: “Score: <total points>”
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as
necessary. To evaluate the response in alignment with this additive scoring model, we’ll
systematically attribute points based on the outlined criteria.
"""

DEFAULT_REWARD_REGEX_TEMPLATE = """
Score: {{ reward }}
"""

def create_parse_reward_fn(reward_regex_template):
    assert find_variables_from_jinja_template(reward_regex_template) == {'reward'}, 'reward template must include "score" variable'
    reward_regex_str = jinja2_env.from_string(reward_regex_template).render(reward = "([0-9\.]+)")

    # @always(lambda: randrange(0, 10))
    def parse_reward_fn(llm_response: str) -> float:
        result = re.search(rf"{reward_regex_str}", llm_response)

        if not exists(result) or result.groups == 0:
            return None

        if not result.groups(1).isnumeric():
            return None

        return float(result.groups(1))

    return parse_reward_fn

# reward config

@dataclass
class RewardConfig:
    prompt_template: str
    reward_regex_template: Optional[str] = None
    parse_reward: Optional[Callable[[str], Optional[float]]] = None
    template_fn: Optional[Callable[..., str]] = None
    auto_dedent: bool = True

    def init(self):

        # maybe dedent

        if self.auto_dedent:
            self.prompt_template = dedent(self.prompt_template)

            if exists(self.reward_regex_template):
                self.reward_regex_template = dedent(self.reward_regex_template)

        # initialize render function for prompt and response template

        prompt_template = self.prompt_template
        assert find_variables_from_jinja_template(prompt_template) == {'prompt', 'response'}, 'template must include prompt and response templating variables'
        self.template_fn = jinja2_env.from_string(prompt_template).render

        # initialize the parse_reward if only the reward regex template is given

        if not exists(self.parse_reward):
            assert exists(self.reward_regex_template), 'reward_regex_template must be given if parse_reward is not passed in'
            self.parse_reward = create_parse_reward_fn(self.reward_regex_template)

        return self

# config, allowing for different types of reward prompting
# colocate with functions for extracting the response and reward

SELF_REWARD_PROMPT_CONFIG = dict(
    default = RewardConfig(
        prompt_template = DEFAULT_LLM_AS_JUDGE_PROMPT,
        reward_regex_template = DEFAULT_REWARD_REGEX_TEMPLATE
    )
)

default_is_valid_reward_pair = lambda preferred_reward, unpreferred_reward: (preferred_reward != unpreferred_reward).all()

@beartype
def default_pick_paired_rewards_fn(rewards: Tensor):
    is_nan_mask = torch.isnan(rewards)
    rewards_max, rewards_min = rewards.clone(), rewards.clone()
    rewards_max[is_nan_mask] = -1e6
    rewards_min[is_nan_mask] = 1e6
    return torch.stack((rewards_max.argmax(dim = -1), rewards_min.argmin(dim = -1)))

# sft trainer

class SFTTrainer(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        accelerator: Accelerator,
        train_dataset: Union[List[Dataset], Dataset],
        valid_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        grad_accum_steps: int = 2,
        num_epochs: int = 3,
        start_learning_rate: float = 5.5e-6,
        end_learning_rate: float = 1.1e-6,
        learning_rate_num_decay_steps: Optional[int] = None,
        dropout: float = 0.,
        weight_decay: float = 0.,
        ignore_index: int = -1,
        adam_kwargs: dict = dict(),
        valid_every: int = 1
    ):
        super().__init__()
        self.accelerator = accelerator
        self.model = model
        self.dropout = dropout

        self.num_epochs = num_epochs
        self.ignore_index = ignore_index

        if isinstance(train_dataset, list):
            train_dataset = ConcatDataset(train_dataset)

        self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        self.num_train_steps = len(self.train_dataloader) // grad_accum_steps * num_epochs
        self.grad_accum_steps = grad_accum_steps

        (
            self.model,
            self.train_dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader
        )

        if not exists(learning_rate_num_decay_steps):
            # default learning rate decay num steps to half of training dataset length
            learning_rate_num_decay_steps = len(train_dataset) // 2

        self.optimizer = adam_optimizer_with_linear_decay(
            model,
            start_learning_rate,
            end_learning_rate,
            num_decay_steps = learning_rate_num_decay_steps,
            accelerator = accelerator,
            weight_decay = weight_decay,
            adam_kwargs = adam_kwargs
        )

        self.valid_every = valid_every

        self.valid_dataloader = None
        if exists(valid_dataset):
            self.valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size)

        self.steps = 0

    def log(self, **data):
        self.accelerator.log(data, step = self.steps)

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def get_cross_entropy_loss(
        self,
        seq: TensorType['batch', 'seq', int],
        prompt_len_or_mask: Union[
            TensorType['batch', int],
            TensorType['batch', 'seq', bool]
        ]
    ):
        if prompt_len_or_mask.dtype == torch.long:
            prompt_mask = prompt_mask_from_len(prompt_len_or_mask, seq)
        else:
            prompt_mask = prompt_len_or_mask

        seq, labels = seq[:, :-1], seq[:, 1:]

        labels.masked_fill_(prompt_mask[:, 1:], self.ignore_index)

        logits = self.model(seq)

        return F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels,
            ignore_index = self.ignore_index
        )

    def forward(self):

        train_dl_iter = cycle(self.train_dataloader)

        set_dropout_(self.model, self.dropout)

        for _ in tqdm(range(self.num_train_steps), desc = 'sft fine-tuning'):
            self.model.train()

            for forward_context in model_forward_contexts(self.accelerator, self.model, self.grad_accum_steps):
                with forward_context():
                    seq, prompt_len_or_mask = next(train_dl_iter)

                    loss = self.get_cross_entropy_loss(seq, prompt_len_or_mask)

                    self.accelerator.backward(loss / self.grad_accum_steps)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.log(loss = loss.item())

            self.steps += 1

            if exists(self.valid_dataloader) and not (step % self.valid_every):
                self.wait()

                if self.accelerator.is_main_process:
                    total_valid_loss = 0.
                    total_batches = 0.

                    self.model.eval()

                    with torch.no_grad():
                        for seq, prompt_len_or_mask in self.valid_dataloader:
                            batch = seq.shape[0]

                            loss = self.get_cross_entropy_loss(seq, prompt_len_or_mask)

                            total_valid_loss += loss.item() * batch
                            total_batches += batch

                    valid_loss = total_valid_loss / total_batches

                    self.log(valid_loss = valid_loss)

# reward generator class

class DPODatasetGenerator(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        prompt_dataset: Dataset,
        num_preference_pairs: int,
        accelerator: Accelerator,
        tokenizer_encode: Callable[[str], TensorType['seq', int]],
        tokenizer_decode: Callable[[TensorType['seq', int]], str],
        self_reward_model: Optional[Module] = None,
        batch_size: int = 16,
        num_candidate_responses: int = 4,
        gen_temperature: float = 0.7,
        gen_filter_fn = top_p,
        gen_filter_kwargs: dict = dict(thres = 0.9),
        eval_temperature: float = 0.7,
        eval_filter_fn = top_p,
        eval_filter_kwargs: dict = dict(thres = 0.9),
        num_evals_to_average: int = 3,
        *,
        reward_config: RewardConfig,
        reward_model: Optional[Module] = None,
        data_folder: str = './',
        preference_seq_memmap_file: str = 'preference_seq.memmap.npy',
        prompt_len_memmap_file: str = 'prompt_len.memmap.npy',
        self_reward_memmap_file: str = 'self_reward.memmap.npy',
        preference_max_seq_len: int = 1024,
        generate_reward_max_seq_len: int = 256,
        is_valid_reward: Callable[float, bool] = lambda *args: True,
        is_valid_reward_pair: Optional[Callable[[float, float], bool]] = None,
        pick_paired_rewards: Callable[[Tensor], Tensor] = default_pick_paired_rewards_fn,
        pad_id: int = -1
    ):
        super().__init__()

        self.model = model
        self.num_candidate_responses = num_candidate_responses

        self.self_reward_model = default(self_reward_model, model)
        self.reward_config = reward_config.init()

        self.batch_size = batch_size
        self.prompt_dataset = prompt_dataset
        self.prompt_dataloader = DataLoader(prompt_dataset, batch_size = batch_size, shuffle = True)

        self.gen_filter_fn = gen_filter_fn
        self.gen_filter_kwargs = gen_filter_kwargs
        self.gen_temperature = gen_temperature

        self.eval_filter_fn = eval_filter_fn
        self.eval_filter_kwargs = eval_filter_kwargs
        self.eval_temperature = eval_temperature

        self.tokenizer_encode = cast_output(lambda t: t.long())(tokenizer_encode)
        self.tokenizer_decode = cast_input(lambda t: t.long() if torch.is_tensor(t) else [*map(int, t)])(tokenizer_decode)

        self.num_evals_to_average = num_evals_to_average

        # logic for sampling the reward pair and to validate it before adding it to generated preference dataset

        self.is_valid_reward = is_valid_reward
        self.is_valid_reward_pair = default(is_valid_reward_pair, lambda *args: True)
        self.pick_paired_rewards = pick_paired_rewards

        # prepare external reward model, if passed in

        self.has_external_reward_model = exists(reward_model)
        self.reward_model = reward_model

        # shapes and padding

        self.generate_reward_max_seq_len = generate_reward_max_seq_len

        self.num_preference_pairs = num_preference_pairs

        self.preference_max_seq_len = preference_max_seq_len

        self.pad_id = pad_id

        memmap_shape = (num_preference_pairs, 2, preference_max_seq_len)

        # save for returning instance of DPO Dataset at the end

        self.dpo_dataset_kwargs = dict(
            data_folder = data_folder,
            preference_seq_memmap_file = preference_seq_memmap_file,
            prompt_len_memmap_file = prompt_len_memmap_file
        )

        # the memmap npy files

        self.data_folder_path = Path(data_folder)
        self.data_folder_path.mkdir(exist_ok = True, parents = True)

        self.preference_seq_memmap_path = self.data_folder_path / preference_seq_memmap_file
        self.prompt_len_memmap_path = self.data_folder_path / prompt_len_memmap_file
        self.self_reward_mmemap_path = self.data_folder_path / self_reward_memmap_file

        self.preference_seq_memmap = open_memmap(str(self.preference_seq_memmap_path), dtype = 'int', mode = 'w+', shape = memmap_shape)
        self.prompt_len_memmap = open_memmap(str(self.prompt_len_memmap_path), dtype = 'int', mode = 'w+', shape = (num_preference_pairs,))
        self.self_reward_memmap_file = open_memmap(str(self.self_reward_mmemap_path), dtype = 'float32', mode = 'w+', shape = (num_preference_pairs, 2))

        self.accelerator = accelerator

    @property
    def device(self):
        return self.accelerator.device

    def generate_reward(
        self,
        prompt: str,
        response: str
    ) -> Optional[float]:

        """
        main contribution of the paper is the logic in this function
        in paper, they sample it 3 times and then average
        """

        device = next(self.model.parameters()).device

        template_fn = self.reward_config.template_fn
        parse_reward = self.reward_config.parse_reward

        reward_prompt_str = template_fn(prompt = prompt, response = response)
        reward_prompt = self.tokenizer_encode(reward_prompt_str).to(device)

        reward_prompt = repeat(reward_prompt, 'n -> b n', b = self.num_evals_to_average)

        reward_prompt = reward_prompt.to(device)
        self_reward_model = self_reward_model.to(device)

        reward_responses = sample(
            self_reward_model,
            prompts = reward_prompt,
            seq_len = self.generate_reward_max_seq_len,
            temperature = self.eval_temperature,
            filter_fn = self.eval_filter_fn,
            filter_kwargs = self.eval_filter_kwargs
        )

        reward_responses_as_str: List[str] = [self.tokenizer_decode(resp[resp != self.pad_id].cpu()) for resp in reward_responses]
        rewards: List[Optional[float]] = [parse_reward(resp_str) for resp_str in reward_responses_as_str]

        rewards = [*filter(exists, rewards)] # for now, just filter out any failed responses

        if len(rewards) == 0:
            return None

        avg_reward = Tensor(rewards).mean().item()
        return avg_reward

    @torch.no_grad()
    def forward(self) -> DPODataset:

        self.model.eval()
        device = next(self.model.parameters()).device

        num_generated = 0

        pbar = tqdm(desc = 'generating dpo dataset with self-rewarding')

        prompt_dl = cycle(self.prompt_dataloader)

        while num_generated < self.num_preference_pairs:

            prompts: List[str] = next(prompt_dl)

            prompt_tensors: List[Tensor] = [*map(self.tokenizer_encode, prompts)]

            responses = []

            model = self.model.to(self.device)

            for prompt, prompt_tensor in zip(prompts, prompt_tensors):

                prompt_len = prompt_tensor.shape[-1]
                repeated_prompt_tensor = repeat(prompt_tensor, 'n -> r n', r = self.num_candidate_responses)
                repeated_prompt_tensor = repeated_prompt_tensor.to(self.device)

                candidate_tensor_responses = sample(
                    model,
                    prompts = repeated_prompt_tensor,
                    seq_len = self.preference_max_seq_len,
                    temperature = self.gen_temperature,
                    filter_fn = self.gen_filter_fn,
                    filter_kwargs = self.gen_filter_kwargs
                )

                candidate_int_responses: List[List[int]] = [response.tolist() for response in candidate_tensor_responses]
                candidate_responses: List[str] = [*map(self.tokenizer_decode, candidate_int_responses)]

                candidate_tensor_responses = pad_sequence(candidate_tensor_responses, batch_first = True, padding_value = self.pad_id)
                responses_with_prompt = torch.cat((repeated_prompt_tensor, candidate_tensor_responses), dim = -1)

                if not self.has_external_reward_model:
                    # get rewards through self-rewarding

                    rewards: List[Optional[float]] = [self.generate_reward(prompt, response) for response in candidate_responses]

                    rewards = [reward if exists(reward) and self.is_valid_reward(reward) else None for reward in rewards]

                    # turn rewards into a Tensor

                    rewards_tensor = Tensor([default(reward, float('nan')) for reward in rewards])
                else:
                    # or use external reward module

                    reward_model_input = responses_with_prompt.masked_fill(responses_with_prompt == self.pad_id, 0)

                    rewards_tensor = self.reward_model(reward_model_input)

                    # auto-handle different types of external reward model output

                    assert rewards_tensor.ndim <= 2

                    if rewards_tensor.shape[-1] > 1:
                        rewards_tensor = rewards_tensor.argmax(dim = -1)
                    elif rewards_tensor.ndim == 2:
                        rewards_tensor = rearrange(rewards_tensor, 'b 1 -> b')

                # if there are less than 2 candidate responses with properly returned reward responses, try again

                if (~torch.isnan(rewards_tensor)).sum(dim = -1) < 2:
                    continue

                preference_pair_indices = self.pick_paired_rewards(rewards_tensor)

                # pick out the max and min reward values

                paired_rewards = rewards_tensor[preference_pair_indices]

                # pick out the preferred and unpreferred response

                paired_responses_with_prompt = responses_with_prompt[preference_pair_indices]

                if not self.is_valid_reward_pair(*paired_rewards.unbind(dim = -1)):
                    break

                memmap_idx = num_generated

                paired_responses_with_prompt = pad_or_slice_to(paired_responses_with_prompt, self.preference_max_seq_len, dim = -1, pad_value = self.pad_id)

                self.prompt_len_memmap[memmap_idx] = prompt_len
                self.preference_seq_memmap[memmap_idx] = paired_responses_with_prompt.cpu().numpy()
                self.self_reward_memmap_file[memmap_idx] = paired_rewards.cpu().numpy()

                num_generated += 1
                pbar.update(1)

                if num_generated >= self.num_preference_pairs:
                    break

        # flush + close and return instance of DPO Dataset for the two memmapped data files

        del self.prompt_len_memmap
        del self.preference_seq_memmap

        return DPODataset(**self.dpo_dataset_kwargs)

# fine tuning configs

class FinetuneConfig:
    pass

default_dict = partial(field, default_factory = dict)

@dataclass
class SFTConfig(FinetuneConfig):
    train_dataset: Union[Dataset, List[Dataset]]
    valid_dataset: Optional[Dataset] = None
    dropout: float = 0.1
    trainer_kwargs: dict = default_dict()

@dataclass
class SelfRewardDPOConfig(FinetuneConfig):
    prompt_dataset: Dataset
    num_generated_preference_pairs: int
    dpo_beta: float = 0.1
    max_seq_len: int = 1024
    rewarding_model: Optional[Module] = None   # defaults to self, but can be an external model, as done in DAP https://arxiv.org/abs/2402.04792 (renamed "LLM Annotator")
    self_reward_config_keyname: str = 'default'
    is_valid_reward: Callable[float, bool] = lambda reward: reward >= 0
    is_valid_reward_pair: Callable[[Tensor, Tensor], bool] = default_is_valid_reward_pair
    pick_paired_rewards_fn: Callable[[Tensor], Tensor] = default_pick_paired_rewards_fn
    dropout: float = 0.1
    early_stopper_eval_module: Optional[Module] = None
    num_train_steps: Optional[Module] = None
    num_candidate_responses: int = 4
    num_sampled_reward_responses: int = 3
    gen_temperature: float = 0.7
    gen_filter_fn: Callable = top_p
    gen_filter_kwargs: dict = default_dict()
    eval_temperature: float = 0.7
    eval_filter_fn: Callable = top_p
    eval_filter_kwargs: dict = default_dict()
    trainer_kwargs: dict = field(default_factory = dict)
    reward_generator_kwargs: dict = default_dict()

@dataclass
class ExternalRewardDPOConfig(FinetuneConfig):
    reward_model: Module
    dpo_beta: float = 0.1
    max_seq_len: int = 1024
    gen_temperature: float = 0.7
    gen_filter_fn: Callable = top_p
    gen_filter_kwargs: dict = default_dict()
    dropout: float = 0.1
    trainer_kwargs: dict = default_dict()
    reward_generator_kwargs: dict = default_dict()

@dataclass
class SelfPlayConfig(FinetuneConfig):
    train_dataset: Dataset
    valid_dataset: Optional[Dataset] = None
    max_seq_len: int = 1024
    spin_λ: float = 0.1
    dropout: float = 0.1
    temperature: float = 0.7
    filter_fn: Callable = top_p
    filter_kwargs: dict = default_dict()
    trainer_kwargs: dict = default_dict()
    spin_kwargs: dict =  default_dict()

# generated default config for paper

@beartype
def create_default_paper_config(
    *,
    train_sft_dataset: Union[Dataset, List[Dataset]],
    self_reward_prompt_dataset: Union[Dataset, Tuple[Dataset, Dataset]],
    valid_sft_dataset: Optional[Dataset] = None,
    num_generated_preference_pairs = (3964, 6942),
    early_stopper_eval_module: Optional[Module] = None,
    dpo_num_train_steps: Optional[int] = None,
    sft_config: dict = dict(),
    self_reward_config: dict = dict()

) -> List[FinetuneConfig]:

    prompt_dataset_iter1, prompt_dataset_iter2 = cast_tuple(self_reward_prompt_dataset, 2, validate = True)
    num_generated_iter1, num_generated_iter2 = num_generated_preference_pairs

    return [
        SFTConfig(
            train_dataset = train_sft_dataset,
            valid_dataset = valid_sft_dataset,
            **sft_config
        ),
        SelfRewardDPOConfig(
            num_generated_preference_pairs = num_generated_iter1,
            prompt_dataset = prompt_dataset_iter1,
            num_train_steps = dpo_num_train_steps,
            early_stopper_eval_module = early_stopper_eval_module,
            **self_reward_config
        ),
        SelfRewardDPOConfig(
            num_generated_preference_pairs = num_generated_iter2,
            prompt_dataset = prompt_dataset_iter2,
            num_train_steps = dpo_num_train_steps,
            early_stopper_eval_module = early_stopper_eval_module,
            **self_reward_config
        )
    ]

# self-rewarding trainer class

class SelfRewardingTrainer(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        finetune_configs: Union[Dict, List[FinetuneConfig]],
        tokenizer_encode: Callable[[str], TensorType['seq', int]],
        tokenizer_decode: Callable[[TensorType['seq', int]], str],
        self_reward_prompt_config: Union[RewardConfig, Dict[str, RewardConfig]] = SELF_REWARD_PROMPT_CONFIG,
        pad_id: int = -1,
        checkpoints_folder: str = './checkpoints',
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()

        if isinstance(self_reward_prompt_config, RewardConfig):
            self_reward_prompt_config = dict(default = self_reward_prompt_config)

        # finetune config

        if isinstance(finetune_configs, dict):
            finetune_configs = create_default_paper_config(**finetune_configs)

        # model and accelerator

        self.model = model

        self.accelerator = Accelerator(**accelerate_kwargs)

        # all trainers

        self.trainers: List[Tuple[str, Callable]] = []

        # config -> trainers

        for ind, config in enumerate(finetune_configs):
            finetune_stage = ind + 1

            if isinstance(config, SFTConfig):
                trainer = SFTTrainer(
                    self.model,
                    accelerator = self.accelerator,
                    dropout = config.dropout,
                    train_dataset = config.train_dataset,
                    valid_dataset = config.valid_dataset,
                    **config.trainer_kwargs
                )

                self.trainers.append(('sft', trainer))

            elif isinstance(config, SelfRewardDPOConfig):

                assert exists(config.early_stopper_eval_module) ^ exists(config.num_train_steps), 'either a validation module is passed in for early stopping, or a max number of training steps is specified'

                assert config.self_reward_config_keyname in self_reward_prompt_config, f'reward prompt must be one of {self_reward_prompt_config.keys()}'

                self_reward_model = default(config.rewarding_model, model)

                self_reward_config = self_reward_prompt_config[config.self_reward_config_keyname]

                self_reward_dataset_generator = DPODatasetGenerator(
                    model = model,
                    self_reward_model = self_reward_model,
                    prompt_dataset = config.prompt_dataset,
                    reward_config = self_reward_config,
                    num_preference_pairs = config.num_generated_preference_pairs,
                    preference_max_seq_len = config.max_seq_len,
                    tokenizer_encode = tokenizer_encode,
                    tokenizer_decode = tokenizer_decode,
                    is_valid_reward_pair = config.is_valid_reward_pair,
                    pick_paired_rewards = config.pick_paired_rewards_fn,
                    gen_temperature = config.gen_temperature,
                    gen_filter_fn = config.gen_filter_fn,
                    gen_filter_kwargs = config.gen_filter_kwargs,
                    eval_temperature = config.eval_temperature,
                    eval_filter_fn = config.eval_filter_fn,
                    eval_filter_kwargs = config.eval_filter_kwargs,
                    accelerator = self.accelerator,
                    **config.reward_generator_kwargs
                )

                trainer = DPOTrainer(
                    dpo = model,
                    accelerator = self.accelerator,
                    dataset_generator = self_reward_dataset_generator,
                    dropout = config.dropout,
                    early_stopper_eval_module = config.early_stopper_eval_module,
                    early_stopper_kwargs = dict(
                        early_stop_checkpoint_folder = f'./early-stop-checkpoint.{finetune_stage}',
                    ),
                    dpo_kwargs = dict(
                        beta = config.dpo_beta,
                        pad_id = pad_id
                    ),
                    **config.trainer_kwargs
                )

                self.trainers.append(('dpo', trainer))

            elif isinstance(config, ExternalRewardDPOConfig):

                self_reward_dataset_generator = DPODatasetGenerator(
                    model = model,
                    prompt_dataset = config.prompt_dataset,
                    reward_model = config.reward_model,
                    num_preference_pairs = config.num_generated_preference_pairs,
                    preference_max_seq_len = config.max_seq_len,
                    tokenizer_encode = tokenizer_encode,
                    tokenizer_decode = tokenizer_decode,
                    is_valid_reward = config.is_valid_reward,
                    is_valid_reward_pair = config.is_valid_reward_pair,
                    pick_paired_rewards = config.pick_paired_rewards,
                    gen_temperature = config.gen_temperature,
                    gen_filter_fn = config.gen_filter_fn,
                    gen_filter_kwargs = config.gen_filter_kwargs,
                    eval_temperature = config.eval_temperature,
                    eval_filter_fn = config.eval_filter_fn,
                    eval_filter_kwargs = config.eval_filter_kwargs,
                    **config.reward_generator_kwargs
                )

                trainer = DPOTrainer(
                    dpo = model,
                    accelerator = self.accelerator,
                    dataset_generator = self_reward_dataset_generator,
                    dropout = dropout,
                    early_stopper_eval_module = config.early_stopper_eval_module,
                    early_stopper_kwargs = dict(
                        early_stop_checkpoint_folder = f'./early-stop-checkpoint.{dpo_iteration}',
                    ),
                    dpo_kwargs = dict(
                        beta = config.dpo_beta,
                        pad_id = pad_id
                    ),
                    **config.dpo_trainer_kwargs
                )

                self.trainers.append(('dpo', trainer))

            elif isinstance(config, SelfPlayConfig):

                trainer = SPINTrainer(
                    self.model,
                    accelerator = self.accelerator,
                    dropout = config.dropout,
                    train_sft_dataset = config.train_dataset,
                    valid_sft_dataset = config.valid_dataset,
                    max_seq_len = config.max_seq_len,
                    pad_id = pad_id,
                    temperature = config.temperature,
                    filter_fn = config.filter_fn,
                    filter_kwargs = config.filter_kwargs,
                    spin_kwargs = {
                        'λ': config.spin_λ,
                        **config.spin_kwargs
                    }
                )

                self.trainers.append(('spin', trainer))

            else:
                raise ValueError(f'you did not write out the logic for your custom trainer from your custom finetune config')

        assert len(self.trainers) == len(finetune_configs)

        # checkpoints folder

        checkpoints_folder = Path(checkpoints_folder)
        checkpoints_folder.mkdir(parents = True, exist_ok = True)
        self.checkpoints_folder = checkpoints_folder

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def print(self, *msg):
        self.accelerator.print(*msg)

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def save(self, path: str, overwrite: bool = False):
        self.wait()

        if self.accelerator.is_main_process:

            path = self.checkpoints_folder / path

            assert not path.exists() or overwrite, f'file already exists'

            pkg = dict(
                model = self.unwrapped_model.state_dict()
            )

            torch.save(pkg, str(path))

    def forward(
        self,
        overwrite_checkpoints: bool = False
    ):

        for ind, (trainer_type, trainer) in enumerate(self.trainers):
            finetuning_stage = ind + 1
            trainer()

            self.save(f'{finetuning_stage}.{trainer_type}.ckpt.pt', overwrite = overwrite_checkpoints)

        self.print(f'self-reward training done')
