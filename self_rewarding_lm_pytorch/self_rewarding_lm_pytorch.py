import re
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
from textwrap import dedent

from beartype import beartype
from beartype.typing import Optional, Dict, List, Union, Callable
from torchtyping import TensorType

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
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

from self_rewarding_lm_pytorch.spin import SPINTrainer

from einops import rearrange, repeat

from accelerate import Accelerator

from pytorch_custom_utils.utils import pad_or_slice_to

from self_rewarding_lm_pytorch.sampling_utils import (
    sample,
    top_p,
    top_k
)

from self_rewarding_lm_pytorch.mocks import always

from tqdm import tqdm

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

REWARD_PROMPT_CONFIG = dict(
    default = RewardConfig(
        prompt_template = DEFAULT_LLM_AS_JUDGE_PROMPT,
        reward_regex_template = DEFAULT_REWARD_REGEX_TEMPLATE
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
        valid_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        num_epochs: int = 3,
        start_learning_rate: float = 5.5e-6,
        end_learning_rate: float = 1.1e-6,
        learning_rate_num_decay_steps: Optional[int] = None,
        weight_decay: float = 0.,
        ignore_index: int = -1,
        adam_kwargs: dict = dict(),
        valid_every: int = 1
    ):
        super().__init__()
        self.accelerator = accelerator
        self.model = model

        self.num_epochs = num_epochs
        self.ignore_index = ignore_index

        if isinstance(train_dataset, list):
            train_dataset = ConcatDataset(train_dataset)

        self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

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
        step = 0

        for epoch in tqdm(range(self.num_epochs), desc = 'sft finetuning epoch'):
            for seq, prompt_len_or_mask in tqdm(self.train_dataloader, desc = 'sft finetuning'):

                self.model.train()

                loss = self.get_cross_entropy_loss(seq, prompt_len_or_mask)

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.accelerator.log(dict(loss = loss.item()), step = step)

                step += 1

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

                                total_valid_loss = loss.item() * batch
                                total_batches += batch

                        valid_loss = total_valid_loss / total_batches

                        self.accelerator.log(dict(valid_loss = valid_loss), step = step)

                    self.wait()

# reward generator class

class DPODatasetGenerator(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        prompt_dataset: Dataset,
        num_preference_pairs: int,
        tokenizer_encode: Callable[[str], TensorType['seq', int]],
        tokenizer_decode: Callable[[TensorType['seq', int]], str],
        batch_size: int = 16,
        num_candidate_responses: int = 4,
        gen_temperature: float = 0.7,
        gen_nucleus_p: float = 0.9,
        eval_temperature: float = 0.7,
        eval_nucleus_p: float = 0.9,
        num_evals_to_average: int = 3,
        *,
        reward_config: RewardConfig,
        data_folder: str = './',
        preference_seq_memmap_file: str = 'preference_seq.memmap.npy',
        prompt_len_memmap_file: str = 'prompt_len.memmap.npy',
        self_reward_memmap_file: str = 'self_reward.memmap.npy',
        preference_max_seq_len: int = 1024,
        generate_reward_max_seq_len: int = 256,
        is_valid_reward_pair: Optional[Callable[[float, float], bool]] = None,
        pad_id: int = -1
    ):
        super().__init__()

        self.model = model
        self.num_candidate_responses = num_candidate_responses

        self.reward_config = reward_config.init()

        self.batch_size = batch_size
        self.prompt_dataset = prompt_dataset
        self.prompt_dataloader = DataLoader(prompt_dataset, batch_size = batch_size, shuffle = True)

        self.gen_nucleus_p = gen_nucleus_p
        self.gen_temperature = gen_temperature

        self.eval_nucleus_p = eval_nucleus_p
        self.eval_temperature = eval_temperature

        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

        self.num_evals_to_average = num_evals_to_average

        # logic for sampling the reward pair and to validate it before adding it to generated preference dataset

        self.is_valid_reward_pair = default(is_valid_reward_pair, lambda *args: True)

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

        reward_responses = sample(
            self.model,
            prompts = reward_prompt.long(),
            seq_len = self.generate_reward_max_seq_len,
            temperature = self.eval_temperature,
            filter_fn = top_p, 
            filter_kwargs = dict(
                thres = self.eval_nucleus_p
            )
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

            for prompt, prompt_tensor in zip(prompts, prompt_tensors):

                prompt_len = prompt_tensor.shape[-1]
                repeated_prompt_tensor = repeat(prompt_tensor, 'n -> r n', r = self.num_candidate_responses)

                candidate_tensor_responses = sample(
                    self.model,
                    prompts = repeated_prompt_tensor.long(),
                    seq_len = self.preference_max_seq_len,
                    temperature = self.gen_temperature,
                    filter_fn = top_p,
                    filter_kwargs = dict(
                        thres = self.gen_nucleus_p
                    )
                )

                candidate_int_responses: List[List[int]] = [response.tolist() for response in candidate_tensor_responses]
                candidate_responses: List[str] = [*map(self.tokenizer_decode, candidate_int_responses)]

                # get rewards

                rewards: List[Optional[float]] = [self.generate_reward(prompt, response) for response in candidate_responses]

                # zip together the responses and rewards and filter out if reward is not generated correctly

                paired_reward_response = [(reward, candidate_response) for reward, candidate_response in zip(rewards, candidate_tensor_responses)]

                paired_reward_response = [*filter(lambda pair: exists(first(pair)), paired_reward_response)]
                paired_reward_response.sort(key = first)

                if len(paired_reward_response) < 2:
                    continue

                unpreferred_reward, unpreferred_response = paired_reward_response[0]
                preferred_reward, preferred_response = paired_reward_response[1]

                if not self.is_valid_reward_pair(preferred_reward, unpreferred_reward):
                    break

                memmap_idx = num_generated

                paired_responses = pad_sequence((preferred_response, unpreferred_response), padding_value = self.pad_id, batch_first = True)

                paired_responses_with_prompt = torch.cat((repeated_prompt_tensor[:2], paired_responses), dim = -1)

                paired_responses_with_prompt = pad_or_slice_to(paired_responses_with_prompt, self.preference_max_seq_len, dim = -1, pad_value = self.pad_id)

                self.prompt_len_memmap[memmap_idx] = prompt_len
                self.preference_seq_memmap[memmap_idx] = paired_responses_with_prompt.cpu().numpy()
                self.self_reward_memmap_file[memmap_idx] = np.array([preferred_reward, unpreferred_reward])

                num_generated += 1
                pbar.update(1)

                if num_generated >= self.num_preference_pairs:
                    break

        # flush and return instance of DPO Dataset for the two memmapped data files

        self.prompt_len_memmap.flush()
        self.preference_seq_memmap.flush()

        return DPODataset(**self.dpo_dataset_kwargs)

# self-rewarding trainer class

class SelfRewardingTrainer(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        tokenizer_encode: Callable[[str], TensorType['seq', int]],
        tokenizer_decode: Callable[[TensorType['seq', int]], str],
        train_sft_dataset: Optional[Union[List[Dataset], Dataset]] = None,
        valid_sft_dataset: Optional[Dataset] = None,
        initial_sft: bool = True,
        spin: bool = False,
        dpo_beta = 0.1,
        spin_λ  = 0.1,
        preference_max_seq_len: int = 1024,
        self_reward_num_iterations = 2,
        reward_prompt_config: Union[RewardConfig, Dict[str, RewardConfig]] = REWARD_PROMPT_CONFIG,
        reward_iteration_type = [
            'default',
            'default'
        ],
        num_preference_pairs: List[int] = [
            3964,
            6942
        ],
        reward_generator_kwargs: dict = dict(
            num_candidate_responses = 4,
            gen_temperature = 0.7,
            gen_nucleus_p = 0.9,
            eval_temperature = 0.7,
            eval_nucleus_p = 0.9
        ),
        prompt_dataset: Optional[Dataset] = None,
        model_generate_prompt: Optional[Module] = None,
        early_stopper: Optional[EarlyStopper] = None,
        accelerate_kwargs: dict = dict(),
        sft_trainer_kwargs: dict = dict(),
        spin_trainer_kwargs: dict = dict(),
        dpo_trainer_kwargs: dict = dict(),
        dropout: float = 0.1,
        checkpoints_folder: str = './checkpoints',
        is_valid_reward_pair: Optional[Callable[[float, float], bool]] = lambda preferred_reward, unpreferred_reward: preferred_reward != unpreferred_reward,
        pad_id: int = -1
    ):
        super().__init__()

        if isinstance(reward_prompt_config, RewardConfig):
            reward_prompt_config = dict(default = reward_prompt_config)

        assert all([key in reward_prompt_config for key in reward_iteration_type]), f'reward prompt must be one of {reward_prompt_config.keys()}'

        # appears the prompts come from llama 70B chat
        # also offer a way for it to be passed in

        assert (int(exists(prompt_dataset)) + int(exists(model_generate_prompt))) <= 1

        if not exists(prompt_dataset) and not exists(model_generate_prompt):
            model_generate_prompt = deepcopy(model)

        assert exists(prompt_dataset), 'for now only support prompt dataset being passed in'

        # model and accelerator

        self.model = model

        self.accelerator = Accelerator(**accelerate_kwargs)

        # sft

        self.sft_trainer = None
        self.first_iterate_on_sft = initial_sft

        if self.first_iterate_on_sft:
            assert exists(train_sft_dataset)

            self.sft_trainer = SFTTrainer(
                model,
                accelerator = self.accelerator,
                train_dataset = train_sft_dataset,
                valid_dataset = valid_sft_dataset,
                **sft_trainer_kwargs
            )

        # spin

        self.spin = spin
        self.spin_trainer = None

        if spin:
            assert exists(train_sft_dataset)

            self.spin_trainer = SPINTrainer(
                model,
                accelerator = self.accelerator,
                sft_dataset = train_sft_dataset,
                max_seq_len = spin_trainer_kwargs.pop('max_seq_len', preference_max_seq_len),
                spin_λ = spin_λ,
                pad_id = pad_id,
                **spin_trainer_kwargs
            )

        # self-reward related

        self.reward_prompt_configs = [reward_prompt_config[key] for key in reward_iteration_type]
        self.self_reward_num_iterations = self_reward_num_iterations

        self.dpo_dataset_generators = [
            DPODatasetGenerator(
                model = model,
                prompt_dataset = prompt_dataset,
                reward_config = reward_config,
                num_preference_pairs = one_stage_num_preference_pairs,
                preference_max_seq_len = preference_max_seq_len,
                tokenizer_encode = tokenizer_encode,
                tokenizer_decode = tokenizer_decode,
                is_valid_reward_pair = is_valid_reward_pair,
                **reward_generator_kwargs
            ) for reward_config, one_stage_num_preference_pairs in zip(self.reward_prompt_configs, num_preference_pairs)
        ]

        # dpo

        set_dropout_(model, dropout)
        self.dpo = DPO(
            model,
            beta = dpo_beta,
            pad_id = pad_id
        )

        self.dpo_trainers = []

        for _ in range(self_reward_num_iterations):
            trainer = DPOTrainer(
                dpo = self.dpo,
                accelerator = self.accelerator,
                **dpo_trainer_kwargs
            )

            self.dpo_trainers.append(trainer)

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
        if not self.accelerator.is_main_process:
            return

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

        if self.first_iterate_on_sft:
            self.sft_trainer()

            self.wait()

            self.save('sft.ckpt.pt', overwrite = overwrite_checkpoints)

            self.wait()

        if self.spin:
            self.spin_trainer()

            self.wait()

            self.save('spin.ckpt.pt', overwrite = overwrite_checkpoints)

            self.wait()

        for ind, (dpo_dataset_generator, dpo_trainer) in enumerate(zip(self.dpo_dataset_generators, self.dpo_trainers)):

            iterate_num = ind + 1

            dpo_dataset_from_self_reward = dpo_dataset_generator()

            dpo_trainer(dpo_dataset_from_self_reward)

            self.wait()

            self.dpo.update_reference_model_with_policy()

            self.save(f'self-reward.{iterate_num}.ckpt.pt', overwrite = overwrite_checkpoints)

            self.wait()

        self.print(f'done')
