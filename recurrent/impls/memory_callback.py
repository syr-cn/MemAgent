import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template, now, unpad
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens3
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    # max_output_length = max_final_response_length if final else max_memorization_length

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.chunk_size + self.max_memorization_length

    # use property incase we want to adapt soft punishment to length.
    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)

class MemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != 'center':
            raise ValueError('MemoryDataset only support center truncation')
        data_config.max_prompt_length=recurrent_config.max_chunks * recurrent_config.chunk_size
        self.context_key = recurrent_config.context_key
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )

    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        context = row_dict.pop(self.context_key)

        model_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt_ids"] = self.tokenizer.encode(
            chat[0]["content"], add_special_tokens=False
        )
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt_ids"]


TEMPLATE_CALLBACK = """You are an intelligent assistant. Your task is to process a document chunk by chunk to answer a question.
You have access to your previous memory and the current chunk of the document.
First, you must decide if you need to look back at a PREVIOUS chunk to better understand the current one.
Your memory should contain a high-level outline of the information gathered so far to help you search.

Based on the current memory and section, do you need to recall a previous chunk for more context?
Explain your reason and output the callback decision in the format <callback>ID</callback> where ID is the chunk number (e.g., <callback>0</callback>).
If you do not need to recall a previous chunk, output <callback>-1</callback>.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Your decision ({chunk_range}):
"""



TEMPLATE_WITH_CALLBACK = """{callback_input}
{callback_response}

<callbacked_section>
{callback_chunk}
</callbacked_section>

Now, update the memory with the new information from the current section and the recalled chunk.
Updated memory:
"""


TEMPLATE_WITH_CALLBACK_2 = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<callbacked_section>
{callback_chunk}
</callbacked_section>

<section>
{chunk}
</section>

Updated memory:
"""


TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""


class MemoryAgent_Callback(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        # A trick to get a simple chat_template for any tokenizer
        # the output text looks like:
        # '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n'
        # This is a format string itself, '{message}' will be replaced by the actual message.
        self.chat_template = chat_template(tokenizer)
        self.token_message_template_before_callback = TokenTemplate(self.chat_template.format(message=TEMPLATE_CALLBACK), tokenizer)
        self.token_message_template_after_callback = TokenTemplate(self.chat_template.format(message=TEMPLATE_WITH_CALLBACK_2), tokenizer)
        self.token_final_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_FINAL_BOXED), tokenizer)
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length + self.token_message_template_before_callback.length + self.token_message_template_after_callback.length + self.config.chunk_size
        logger.info(f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) + {self.token_message_template_before_callback.length}(message_template_before_callback)'
              f'+ {self.token_message_template_after_callback.length}(message_template_after_callback) + {self.config.chunk_size}(chunk_size) = {self.max_input_length}\n')
        self.NO_MEMORY_TOKENS = tokenizer.encode("No previous memory", add_special_tokens=False)
        self.NO_CHUNK_TOKENS = tokenizer.encode("No chunk was recalled.", add_special_tokens=False)
    
    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch
        
        self.ctx_length = gen_batch.batch['context_length'] # if all context is used, then the sample will no more be active
        self.bsz = len(self.ctx_length)
        self.memory = np.empty(self.bsz, dtype=object)
        self.is_final = False
    
    def action_callback(self) -> Tuple[List[torch.Tensor], dict]:
        """Prepares the inputs for the callback decision step."""
        active_mask = self.ctx_length > self.step * self.config.chunk_size
        if active_mask.sum().item() == 0:
            self.is_final = True
            return None, None
        self.active_mask = active_mask # Store for use in the main action and update steps
        
        prompt_i = self.gen_batch.non_tensor_batch['prompt_ids'][active_mask]
        chunk_i = self.gen_batch.batch['context_ids'][active_mask, self.config.chunk_size * self.step: self.config.chunk_size * (self.step+1)]
        memory_i = self.memory[active_mask]

        chunk_range_str = f"range: -1 or 1<=x<{self.step}" if self.step > 1 else 'range: -1'
        chunk_range = torch.tensor(self.tokenizer.encode(chunk_range_str, add_special_tokens=False), device=chunk_i.device, dtype=chunk_i.dtype)
        self.callback_messages = [
            self.token_message_template_before_callback.format( # this function accept tokenized string
                prompt=prompt,
                memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                chunk=chunk[chunk != self.tokenizer.pad_token_id],
                chunk_range=chunk_range
            )
            for prompt, memory, chunk in zip(prompt_i, memory_i, chunk_i)
        ]
        sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask] # map active sample to original batch
        final_mask = torch.full(sample_index.shape, False, dtype=torch.bool) # all False

        meta_info = {
            'input_pad_to': self.max_input_length,
            'pad_to': self.config.gen_pad_to,
            'generation_kwargs': {
                'max_tokens': 100,
                'n': 1,
                'eos_token_id': self.tokenizer.encode('>', add_special_tokens=False)[0],
            }
        }
        logger.info(f'MemoryAgent.action_callback() done for step {self.step}')
        
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.callback_messages, meta_info
    
    @override
    def action(self, callback_inputs: List[torch.Tensor] = None, callback_responses: List[torch.Tensor] = None, callback_chunks: List[torch.Tensor] = None) -> Tuple[List[torch.Tensor], dict]:
        """
        Prepares the inputs for the memory generation step, now including the recalled chunk.
        Params are set to None for final turn.
        """
        active_mask = self.active_mask
        gen_batch = self.gen_batch
        # if all context is used, and its not done, then it will be the final turn for this batch
        if callback_inputs is None or callback_responses is None or callback_chunks is None:
            self.is_final = True
        if self.is_final:
            self.messages = [
                self.token_final_message_template.format(
                    prompt=prompt,
                    memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                )
                for prompt, memory in zip(gen_batch.non_tensor_batch['prompt_ids'], self.memory)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.int)
            final_mask = torch.full(sample_index.shape, True, dtype=torch.bool) # all True
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'FINAL TURN: MemoryAgent.next() done')
        else:
            prompt_i = self.gen_batch.non_tensor_batch['prompt_ids'][active_mask]
            chunk_i = self.gen_batch.batch['context_ids'][active_mask, self.config.chunk_size * self.step: self.config.chunk_size * (self.step+1)]
            memory_i = self.memory[active_mask]

            self.messages = [
                self.token_message_template_after_callback.format(
                        prompt=prompt,
                        memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                        chunk=chunk[chunk != self.tokenizer.pad_token_id],
                        callback_chunk=callback_chunk[callback_chunk != self.tokenizer.pad_token_id] if callback_chunk is not None else self.NO_CHUNK_TOKENS,                        
                )
                for prompt, memory, chunk, callback_chunk in zip(prompt_i, memory_i, chunk_i, callback_chunks)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask] # map active sample to original batch
            final_mask = torch.full(sample_index.shape, False, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'MemoryAgent.action() done for step {self.step}')

        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.messages, self.meta_info

    @override
    def action_bak(self, callback_inputs: List[torch.Tensor] = None, callback_responses: List[torch.Tensor] = None, callback_chunks: List[torch.Tensor] = None) -> Tuple[List[torch.Tensor], dict]:
        """
        Prepares the inputs for the memory generation step, now including the recalled chunk.
        Params are set to None for final turn.
        """
        active_mask = self.active_mask
        gen_batch = self.gen_batch
        # if all context is used, and its not done, then it will be the final turn for this batch
        if callback_inputs is None or callback_responses is None or callback_chunks is None:
            self.is_final = True
        if self.is_final:
            self.messages = [
                self.token_final_message_template.format(
                    prompt=prompt,
                    memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                )
                for prompt, memory in zip(gen_batch.non_tensor_batch['prompt_ids'], self.memory)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.int)
            final_mask = torch.full(sample_index.shape, True, dtype=torch.bool) # all True
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'FINAL TURN: MemoryAgent.next() done')
        else:
            self.messages = [
                self.token_message_template_after_callback.format(
                        callback_input=cb_input[cb_input != self.tokenizer.pad_token_id],
                        callback_response=cb_response[cb_response != self.tokenizer.pad_token_id],
                        callback_chunk=cb_chunk[cb_chunk != self.tokenizer.pad_token_id] if cb_chunk is not None else self.NO_CHUNK_TOKENS,                        
                )
                for cb_input, cb_response, cb_chunk in zip(callback_inputs, callback_responses, callback_chunks)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask] # map active sample to original batch
            final_mask = torch.full(sample_index.shape, False, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'MemoryAgent.action() done for step {self.step}')

        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.messages, self.meta_info

    @override
    def update(self, callback_output: DataProto, gen_output: DataProto) -> DataProto:
        if not self.is_final:
            self.memory[self.active_mask] = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
            self.log_step(self.callback_messages, callback_output, step_type='Callback Generation')
        self.log_step(self.messages, gen_output, step_type='Memory Generation')
        self.step += 1
        return gen_output

    def get_previous_chunk(self, in_batch_id, step_id) -> str:
        """Get the previous chunk from the processed chunks."""
        previous_chunk = self.gen_batch.batch['context_ids'][in_batch_id, self.config.chunk_size * (step_id-1): self.config.chunk_size * step_id]
        previous_chunk = previous_chunk[previous_chunk != self.tokenizer.pad_token_id]
        return previous_chunk

    ## MODIFIED: Helper function to parse the callback ID from the LLM's text output
    def _parse_callback_id(self, text_response: str) -> int:
        """Extracts the chunk ID from a string like '<callback>2</callback>'."""
        try:
            match = re.search(r'<callback>(\-?\d+)</callback>', text_response)
            if match:
                chunk_id = int(match.group(1))
                # Ensure the requested ID is valid and not out of bounds
                if -1 <= chunk_id <= self.step:
                    return chunk_id
        except (ValueError, TypeError):
            pass # Fall through to return -1 if parsing fails
        
        return -1

    @override
    def done(self):
        return self.is_final
    
    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.memory
        del self.messages
        del self.callback_messages
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index
        

    def log_step(self, messages, gen_output, step_type='Generation'):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=4000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        # Header with dynamic step number
        step = self.step if not self.is_final else "FINAL"
        logger.info(f"\n{'='*30}[RECURRENT] STEP{step} ({step_type}){'='*30}")

        # Message and Response section
        if self.active_mask[0]:
            decoded_message = self.tokenizer.decode(messages[0])
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent_Callback)
