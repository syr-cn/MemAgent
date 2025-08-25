# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple, Type

import torch
from codetiming import Timer
import numpy as np

from verl import DataProto

from .interface import RAgent, RConfig
from .utils import (chat_template, create_attention_mask, create_position_ids,
                    graceful_padding, indexing_proto,
                    pad_tensor_list_to_length)

logger = logging.getLogger(__file__)
logger.setLevel('INFO')



@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timing_raw.get(name, 0.) + timer.last




class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: RConfig,
        agent_cls: Type[RAgent]
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.world_size = actor_rollout_wg.world_size
        self.agent = agent_cls(tokenizer, config)
        self.chat_template = chat_template(tokenizer)
        self.PADDING_WORD_TOKENS = tokenizer.encode(self.chat_template.format(message="Hello."), add_special_tokens=False)


    from functools import lru_cache
    @lru_cache(maxsize=3)
    def get_paddings(self, shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return padding_token_ids, padding_attention_masks, padding_position_ids
        """
        pad_shape = shape[1:]
        padding_word_ids = self.PADDING_WORD_TOKENS
        padding_token_ids = torch.full(pad_shape, fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        padding_attention_masks = torch.zeros(pad_shape, dtype=torch.long)
        padding_position_ids = torch.zeros(pad_shape, dtype=torch.long)
        # token_ids <pad> <pad> <pad> <tok> <tok> <tok>
        # attn_mask 0     0     0     1     1     1
        # posit_ids 0     0     0     0     1     2
        padding_token_ids[-len(padding_word_ids):] = torch.tensor(padding_word_ids, dtype=torch.long)
        padding_attention_masks[-len(padding_word_ids):] = 1
        padding_position_ids[-len(padding_word_ids):] = torch.arange(0, len(padding_word_ids))
        return padding_token_ids, padding_attention_masks, padding_position_ids
    
    def generate_with_graceful_padding(self, input_ids: torch.Tensor,
                                    attention_masks: torch.Tensor,
                                    position_ids: torch.Tensor,
                                    meta_info: dict):

        """
        batch may not be divisible by wordsize.
        Use "Hello" as padding, insert padding data into batch so that data 
        """
        bsz = input_ids.shape[0]

        group_nums = self.world_size
        remainder = bsz % group_nums
        if remainder:
            # Example pattern for bsz=7, group_nums=3:
            # no_padding_mask: [1, 1, 1, 0, 1, 1, 0, 1, 1]
            # padding_index:   [0, 1, 2, -1, 3, 4, -1, 5, 6]
            padding_index, no_padding_mask = graceful_padding(bsz, group_nums)
            padding_token_ids, padding_attention_masks, padding_position_ids = self.get_paddings(input_ids.shape)
            def padding_by_index(tensor, padding, padding_index):
                if not len(padding.shape) == 2:
                    padding = padding.unsqueeze(0)
                # 2. prepare data for padding, concat padding to the end of batch
                tensor_for_indexing = torch.cat([tensor, padding], dim=0)
                # 3. index, -1 will select padding, else select the corresponding original data 
                return tensor_for_indexing[padding_index]
            
            input_ids = padding_by_index(input_ids, padding_token_ids, padding_index)
            attention_masks = padding_by_index(attention_masks, padding_attention_masks, padding_index)
            position_ids = padding_by_index(position_ids, padding_position_ids, padding_index)

        batch = DataProto.from_dict(tensors={
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_masks
        }, meta_info=meta_info)
        output_batch = self.actor_rollout_wg.generate_sequences(batch)
        if remainder:
            # 4. remove padding
            output_batch = indexing_proto(output_batch, no_padding_mask)
        return output_batch

    def run_llm_loop(self, gen_batch, timing_raw) -> Tuple[DataProto, torch.BoolTensor, torch.LongTensor]:
        """Run main LLM generation loop.
        genbatch: 'context_ids','context_length','prompt_ids'
        timing_raw: timing dict used in ray_trainer, note that we will accumulate the time cost in this loop, instead of override each time as in ray_trainer.
        see `_timer` implementation at the top of this file for more details.
        
        # This is a simplified diagram to show how sample_index works.
        # DataProto and 2D tensors represented as a list of samples.

        # ex. batch = [s1, s2, s3, s4]
        #     gen_batch = [s1_turn1, s2_turn1, s3_turn1, s4_turn1, s1_turn2, s3_turn2, s3_turn3, s1_final, s2_final, s3_final, s4_final]
        #     final_mask = [      F,        F,        F,        F,        F,        F,        F,        T,        T,        T,        T]
        #     sample_index = [    0,        1,        2,        3,        0,        2,        2,        0,        1,        2,        3]
        
        # then, batch[sample_index] will be
        #                 [      s1,       s2,       s3,       s4,       s1,       s3,       s3,       s1,       s2,       s3,       s4]
        # We map info from original_sample to gen_batch_output now, e.x. in reward computation
        """
        active_num_list = [] # trace the active number of sample in each turn
        gen_output_list = [] # store I/O batch in each turn, used for policy optimization
        meta_info = gen_batch.meta_info #  do_sample, is_validate, eos/pad are stored in here.
        pad_token_id = self.tokenizer.pad_token_id
        self.agent.start(gen_batch, timing_raw)
        # Main generation loop, agent should indicate whether to stop
        while not self.agent.done():
            with _timer('mt_prepare', timing_raw):
                messages, meta_info_gen = self.agent.action()
                meta_info_gen.update(meta_info)
                # [len(x) for x in messages] == [len(x[x!=pad_token_id]) for x in input_ids]
                # torch.all(attention_masks.sum(-1) == torch.tensor([len(x[x!=pad_token_id]) for x in input_ids]))
                input_ids = pad_tensor_list_to_length(messages, 
                                                pad_token_id=pad_token_id,
                                                max_length=meta_info_gen['input_pad_to'], 
                                                left_pad=True)
                attention_masks = create_attention_mask(input_ids, pad_token_id=pad_token_id)
                position_ids = create_position_ids(attention_masks)
                active_num_list.append(len(messages))
                logger.info(f'padding done')
            with _timer('mt_gen', timing_raw):
                gen_output = self.generate_with_graceful_padding(input_ids, attention_masks, position_ids, meta_info_gen)
                logger.info('generation done')
            with _timer('mt_update', timing_raw):
                gen_output = self.agent.update(gen_output)
                gen_output_list.append(gen_output)
                logger.info('agent update done')
        final_mask, sample_index = self.agent.end()
        
        # OK, now we've got all we need in gen_output_list, and the final_mask indicates which one is final answer.
        assert len(sample_index) == sum(active_num_list)
        assert sum(final_mask) == len(gen_batch)
        logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")
        return DataProto.concat(gen_output_list), final_mask, sample_index # pyright: ignore

    
    def run_llm_loop_callback(self, gen_batch, timing_raw) -> Tuple[DataProto, torch.BoolTensor, torch.LongTensor]:
        active_num_list = []
        gen_output_list = []
        meta_info = gen_batch.meta_info
        pad_token_id = self.tokenizer.pad_token_id
        self.agent.start(gen_batch, timing_raw)

        while not self.agent.done():
            # -----------------------------------------------------------------
            # STEP 1: Generate Callback Decision
            # -----------------------------------------------------------------
            with _timer('mt_prepare_callback', timing_raw):
                messages_callback, meta_info_callback = self.agent.action_callback()

                if self.agent.is_final:
                    pass # No callback for final turn
                else:
                    if not messages_callback: # No active samples left
                        logger.info("No active samples remaining, breaking loop.")
                        break
                    
                    meta_info_callback.update(meta_info)
                    input_ids_cb = pad_tensor_list_to_length(messages_callback, pad_token_id=pad_token_id, max_length=meta_info_callback['input_pad_to'], left_pad=True)

                    # debug_input_string = self.tokenizer.batch_decode(input_ids_cb, skip_special_tokens=True)
                    
                    attention_masks_cb = create_attention_mask(input_ids_cb, pad_token_id=pad_token_id)
                    position_ids_cb = create_position_ids(attention_masks_cb)
                    active_num_list.append(len(messages_callback))
                    logger.info(f'Callback preparation done for step {self.agent.step}')

            with _timer('mt_gen_callback', timing_raw):
                if self.agent.is_final:
                    pass # No callback for final turn
                else:
                    gen_output_callback = self.generate_with_graceful_padding(input_ids_cb, attention_masks_cb, position_ids_cb, meta_info_callback)
                    gen_output_list.append(gen_output_callback)
                    logger.info('Callback generation done')

            # -----------------------------------------------------------------
            # STEP 2: Parse Callback, Retrieve Chunk, and Generate Memory
            # -----------------------------------------------------------------
            with _timer('mt_prepare_memory', timing_raw):
                # Decode and parse callback IDs
                if self.agent.is_final:
                    messages_mem, meta_info_mem = self.agent.action()
                else:
                    decoded_responses = self.tokenizer.batch_decode(gen_output_callback.batch['responses'], skip_special_tokens=True)

                    recalled_chunks = []
                    active_indices = np.where(self.agent.active_mask)[0]

                    for i, resp_text in enumerate(decoded_responses):
                        original_sample_idx = active_indices[i] # range: [0, bsz)
                        chunk_id = self.agent._parse_callback_id(resp_text)

                        if chunk_id != -1:
                            recalled_chunk = self.agent.get_previous_chunk(original_sample_idx, chunk_id)
                            recalled_chunks.append(recalled_chunk)
                        else:
                            recalled_chunks.append(None) # Sentinel for no callback

                    assert len(recalled_chunks) == len(messages_callback), "Recalled chunks length mismatch with messages_callback length"

                    # Prepare for the main memory generation
                    messages_mem, meta_info_mem = self.agent.action(input_ids_cb, gen_output_callback.batch['responses'], recalled_chunks)
                
                meta_info_mem.update(meta_info)
                input_ids_mem = pad_tensor_list_to_length(messages_mem, pad_token_id=pad_token_id, max_length=meta_info_mem['input_pad_to'], left_pad=True)
                attention_masks_mem = create_attention_mask(input_ids_mem, pad_token_id=pad_token_id)
                position_ids_mem = create_position_ids(attention_masks_mem)
                active_num_list.append(len(messages_mem))
                logger.info(f'Memory preparation done for step {self.agent.step}')

            with _timer('mt_gen_memory', timing_raw):
                gen_output_mem = self.generate_with_graceful_padding(input_ids_mem, attention_masks_mem, position_ids_mem, meta_info_mem)
                logger.info('Memory generation done')

            with _timer('mt_update', timing_raw):
                # Update agent state (memory, step count) using the output from the memory generation
                gen_output_final_for_turn = self.agent.update(gen_output_callback, gen_output_mem)
                gen_output_list.append(gen_output_final_for_turn)
                logger.info('Agent update done')

        # This handles the final turn generation if all context was used up
        if self.agent.active_mask.sum().item() == 0 and not self.agent.is_final:
            logger.info("Entering final turn generation.")
            messages_final, meta_info_final = self.agent.action() # No callback chunks for final action
            meta_info_final.update(meta_info)
            input_ids_final = pad_tensor_list_to_length(messages_final, pad_token_id=pad_token_id, max_length=meta_info_final['input_pad_to'], left_pad=True)
            attention_masks_final = create_attention_mask(input_ids_final, pad_token_id=pad_token_id)
            position_ids_final = create_position_ids(attention_masks_final)
            
            with _timer('mt_gen_final', timing_raw):
                gen_output_final = self.generate_with_graceful_padding(input_ids_final, attention_masks_final, position_ids_final, meta_info_final)
            
            with _timer('mt_update_final', timing_raw):
                 # We don't call update here as it increments step; just append the output
                 gen_output_list.append(gen_output_final)


        final_mask, sample_index = self.agent.end()

        assert len(sample_index) == sum(active_num_list)
        assert sum(final_mask) == len(gen_batch) if any(final_mask) else 0
        logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")
        
        if not gen_output_list:
             return DataProto(), torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.long)

        return DataProto.concat(gen_output_list), final_mask, sample_index
