from .aio import get_async_client
from utils import extract_solution
from .envs import URL, API_KEY, RECURRENT_CHUNK_SIZE, RECURRENT_MAX_NEW, RECURRENT_MAX_CONTEXT_LEN
import re

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

TEMPLATE_FINAL = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and format your response as follows "Therefore, the answer is (insert answer here)".

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

NO_MEMORY = "No previous memory"
NO_CALLBACK = "No chunk was recalled"

def _parse_callback_id(text_response: str, step: int) -> int:
    """Extracts the chunk ID from a string like '<callback>2</callback>'."""
    try:
        match = re.search(r'<callback>(\-?\d+)</callback>', text_response)
        if match:
            chunk_id = int(match.group(1))
            # Ensure the requested ID is valid and not out of bounds
            if -1 <= chunk_id <= step:
                return chunk_id
    except (ValueError, TypeError):
        pass # Fall through to return -1 if parsing fails
    
    return -1

def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    # assert max_length > 50, "max_length must be greater than 50"
    if not len(string) > max_length:
        return string
    target_len = max_length - len('\n\n...(truncated)\n\n')
    return string[:target_len//2] + '\n\n...(truncated)\n\n' + string[-target_len//2:]


async def async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None):
    idx = item["_id"]
    context = item["context"].strip()
    prompt = item['input'].strip()
    session = await get_async_client()
    async with session:
        max_len = RECURRENT_MAX_CONTEXT_LEN
        input_ids = tokenizer.encode(context)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        memory = NO_MEMORY
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
            chunk = input_ids[i:i+RECURRENT_CHUNK_SIZE]
            chunk_str = tokenizer.decode(chunk)
            
            # Generate callback message
            chunk_range_str = f"range: -1 or 1<=x<{i}" if i > 1 else 'range: -1'
            msg_callback = TEMPLATE_CALLBACK.format(prompt=prompt, chunk=chunk_str, memory=memory, chunk_range=chunk_range_str)
            if idx == 0:
                print("user:")
                print(clip_long_string(msg_callback))
            try:
                async with session.post(
                    url=URL + "/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json=dict(model=model,
                        messages=[{"role": "user", "content": msg_callback}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=RECURRENT_MAX_NEW
                    )
                ) as resp:
                    status = resp.status
                    if status!= 200:
                        print(f"{status=}, {model=}")
                        return ''
                    data = await resp.json()
                    callback_id = _parse_callback_id(data['choices'][0]['message']['content'], i)
                    if idx == 0:
                        print("assistant:")
                        print(clip_long_string(memory))
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                return ''
            
            # Generate memory based on callback
            if callback_id != -1:
                callback_chunk = input_ids[(callback_id - 1) * RECURRENT_CHUNK_SIZE: callback_id * RECURRENT_CHUNK_SIZE]
                callback_chunk_str = tokenizer.decode(callback_chunk)
            else:
                callback_chunk_str = NO_CALLBACK
                
            msg_mem = TEMPLATE_WITH_CALLBACK_2.format(prompt=prompt, chunk=chunk_str, memory=memory, callback_chunk=callback_chunk_str)
            if idx == 0:
                print("user:")
                print(clip_long_string(msg_mem))
            try:
                async with session.post(
                    url=URL + "/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json=dict(model=model,
                        messages=[{"role": "user", "content": msg_mem}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=RECURRENT_MAX_NEW
                    )
                ) as resp:
                    status = resp.status
                    if status!= 200:
                        print(f"{status=}, {model=}")
                        return ''
                    data = await resp.json()
                    memory, _ = extract_solution(data['choices'][0]['message']['content'])
                    if idx == 0:
                        print("assistant:")
                        print(clip_long_string(memory))
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                return ''

        msg_final = TEMPLATE_FINAL.format(prompt=prompt, memory=memory)
        if idx == 0:
            print("user:")
            print(clip_long_string(msg_final))
        try:
            async with session.post(
                url=URL + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=dict(model=model,
                    messages=[{"role": "user", "content": msg_final}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=RECURRENT_MAX_NEW
                )
            ) as resp:
                status = resp.status
                if status!= 200:
                    print(f"{status=}, {model=}")
                    return ''
                data = await resp.json()
                if idx == 0:
                    print("assistant:")
                    print(data['choices'][0]['message']['content'])
                return data['choices'][0]['message']['content']
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback
            traceback.print_exc()
        return ''

