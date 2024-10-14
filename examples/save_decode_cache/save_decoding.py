import copy
import json
import os
import time

import numpy as np

from lmcache_vllm.vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
output_file = "save_decoding_outputs.jsonl"


tokenizer = AutoTokenizer.from_pretrained(model_name)

context_messages = [
    {
        "role": "user",
        "content": "You are a helpful assistant."
    },
    {
        "role": "assistant",
        "content": "Got it."
    },
]
user_inputs_batch = [
    "Please introduce the power of AI. "
    "Your answer should be around 2k words.",
]


def get_context_length(tokenizer, context_messages):
    return len(tokenizer.apply_chat_template(context_messages, tokenize=False))


def gen_prompts(tokenizer, context_messages, user_inputs_of_batch):
    generated_prompts = []
    for user_input in user_inputs_of_batch:
        copyed_context_messages = copy.deepcopy(context_messages)
        copyed_context_messages.append({"role": "user", "content": user_input})
        generated_prompts.append(
            tokenizer.apply_chat_template(copyed_context_messages,
                                          tokenize=False))
    return generated_prompts


def append_outputs(output_file_name, outputs, context_length, time_taken):
    user_inputs = []
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        user_input = prompt[context_length:]
        user_inputs.append(user_input)
        generated_text = output.outputs[0].text
        generated_texts.append(f"{generated_text!r}")
    json_dict = {
        "user_inputs": user_inputs,
        "generated_texts": generated_texts,
        "time in seconds": time_taken
    }
    with open(output_file_name, "a") as f:
        f.write(json.dumps(json_dict) + '\n')



prompts = gen_prompts(tokenizer, context_messages, user_inputs_batch)
# Create an LLM.
llm = LLM(model=model_name,
          gpu_memory_utilization=0.8,
          enable_chunked_prefill=False,
          max_model_len=32768)

dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(8,
                                                     32))
dummy_prompts = [{
        "prompt_token_ids": batch
} for batch in dummy_prompt_token_ids.tolist()]
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=10)
print("Warming up.")
llm.generate(dummy_prompts, sampling_params)
print("Wram up done.")
# Clear output file.
with open(output_file, "w") as f:
    pass
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
context_length = get_context_length(tokenizer, context_messages)
t1 = time.perf_counter()
first_outputs = llm.generate(prompts, sampling_params)
t2 = time.perf_counter()
print(f"\n\nFirst request Time: {t2 - t1} seconds\n\n")
append_outputs(output_file, first_outputs, context_length, t2 - t1)
context_messages.extend([
    {
        "role": "user",
        "content": user_inputs_batch[0]
    },
    {
        "role": "assistant",
        "content": first_outputs[0].outputs[0].text
    },
])
user_inputs_batch = [
    "Score your answer from 1-10",
]
context_length = get_context_length(tokenizer, context_messages)
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=10)
prompts = gen_prompts(tokenizer, context_messages, user_inputs_batch)
t3 = time.perf_counter()
second_outputs = llm.generate(prompts, sampling_params)
t4 = time.perf_counter()
print(f"\n\nSecond request Time: {t4 - t3} seconds\n\n")
append_outputs(output_file, second_outputs, context_length, t4 - t3)
second_prompt = prompts[0]
# Modify the prompt to make it not reused.
replace_key = "Please introduce the power of AI."
replace_index = second_prompt.find(replace_key)
second_prompt = second_prompt[:replace_index] + "Tell me the usage of AI." + \
    second_prompt[replace_index + len(replace_key):]
t5 = time.perf_counter()
third_outputs = llm.generate([second_prompt], sampling_params)
t6 = time.perf_counter()
print(f"\n\nThird request Time: {t6 - t5} seconds\n\n")
os._exit(0)