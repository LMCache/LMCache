import copy
import json
import os
import time

from lmcache_vllm.vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
context_file = os.path.join(os.pardir, 'ffmpeg.txt')
output_file = "offline_inference_outputs.jsonl"

context_text = None
with open(context_file, 'r') as f:
    context_text = f.read()
assert context_text is not None
tokenizer = AutoTokenizer.from_pretrained(model_name)

context_messages = [
    {
        "role":
        "user",
        "content":
        "I've got a document, "
        f"here's the content:```\n{context_text}\n```."
    },
    {
        "role": "assistant",
        "content": "I've got your document"
    },
]
user_inputs_batch = [
    "Give me a concise description for the format"
    " of ffmpeg command in one line.",
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


context_length = get_context_length(tokenizer, context_messages)
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
prompts = gen_prompts(tokenizer, context_messages, user_inputs_batch)
# Create an LLM.
llm = LLM(model=model_name,
          gpu_memory_utilization=0.8,
          enable_chunked_prefill=False,
          max_model_len=32768)

# Clear output file.
with open(output_file, "w") as f:
    pass

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
t1 = time.perf_counter()
first_outputs = llm.generate(prompts, sampling_params)
t2 = time.perf_counter()
print(f"\n\nFirst request Time: {t2 - t1} seconds\n\n")
append_outputs(output_file, first_outputs, context_length, t2 - t1)
t3 = time.perf_counter()
second_outputs = llm.generate(prompts, sampling_params)
t4 = time.perf_counter()
print(f"\n\nSecond request Time: {t4 - t3} seconds\n\n")
append_outputs(output_file, second_outputs, context_length, t4 - t3)
