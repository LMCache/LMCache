import lmcache_vllm
import torch
from lmcache_vllm.blend_adapter import (append_separator,
                                        combine_input_prompt_chunks)
from lmcache_vllm.vllm import LLM, SamplingParams

torch.multiprocessing.set_start_method('spawn')


def precompute_kv(text_chunk, llm):
    sampling_params_prefix = SamplingParams(temperature=0.0,
                                            top_p=0.95,
                                            max_tokens=1)
    text_chunk = append_separator(text_chunk)
    llm.generate([text_chunk], sampling_params_prefix)


context_files = ["chunk1.txt", "chunk2.txt"]
chunks = []

for context_file in context_files:
    with open(context_file, "r") as fin:
        context = fin.read()
    chunks.append(context)

sys_prompt = "Here's a document from the user: "
question = "Question: What does this document mainly talks about? Answer: "

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
          gpu_memory_utilization=0.5,
          tensor_parallel_size=1)
sampling_params_generation = SamplingParams(temperature=0.0,
                                            top_p=0.95,
                                            max_tokens=30)

print(
    "-------------- Pre-computing KV cache for the chunks -------------------")
for chunk in chunks:
    precompute_kv(chunk, llm)

print("Running the real query here!")

user_prompt = [sys_prompt, chunks[0], chunks[1], question]
user_prompt = combine_input_prompt_chunks(user_prompt)
outputs = llm.generate(user_prompt, sampling_params_generation)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Newly generated text: {generated_text!r}")

# Graceful exit
lmcache_vllm.close_lmcache_engine()
