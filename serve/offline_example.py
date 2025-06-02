# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of cpu offloading
with LMCache.

Note that `pip install lmcache` is needed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""
import os
import time

from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CONFIG_FILE"] = "../config/May_23_1_sum/baseline_kivi/02.yaml"

# This example script runs two requests with a shared prefix.
shared_prompt = "Hi " * 8000
shared_prompt2 = "Hi " * 10000
shared_prompt3 = "Hi " * 12000
shared_prompt4 = "Hi " * 14000
shared_prompt5 = "Hi " * 16000
first_prompt = [
    "1" + shared_prompt,
]
second_prompt = [
    "2" + shared_prompt2,
]
third_prompt = [
    "3" + shared_prompt3,
]
fourth_prompt = [
    "4" + shared_prompt4,
]
fifth_prompt = [
    "5" + shared_prompt5,
]

sampling_params = SamplingParams(temperature=0, max_tokens=100)

ktc = KVTransferConfig.from_cli(
    '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}')
# Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
# memory. Reduce the value if your GPU has less memory.
# Note that LMCache is not compatible with chunked prefill for now.
llm = LLM(model="meta-llama/Llama-3.1-8b-Instruct",
          kv_transfer_config=ktc,
          max_model_len=30000,
          enable_chunked_prefill=False,
          gpu_memory_utilization=0.8,
          enforce_eager=True
          )

for i in range(2):
    outputs = llm.generate(first_prompt, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("First request done.")

    outputs = llm.generate(second_prompt, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("Second request done.")

    outputs = llm.generate(third_prompt, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("Third request done.")

    outputs = llm.generate(fourth_prompt, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("Fourth request done.")

    outputs = llm.generate(fifth_prompt, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("Fifth request done.")

# Clean up lmcache backend
LMCacheEngineBuilder.destroy(ENGINE_NAME)
