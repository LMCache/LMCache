import time

import lmcache_vllm
from lmcache_vllm.blend_adapter import combine_input_prompt_chunks
from lmcache_vllm.vllm import LLM, SamplingParams


def precompute_kv(text_chunk, llm):
    sampling_params_prefix = SamplingParams(temperature=0.0,
                                            top_p=0.95,
                                            max_tokens=1)
    llm.generate([text_chunk], sampling_params_prefix)


if __name__ == "__main__":
    context_files = ["../chunk1.txt", "../chunk2.txt"]
    chunks = []

    for context_file in context_files:
        with open(context_file, "r") as fin:
            context = fin.read()
        chunks.append(context)

    sys_prompt = "Here's a document from the user: "
    question = "What can ffmpeg be used for?"

    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
              gpu_memory_utilization=0.8,
              tensor_parallel_size=1)

    # It is common to precompute docs && system prompt.
    print("-------------- Pre-computing KV cache"
          " for the chunks and system prompt -------------------")
    for chunk in chunks:
        precompute_kv(chunk, llm)

    precompute_kv(sys_prompt, llm)
    time.sleep(3)
    print("Running the real query here!")

    user_prompt = [sys_prompt, chunks[0], chunks[1], question]
    user_prompt = combine_input_prompt_chunks(user_prompt)
    sampling_params_generation = SamplingParams(temperature=0.0,
                                                top_p=0.95,
                                                max_tokens=100)
    outputs = llm.generate(user_prompt, sampling_params_generation)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Newly generated text: {generated_text!r}")
        print("\n")
    # Graceful exit
    lmcache_vllm.close_lmcache_engine()
