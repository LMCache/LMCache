from vllm import LLM, SamplingParams
import time
from lmcache.cache_engine import LMCacheEngineBuilder, LMCacheEngineConfig

use_cache = True

cache_engine_cfg = LMCacheEngineConfig.from_defaults(chunk_size = 256, persist_path = "/local/yihua98/cache-engine.pth")
cache_engine_id = "vllm" # this is also hard-coded in vllm
if use_cache:
    engine = LMCacheEngineBuilder.get_or_create(cache_engine_id, cache_engine_cfg)

context_file = "f.txt"
with open(context_file, "r") as fin:
    context = fin.read()
context = context * 2

questions = [
        "Question: What does this document mainly talks about? Answer: ",
]
prompts = [f"{context} {p}" for p in questions]

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1)

#llm = LLM(model="facebook/opt-125m")
#llm = LLM(model="lmsys/longchat-7b-16k")
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.7)
#llm = LLM(model="gpt2")

#profile = cProfile.Profile()
print("\033[32mStart timing...\033[0m")
st = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
#profile.runctx("outputs = llm.generate(prompts, sampling_params)", globals(), locals())
ed = time.perf_counter()
print(f"\033[32mFinished timing... Time is {ed - st}\033[0m")

#outfile = "cache.prof" if use_cache else "normal.prof"
#profile.dump_stats(outfile)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")

