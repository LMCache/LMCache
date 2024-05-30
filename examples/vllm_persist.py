from vllm import LLM, SamplingParams

from lmcache.cache_engine import LMCacheEngineBuilder, LMCacheEngineConfig

cache_engine_cfg = LMCacheEngineConfig.from_defaults(chunk_size = 256, persist_path = "/local/yihua98/cache-engine.pth")
cache_engine_id = "vllm" # this is also hard-coded in vllm
engine = LMCacheEngineBuilder.get_or_create(cache_engine_id, cache_engine_cfg)

context_file = "f.txt"
with open(context_file, "r") as fin:
    context = fin.read()

context = context * 2

prompts = [context]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
#llm = LLM(model="facebook/opt-125m")
#llm = LLM(model="lmsys/longchat-7b-16k")
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.7)
#llm = LLM(model="gpt2")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")

engine.persist()
