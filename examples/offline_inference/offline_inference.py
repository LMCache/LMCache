import time

from lmcache_vllm.vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The most popular sports in the world is",
    "The father of computer is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
          gpu_memory_utilization=0.8)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
t1 = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
t2 = time.perf_counter()
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
print(f"Total Time: {t2 - t1} seconds")