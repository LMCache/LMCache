import os
import copy
import json

from lmcache_vllm.vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def gen_prompts(tokenizer, context_messages, user_inputs_of_batch):
    generated_prompts = []
    for user_input in user_inputs_of_batch:
        copyed_context_messages = copy.deepcopy(context_messages)
        copyed_context_messages.append({"role": "user", "content": user_input})
        generated_prompts.append(
            tokenizer.apply_chat_template(copyed_context_messages,
                                          tokenize=False))
    return generated_prompts

def get_context_length(tokenizer, context_messages):
    return len(tokenizer.apply_chat_template(context_messages, tokenize=False))

def gen_input(tokenizer, context, user_prompt):
    context_messages = [
        {
            "role":
            "user",
            "content":
            "I've got a document, "
            f"here's the content:```\n{context}\n```."
        },
        {
            "role": "assistant",
            "content": "I've got your document"
        },
    ]
    user_inputs_batch = [
        user_prompt,
    ]
    print(f"Context length: {get_context_length(tokenizer, context_messages)}")
    return gen_prompts(tokenizer, context_messages, user_inputs_batch)

def save_outputs(outputs, filename):
    with open(filename, "a") as f:
        for output in outputs:
            generated_text = output.outputs[0].text
            dump_dict = {"generated_text": generated_text}
            f.write(json.dumps(dump_dict) + "\n")

def longer_cache_in_lmcache(model_name, context, question):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["LMCACHE_CONFIG_FILE"] = "example.yaml"
    output_file = "vllm_prefix_output_second.jsonl"
    llm = LLM(model=model_name,
          gpu_memory_utilization=0.6,
          enable_chunked_prefill=False,
          max_model_len=32768,
          enable_prefix_caching=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = gen_input(tokenizer, context, question)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)
    outputs = llm.generate(prompts, sampling_params)
    with open(output_file, "w") as f:
        pass
    save_outputs(outputs, output_file)
    return outputs

if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    context_file = os.path.join(os.pardir, 'ffmpeg.txt')
    question = "What's the name of the tool. Answer with no more than 10 tokens."
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(context_file, 'r') as f:
        context_text = f.read()
    longer_cache_in_lmcache(model_name, context_text, question)
    print("Second script vllm_second.py DONE!")
    os._exit(0)