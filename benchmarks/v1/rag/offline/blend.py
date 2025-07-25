# Standard
from dataclasses import asdict
import argparse
import contextlib
import json
import os
import time

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False, blend_special_str: str = " # # "
):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"

    if use_disk:
        # Disable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "False"

        # Set the maximum size of the local CPU buffer size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

        # Enable local disk backend in LMCache
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"

        # Set the maximum size of the local disk size to 10GB
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        # Enable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "True"

        # Set the maximum size of the local CPU size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(prompt_token_ids=prompt, sampling_params=sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        # print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Specify whether to use disk as backend (default: False)",
    )

    parser.add_argument(
        "-b",
        "--blend-special-str",
        default=" # # ",
        help="Specify the special separators to separate chunks (default: ' # # ')",
    )

    return parser.parse_args()


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def main():
    args = parse_args()
    eval_dataset = load_dataset("/mnt/afs/lihao7/datasets/musique_s.json")
    eval_dataset = eval_dataset[:5]  # Limit to 10 items for testing
    lmcache_connector = "LMCacheConnectorV1"
    model = "mistralai/Mistral-7B-Instruct-v0.2"

    setup_environment_variables(args.use_disk, args.blend_special_str)

    blend_special_str = os.getenv("LMCACHE_BLEND_SPECIAL_STR")
    tokenizer = AutoTokenizer.from_pretrained(model)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        # This example script runs two requests with a shared prefix.
        # Define the shared prompt and specific prompts
        sys_prompt = tokenizer.encode("You are a very helpful assistant.")
        blend_special_prompt = tokenizer.encode(blend_special_str)[1:]

        precompute_prompts = []
        tests_prompts = []

        for index, item in enumerate(eval_dataset):
            # if index != 5:
            #     continue
            item_precompute_prompts = []
            item_test_prompt = sys_prompt[:]
            for doc in item["ctxs"]:
                # Precompute the prompts for each document
                item_precompute_prompts.append(
                    tokenizer.encode(doc["title"] + doc["text"])[1:]
                )

            for doc in item_precompute_prompts:
                precompute_prompts.append(sys_prompt + blend_special_prompt + doc)

            item_test_prompt.extend(blend_special_prompt)
            for doc in item_precompute_prompts:
                item_test_prompt.extend(doc)
                item_test_prompt.extend(blend_special_prompt)
            item_question_prompt = tokenizer.encode(item["question"] + "Answers:")[1:]
            item_test_prompt.extend(item_question_prompt)

            tests_prompts.append(item_test_prompt)
        print("[debug] Precompute prompts:", len(precompute_prompts))
        print("[debug] Tests prompts:", len(tests_prompts))
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

        for prompt in precompute_prompts:
            # Add the first prompt to the cache
            print_output(llm, prompt, sampling_params, "first")
            # Wait for a while to simulate some delay before the second request
        time.sleep(1)
        print("Precompute prompts done, now running tests...")
        print("-" * 50)
        for prompt in tests_prompts:
            # Print the test prompts
            print_output(llm, prompt, sampling_params, "test")


if __name__ == "__main__":
    main()
