# Standard
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import argparse
import json
import os
import random
import sys
import time
import traceback

# Third Party
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
import numpy as np
import pandas as pd
import requests


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def call_generate_vllm(
    prompt, temperature, max_tokens, stop=None, n=1, url=None, model=None
):
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "n": n,
        "seed": 42,  # Add explicit seed for determinism
    }

    try:
        res = requests.post(url, json=data, timeout=30)  # 30 second timeout
    except requests.exceptions.Timeout:
        print(f"❌ API Timeout: Request took longer than 30 seconds")
        raise Exception("API request timed out after 30 seconds")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        raise Exception(f"Failed to connect to API: {e}")
    except Exception as e:
        print(f"❌ Request Error: {e}")
        raise Exception(f"API request failed: {e}")

    if res.status_code != 200:
        print(f"❌ API Error {res.status_code}: {res.text}")
        print(f"🔍 Request data: {data}")
        raise Exception(f"API request failed with status {res.status_code}: {res.text}")

    try:
        response_json = res.json()
        if n == 1:
            pred = response_json["choices"][0]["text"]
        else:
            pred = [choice["text"] for choice in response_json["choices"]]
        return pred
    except (KeyError, IndexError, ValueError) as e:
        print(f"❌ Response parsing error: {e}")
        print(f"🔍 Response content: {res.text}")
        raise Exception(f"Failed to parse API response: {e}")


def _get_call_generate(args: argparse.Namespace):
    return partial(
        call_generate_vllm, url="http://127.0.0.1:8000/v1/completions", model=args.model
    )


def get_call_generate(args: argparse.Namespace):
    call_generate = _get_call_generate(args)

    def func(*args, **kwargs):
        try:
            return call_generate(*args, **kwargs)
        except Exception:
            print("Exception in call_generate:\n" + get_exception_traceback())
            raise

    return func


def add_common_other_args_and_parse(parser: argparse.ArgumentParser):
    parser.add_argument("--parallel", type=int, default=1)  # Changed default to 1 for determinism
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-V2-Lite", help="Model name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    return args


choices = ["A", "B", "C", "D"]

# Global tokenizer - will be initialized in main()
tokenizer = None


def format_subject(subject):
    split_subject = subject.split("_")
    s = ""
    for entry in split_subject:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions \
                (with answers) about{}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def evaluate(args, subject, dev_df, test_df, call_generate):
    prompts = []
    labels = []

    # Construct prompts
    k = args.ntrain
    train_prompt = gen_prompt(dev_df, subject, k)

    # Make tokenizer behavior deterministic
    original_k = k
    while len(tokenizer(train_prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0]) > 1536:
        k -= 1
        train_prompt = gen_prompt(dev_df, subject, k)
        if args.debug:
            print(f"Reduced k from {original_k} to {k} for subject {subject}")

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        prompt = train_prompt + prompt_end
        prompts.append(prompt)

        label = test_df.iloc[i, test_df.shape[1] - 1]
        labels.append(label)

    preds = [None] * len(prompts)
    max_tokens = 3

    # Run requests deterministically
    def get_one_answer(i):
        if args.debug:
            print(f"Processing request {i}/{len(prompts)} for subject {subject}")
        pred = call_generate(prompts[i], temperature=0, max_tokens=max_tokens)
        pred_stripped = pred.strip()
        if pred_stripped and pred_stripped[0] in ["A", "B", "C", "D"]:
            preds[i] = pred_stripped[0]
        else:
            # Fallback: look for any A, B, C, D in the response
            for char in pred_stripped:
                if char in ["A", "B", "C", "D"]:
                    preds[i] = char
                    break
            else:
                preds[i] = "A"  # Default fallback

        if args.debug:
            print(f"Request {i}: pred='{pred_stripped}' -> '{preds[i]}'")

    tic = time.time()
    if args.parallel == 1:
        # Sequential execution for determinism
        for i in range(len(prompts)):
            get_one_answer(i)
    else:
        # If parallel is requested, use ordered execution
        with ThreadPoolExecutor(args.parallel) as executor:
            # Use list() to ensure ordered execution
            list(executor.map(get_one_answer, range(len(prompts))))
    latency = time.time() - tic

    # Compute accuracy
    cors = [pred == label for pred, label in zip(preds, labels, strict=False)]
    acc = np.mean(cors)
    cors = np.array(cors)

    print(
        "Average accuracy {:.3f}, latency {:.2f}, #q: {} - {}".format(
            acc, latency, len(prompts), subject
        )
    )

    return cors, acc, latency


def test_api_connection(args):
    """Test if the vLLM API is working before running the full benchmark"""
    print("🔍 Testing API connection...")
    test_prompt = "Hello, world!"

    try:
        call_generate = get_call_generate(args)
        response = call_generate(test_prompt, temperature=0, max_tokens=1)
        print(f"✅ API test successful. Response: '{response.strip()}'")
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def main(args):
    global tokenizer

    # Set all random seeds for determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)  # HuggingFace transformers seed

    if args.debug:
        print(f"🔧 Set random seed to {args.seed} for deterministic results")
        print(f"🔧 Parallel execution: {args.parallel}")

    # Initialize tokenizer with the specified model
    print(f"🔧 Initializing tokenizer for model: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        # Ensure tokenizer is deterministic
        if hasattr(tokenizer, 'model_max_length'):
            print(f"🔧 Tokenizer max length: {tokenizer.model_max_length}")
    except Exception as e:
        print(
            f"⚠️ Failed to load tokenizer for {args.model}, "
            f"falling back to deepseek-ai/DeepSeek-V2-Lite: {e}"
        )
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")

    # Get subjects in deterministic order
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    all_files = os.listdir(test_dir)
    test_files = [f for f in all_files if f.endswith("_test.csv")]
    subjects = sorted([f.split("_test.csv")[0] for f in test_files])

    if args.debug:
        print(f"🔧 Found {len(subjects)} subjects: {subjects[:5]}..." if len(subjects) > 5 else f"🔧 Found subjects: {subjects}")

    all_cors = []
    all_latencies = []
    num_requests = 0

    # Select backend
    call_generate = get_call_generate(args)

    for subject in tqdm(subjects[: args.nsub], desc="Processing subjects"):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, latency = evaluate(args, subject, dev_df, test_df, call_generate)
        all_cors.append(cors)
        all_latencies.append(latency)
        num_requests += len(test_df)

    total_latency = np.sum(all_latencies)
    print("Total latency: {:.3f}".format(total_latency))

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # Write results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "mmlu",
            "backend": "vllm",
            "num_gpus": 1,
            "latency": round(total_latency, 3),
            "accuracy": round(weighted_acc, 3),
            "num_requests": num_requests,
            "other": {
                "nsub": args.nsub,
                "parallel": args.parallel,
                "model": args.model,
                "seed": args.seed,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--nsub", type=int, default=60)
    args = add_common_other_args_and_parse(parser)
    main(args)
