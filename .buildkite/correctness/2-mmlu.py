# This is a special MMLU test script that will send each query twice
# (first to 8000 and then to 8001) in order to test the correctness
# of LMCache KV Transfer

# ASSUMPTIONS:
# 1. two lmcache serving engines are running on ports 8000 (as producer) and 8001
#    (as consumer) and connected to the same lmcache server
# 2. the mmlu dataset is in a "data" directory
# 3. all invocations of this script should be run in the same directory
#    (for later consolidation)

# Manually Test KV Transfer first with curl:
"""
# JSON payload (swap model name for your own)
cat > payload.json <<EOF
{
  "model": "deepseek-ai/DeepSeek-V2-Lite",
  "prompt": "The capital of France is\\n" \
            "A. Berlin\\n" \
            "B. Madrid\\n" \
            "C. Paris\\n" \
            "D. Rome\\n" \
            "Answer:",
  "temperature": 0,
  "max_tokens": 3,
  "stop": null,
  "n": 1,
  "seed": 42
}
EOF

curl -X POST http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d @payload.json

curl -X POST http://localhost:8001/v1/completions \
     -H "Content-Type: application/json" \
     -d @payload.json
"""

# Standard
import argparse
import json
import os

# Third Party
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
import numpy as np
import pandas as pd
import requests

global tokenizer
choices = ["A", "B", "C", "D"]


# for complete determinism between runs of MMLU, we should:
# 1. set the seed of LLM requests to a fixed number (42)
# 2. set temperature to 0 on requests
def get_llm_response(args, prompt):
    data = {
        "model": args.model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 3,
        "stop": None,
        "n": 1,
        "seed": 42,  # Add explicit seed for determinism
    }
    # first hit the kv producer (this will block)
    res1 = requests.post("http://localhost:8000/v1/completions", json=data, timeout=30)
    if res1.status_code != 200:
        raise Exception(f"Error: {res1.status_code} {res1.text}")
    # then hit the kv consumer (this will not run until the kv producer is done)
    res2 = requests.post("http://localhost:8001/v1/completions", json=data, timeout=30)
    if res2.status_code != 200:
        raise Exception(f"Error: {res2.status_code} {res2.text}")
    # the kv consumer response is what we actually care about
    response_json = res2.json()
    return response_json["choices"][0]["text"]


# grab the idx'th row of the df and generate a prompt string
# format of the MMLU csvs:
# question,option_A,option_B,option_C,option_D,answer
def prompt_string(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2  # number of columns - 2 (question and answer)
    for i in range(k):
        prompt += f"\n{choices[i]}. {df.iloc[idx, i + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k]}\n\n"
    return prompt


def evaluate(args, subject, dev_df, test_df):
    prompts, labels = [], []

    shared_multi_shot_prefix = [
        f"The following are multiple choice questions (with answers) \
                                about {subject}. \n\n"
    ]
    shared_multi_shot_prefix_length = 0
    for i in range(dev_df.shape[0]):
        # the multi-shot examples should contain answers
        shared_multi_shot_prefix.append(prompt_string(dev_df, i))

        # Use plain list of token IDs, no torch tensors
        token_ids = tokenizer(shared_multi_shot_prefix[-1], add_special_tokens=True)[
            "input_ids"
        ]
        shared_multi_shot_prefix_length += len(token_ids)

        if shared_multi_shot_prefix_length > 4000:
            break

    # all already have double newlines at the end
    shared_multi_shot_prefix = "".join(shared_multi_shot_prefix)

    for i in range(test_df.shape[0]):
        # do NOT include the answer for the actual question we want the LLM to answer
        query_prompt = prompt_string(test_df, i, include_answer=False)
        prompt = f"{shared_multi_shot_prefix}\n\n{query_prompt}"
        prompts.append(prompt)
        label = test_df.iloc[i, test_df.shape[1] - 1]
        labels.append(label)

    predictions = []
    for i, prompt in enumerate(prompts):
        prediction = get_llm_response(args, prompt)
        prediction_stripped = prediction.strip()
        if prediction_stripped and prediction_stripped[0] in ["A", "B", "C", "D"]:
            predictions.append(prediction_stripped[0])
        else:
            # Fallback: look for any A, B, C, D in the response
            for char in prediction_stripped:
                if char in ["A", "B", "C", "D"]:
                    predictions.append(char)
                    break
            else:
                predictions.append("A")  # Default fallback

    accuracy = np.mean(np.array(predictions) == np.array(labels))
    return accuracy


def main(args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    mmlu_files = os.listdir("data/test")
    test_files = [f for f in mmlu_files if f.endswith("_test.csv")]
    subjects = sorted([f.split("_test.csv")[0] for f in test_files])

    accuracies = []
    num_questions = []
    output_dict = {}

    for subject_raw in tqdm(
        subjects[: args.number_of_subjects], desc="Processing subjects"
    ):
        subject = " ".join(subject_raw.split("_"))  # replace underscores with spaces
        dev_df = pd.read_csv(
            os.path.join("data/dev", subject_raw + "_dev.csv"), header=None
        )
        test_df = pd.read_csv(
            os.path.join("data/test", subject_raw + "_test.csv"), header=None
        )
        accuracy = evaluate(args, subject, dev_df, test_df)
        accuracies.append(accuracy)
        num_questions.append(len(test_df))
        output_dict[subject_raw] = {"accuracy": accuracy, "num_questions": len(test_df)}

    total_accuracy = np.mean(accuracies)
    total_num_questions = sum(num_questions)
    output_dict["total"] = {
        "accuracy": total_accuracy,
        "num_questions": total_num_questions,
    }

    with open(args.result_file, "w") as f:
        # output will be a jsonl file
        for subject, value in output_dict.items():
            f.write(json.dumps({subject: value}) + "\n")


if __name__ == "__main__":
    set_seed(42)  # some tokenizers may have randomness
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=False)
    parser.add_argument("--number-of-subjects", type=int, required=True)

    args = parser.parse_args()
    if args.result_file is None:
        # Clean model name if it's a path or has slashes
        model_name = args.model.split("/")[-1]
        args.result_file = f"lmcache-{model_name}.jsonl"

    main(args)
