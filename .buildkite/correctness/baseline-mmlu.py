# Simple MMLU baseline test for single vLLM engine
# ASSUMPTION: A single vLLM engine is running on port 8000

from transformers import AutoTokenizer, set_seed
import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests

global tokenizer
choices = ["A", "B", "C", "D"]

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
    res = requests.post("http://localhost:8000/v1/completions", json=data, timeout=30)
    if res.status_code != 200:
        raise Exception(f"Error: {res.status_code} {res.text}")
    response_json = res.json()
    return response_json["choices"][0]["text"]

def prompt_string(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2 # number of columns - 2 (question and answer)
    for i in range(k):
        prompt += f"\n{choices[i]}. {df.iloc[idx, i + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k]}\n\n"
    return prompt

def evaluate(args, subject, dev_df, test_df):
    prompts, labels = [], []

    shared_multi_shot_prefix = [f"The following are multiple choice questions (with answers) about {subject}.\n\n"]
    shared_multi_shot_prefix_length = 0
    for i in range(dev_df.shape[0]):
        shared_multi_shot_prefix.append(prompt_string(dev_df, i))
        shared_multi_shot_prefix_length += len(tokenizer(shared_multi_shot_prefix[-1], add_special_tokens=True, return_tensors="pt")["input_ids"][0])
        if shared_multi_shot_prefix_length > 4000:
            break

    shared_multi_shot_prefix = "".join(shared_multi_shot_prefix)

    for i in range(test_df.shape[0]):
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

    mmlu_files = os.listdir("../../data/test")
    test_files = [f for f in mmlu_files if f.endswith("_test.csv")]
    subjects = sorted([f.split("_test.csv")[0] for f in test_files])

    accuracies = []
    num_questions = []
    all_cors = []

    for subject_raw in tqdm(subjects[: args.number_of_subjects], desc="Processing subjects"):
        subject = " ".join(subject_raw.split("_"))
        dev_df = pd.read_csv(
            os.path.join("../../data/dev", subject_raw + "_dev.csv"), header=None
        )
        test_df = pd.read_csv(
            os.path.join("../../data/test", subject_raw + "_test.csv"), header=None
        )
        accuracy = evaluate(args, subject, dev_df, test_df)
        accuracies.append(accuracy)
        num_questions.append(len(test_df))
        print(f"Average accuracy {accuracy:.3f} - {subject_raw}")

    total_accuracy = np.mean(accuracies)
    total_num_questions = sum(num_questions)
    
    print(f"Average accuracy: {total_accuracy:.3f}")
    print(f"Total latency: 0.000")  # For compatibility with old format

    # Also create new format output
    output_dict = {}
    for i, subject_raw in enumerate(subjects[: args.number_of_subjects]):
        output_dict[subject_raw] = {
            "accuracy": accuracies[i],
            "num_questions": num_questions[i]
        }
    
    output_dict["total"] = {
        "accuracy": total_accuracy,
        "num_questions": total_num_questions
    }

    with open(args.result_file.replace('.txt', '.jsonl'), "w") as f:
        for subject, value in output_dict.items():
            f.write(json.dumps({subject: value}) + "\n")

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--number-of-subjects", type=int, required=True)

    args = parser.parse_args()
    main(args) 