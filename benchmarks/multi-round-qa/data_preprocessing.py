import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="Process data percentage.")
parser.add_argument(
    "--parse",
    type=float,
    default=1,
    help="The percentage of data to process (0 to 1). Default is 1 (100%).")

args = parser.parse_args()

with open('ShareGPT_V3_unfiltered_cleaned_split.json', 'r',
          encoding='utf-8') as file:
    data = json.load(file)


def estimate_num_tokens(text: str) -> int:
    if not hasattr(estimate_num_tokens, "tokenizer"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        estimate_num_tokens.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2")
    return len(estimate_num_tokens.tokenizer.tokenize(text))


num_of_ids = len(data)
print(f"Number of IDs: {num_of_ids}")
data = data[:int(num_of_ids * args.parse)]

count = 0

for d in data:
    d['num_round'] = len(
        d['conversations'])  # human is one round, gpt is another round
    human_tokens = []
    gpt_tokens = []
    for conv in d['conversations']:
        if conv['from'] == 'human':
            human_tokens.append(estimate_num_tokens(conv['value']))
        if conv['from'] == 'gpt':
            token_number = estimate_num_tokens(conv['value'])
            conv['num_tokens'] = token_number
            gpt_tokens.append(token_number)
    if len(human_tokens) == 0:
        d['average_human_token'] = 0
        d['max_human_token'] = 0
    else:
        d['average_human_token'] = float(np.mean(human_tokens))
        d['max_human_token'] = float(np.max(human_tokens))
    if len(gpt_tokens) == 0:
        d['average_gpt_token'] = 0
        d['max_gpt_token'] = 0
    else:
        d['average_gpt_token'] = float(np.mean(gpt_tokens))
        d['max_gpt_token'] = float(np.max(gpt_tokens))

    count += 1
    print(f"Finished {count}")

# Remove the data that has two consecutive human rounds
del data[260]

with open('ShareGPT.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
