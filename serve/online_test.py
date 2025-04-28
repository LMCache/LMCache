# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of cpu offloading
with LMCache.

Note that `pip install lmcache` is needed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""
import os
import time
from typing import Tuple
import openai
import pandas as pd
import traceback

NUM_QUERY = 200
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PORT = 8000
FILES = [
    'dataset/qmsum.csv'
]
FILE_TYPE = "sum" # or "qa"
PREFILL_ONLY = True

dataset_entries = 0

# Trace generator
def generate_workload_trace(trace_files, num_query):
    df_list = []
    for file_path in trace_files:
        # Read CSV file
        df = pd.read_csv(file_path)

        global dataset_entries
        dataset_entries += len(df)

        # Rename "Unnamed: 0" to "index_in_dataset" if it exists
        df = df.rename(columns={'Unnamed: 0': 'index_in_dataset'})
        # Derive dataset name from the file name (e.g., "hotpotqa" from "hotpotqa.csv")
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        # Insert the new column "dataset" at the beginning
        df.insert(0, 'dataset', dataset_name)
        df_list.append(df)
    
    # Concatenate all DataFrames
    df_all = pd.concat(df_list)
    
    # Generate the workload trace by sampling with replacement
    run_workload_trace = df_all.sample(n=num_query, replace=True, random_state=42)
    # workload_trace = pd.concat([df_all, run_workload_trace]).reset_index(drop=True)
    workload_trace = pd.concat([df_all]).reset_index(drop=True)

    return workload_trace


def execute_openai_request_with_output(row, model: str, client: openai.Client) -> Tuple[float, float, float, str]:
    """
    Execute a single request to the OpenAI engine
    Returns: TTFT (seconds) and throughput (tokens per second)
    """

    # Build the prompt using your template

    prompt = f"This is user {row.index_in_dataset} in {row.dataset}."

    if FILE_TYPE == "qa":
        prompt += (
            "Answer the question based on the given passages. Only give me the answer and do not output any other words. Answer within 10 words."
            "\n\nThe following are given passages."
            f"{row.context}"
        )
    elif FILE_TYPE == "sum":
        prompt += (
            "Answer the question based on the given passages. Only give me the answer and do not output any other words."
            "\n\nThe following are given passages."
            f"{row.context}"
        )

    # If there's a question column and it is non-empty, append the question prompt
    if hasattr(row, 'question') and row.question.strip():
        if FILE_TYPE == "qa":
            prompt += (
                "\n\nAnswer the question based on the given passages. "
                "Answer the question precisely. Answer within 10 words. Do NOT repeat the question or output any other words. "
                f"Question: {row.question.strip()}\nAnswer:"
            )
        elif FILE_TYPE == "sum":
            prompt += (
                "\n\nAnswer the question based on the given passages. "
                "Answer the question precisely. Do NOT repeat the question or output any other words. "
                f"Question: {row.question.strip()}\nAnswer:"
            )
    
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        chat_completion = client.chat.completions.create(
                messages = messages,
                model = model,
                temperature = 0,
                stream = True,
                max_tokens = 100,
            )

        start_time = time.perf_counter()
        first_token_time = None
        ntokens = 0
        messages = []
        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != " " and chunk_message != "":
                    first_token_time = time.perf_counter()
                messages.append(chunk_message)
                ntokens += 1
        end_time = time.perf_counter()

        ttft = first_token_time - start_time
        finish_time = end_time - start_time
        throughput = ntokens / (end_time - first_token_time)
    except Exception as e:
        traceback.print_exc()
        print(f"OpenAI request failed: {e}")
        return -1, -1, -1, "ERROR"

    return ttft, finish_time, throughput, f"{''.join(messages)}"


def create_openai_client(port: int, model) -> openai.Client:
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"

    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    return client


def main():
    # Load the workload trace
    workload_trace = generate_workload_trace(FILES, NUM_QUERY)

    # Initialize OpenAI client
    client = create_openai_client(PORT, MODEL)

    # Record answers
    answers = []
    ttfts = []
    finish_times = []
    throughputs = []

    # List to store indices of rows to delete
    rows_to_drop = []

    # Iterate over each row in workload_trace
    for row in workload_trace.itertuples():
        # Execute the OpenAI request
        ttft, finish_time, throughput, generated_answer = execute_openai_request_with_output(row, MODEL, client)

        # If ttft equals -1, mark this row for deletion and skip processing its results
        if ttft == -1:
            rows_to_drop.append(row.Index)

        answers.append(generated_answer)
        ttfts.append(ttft)
        finish_times.append(finish_time)
        throughputs.append(throughput)

    # Add the results as new columns to the DataFrame
    workload_trace["answer"] = answers
    workload_trace["ttft"] = ttfts
    workload_trace["finish_time"] = finish_times
    workload_trace["throughput"] = throughputs

    # Delete the rows from workload_trace where ttft was -1
    if not PREFILL_ONLY:
        rows_to_drop = list(set(rows_to_drop) | set(range(dataset_entries)))
    workload_trace = workload_trace.drop(index=rows_to_drop)

    # Save the results
    workload_trace = workload_trace.reset_index(drop=True)
    workload_trace.to_csv("result.csv")

if __name__ == "__main__":
    main()