# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of cpu offloading
with LMCache.

Note that `pip install lmcache` is needed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""
import time
from typing import Tuple
import openai
import pandas as pd
import traceback
import argparse

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FILES = [
    'dataset/samsum_processed_v3.csv'
]

# Sends each CSV entry as a separate request
# No index insertion or CSV reuse logic

def execute_openai_request_with_output(row, model: str, client: openai.Client) -> Tuple[float, float, float, float, str]:
    """
    Execute a single request to the OpenAI engine
    Returns: start_time (seconds), TTFT (seconds), finish_time (seconds), throughput (tokens per second), and generated text
    """
    # Build the prompt using your template
    prompt = f"This is user {row.index_in_dataset} in {row.dataset}.\n\n"
    prompt += f"""
    You are an expert at summarising conversations into concise summaries.  
    Below are example contexts (with their reference summaries) to illustrate the desired format:

    {row.context}

    Now, using that style, please summarise the following dialogue.
    **Do not include any heading or the word “Summary:” — just output the summary text.**

    {row.input}
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0,
            stream=True,
            max_tokens=100,
        )
        # Record the time when request starts
        start_time = time.perf_counter()
        first_token_time = None
        ntokens = 0
        output_parts = []
        for chunk in chat_completion:
            content = chunk.choices[0].delta.content
            if content is not None:
                if first_token_time is None and content.strip():
                    first_token_time = time.perf_counter()
                output_parts.append(content)
                ntokens += 1
        end_time = time.perf_counter()

        ttft = first_token_time - start_time
        finish_time = end_time - start_time
        throughput = ntokens / (end_time - first_token_time)
    except Exception as e:
        traceback.print_exc()
        print(f"OpenAI request failed: {e}")
        return -1, -1, -1, -1, "ERROR"

    return start_time, ttft, finish_time, throughput, "".join(output_parts)


def create_openai_client(port: int, model) -> openai.Client:
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client


def main():
    # parse output filename
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", "-o",
        default="result.csv",
        help="Output CSV file name"
    )
    parser.add_argument(
        "--port", "-p",
        default=8000,
        type=int,
        help="Port number for OpenAI API"
    )
    args = parser.parse_args()

    # Load the workload trace
    workload_trace = pd.read_csv(FILES[0])

    # Initialize OpenAI client
    client = create_openai_client(args.port, MODEL)

    # Prepare result lists
    start_times = []
    answers = []
    ttfts = []
    finish_times = []
    throughputs = []

    # Iterate over each row in workload_trace
    for row in workload_trace.itertuples():
        # Execute the OpenAI request
        st, ttft, finish_time, throughput, generated_answer = execute_openai_request_with_output(
            row, MODEL, client
        )
        start_times.append(st)
        answers.append(generated_answer)
        ttfts.append(ttft)
        finish_times.append(finish_time)
        throughputs.append(throughput)

    # Add the results as new columns to the DataFrame
    workload_trace["start_time"] = start_times
    workload_trace["answer"] = answers
    workload_trace["ttft"] = ttfts
    workload_trace["finish_time"] = finish_times
    workload_trace["throughput"] = throughputs

    # Save the results
    workload_trace.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
