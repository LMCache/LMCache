# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_long_document_qa_throughput.py

"""
Commandline arguments:
    --num-documents: The number of documents to sample prompts from.

    --document-length: The length of each document in tokens.
                       (Optional, default: 20000)

    --output-len: The number of tokens to generate for each prompt.
                  (Optional, default: 100)

    --repeat-count: The number of times to repeat each prompt.
                    (Optional, default: 2)

    --repeat-mode: The mode to repeat prompts. The supported modes are:
        - 'random': shuffle the prompts randomly. (Default)
        - 'tile': the entire prompt list is repeated in sequence. (Potentially
                  lowest cache hit)
        - 'interleave': each prompt is repeated consecutively before
                        moving to the next element. (Highest cache hit)

    --shuffle-seed: Random seed when the repeat mode is "random".
                    (Optional, default: 0)

    --port: Port to query the vLLM server

    --model: Model name

    --max-inflight-requests: Maximum number of in-flight requests. Default is 2

    --sleep-time-after-warmup: Sleep time after warm up iteration.
                              (Optional, default: 0.0 seconds)

    --output: Filename to write all responses to. If omitted, writes to stdout.

    --expected-ttft-gain: Expected minimum speed-up in time-to-first-token
                         (warmup/query) as a factor, e.g. 4.3 for 4.3×. If
                         actual gain is below this, exits.

    --expected-latency-gain: Expected minimum speed-up in total round time
                            (warmup/query) as a factor, e.g. 4.5 for 4.5×.
                            If actual gain is below this, exits.
"""

# Standard
import argparse
import asyncio
import random
import sys
import time

# Third Party
from openai import AsyncOpenAI

# Global output filename (set in __main__)
OUTPUT_FILE = None


def has_content(chunk):
    """
    Check if the chunk has content in the choices.
    Args:
        chunk: The response chunk from OpenAI API.

    Returns:
        bool: True if content exists, False otherwise.
    """
    return (
        chunk.choices
        and chunk.choices[0].delta
        and (
            chunk.choices[0].delta.content is not None
            or chunk.choices[0].delta.reasoning_content is not None
        )
    )


def extract_content(chunk):
    """
    Extract content from the response chunk.
    Args:
        chunk: The response chunk from OpenAI API.
    Returns:
        str: The content extracted from the chunk.
    """
    if chunk.choices[0].delta.content is not None:
        return chunk.choices[0].delta.content
    elif chunk.choices[0].delta.reasoning_content is not None:
        return chunk.choices[0].delta.reasoning_content
    else:
        return ""


def write_resp(text: str):
    """
    Write text to the specified output file (if any), otherwise to stdout.
    """
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, "a") as resp_file:
            resp_file.write(text)
    else:
        sys.stdout.write(text)


async def process_single_prompt(
    client, model, prompt, prompt_index, total_prompts, output_len, semaphore
):
    """
    Process a single prompt with the given client and model.

    Args:
        client: The OpenAI client for making API calls.
        model: The model name to use for generation.
        prompt: The prompt string to be processed.
        prompt_index: Index of the current prompt (0-based).
        total_prompts: Total number of prompts being processed.
        output_len: The maximum number of tokens to generate.
        semaphore: Asyncio semaphore to limit concurrent requests.

    Returns:
        float: Time-to-first-token measurement
    """
    async with semaphore:  # Acquire semaphore to limit concurrent requests
        write_resp(f"\n--- Sending prompt {prompt_index + 1}/{total_prompts} ---\n")
        start_time = time.time()
        first_token_time = None
        words = ""

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=output_len,
            temperature=0.0,
            stream_options={"include_usage": True},
        )

        responses = []
        # Collect the response chunks
        async for chunk in response:
            if not chunk.choices:
                continue

            # Handle content for chat completions
            if has_content(chunk):
                content = extract_content(chunk)
                if first_token_time is None and content != "":
                    first_token_time = time.time()
                responses.append(content)
                words += content

        final_response = "".join(responses)
        write_resp(f"\nResponse of request {prompt_index}: {final_response}\n")

        if first_token_time is not None:
            return first_token_time - start_time
        else:
            # If no content was generated, return a default value
            return 0.0


async def test_long_document_qa(
    client, model, prompts=None, output_len=100, max_inflight_requests=10
):
    """
    Test long document QA with the given prompts and sampling parameters.
    Process prompts concurrently with a limit on inflight requests.

    Args:
        client: The OpenAI client for making API calls.
        model: The model name to use for generation.
        prompts: A list of prompt strings to be processed by the LLM.
        output_len: The maximum number of tokens to generate.
        max_inflight_requests: Maximum number of concurrent requests.

    Returns:
        list: ttfts - a list of time-to-first-token measurements
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_inflight_requests)

    # Create tasks for all prompts
    tasks = []
    for i, prompt in enumerate(prompts):
        task = process_single_prompt(
            client=client,
            model=model,
            prompt=prompt,
            prompt_index=i,
            total_prompts=len(prompts),
            output_len=output_len,
            semaphore=semaphore,
        )
        tasks.append(task)

    # Execute all tasks concurrently and collect results
    ttfts = await asyncio.gather(*tasks)

    return ttfts


def repeat_prompts(prompts, repeat_count, mode: str):
    """
    Repeat each prompt in the list for a specified number of times.
    The order of prompts in the output list depends on the mode.

    Args:
        prompts: A list of prompts to be repeated.
        repeat_count: The number of times each prompt is repeated.
        mode: The mode of repetition. Supported modes are:
            - 'random': Shuffle the prompts randomly after repetition.
            - 'tile': Repeat the entire prompt list in sequence.
              Example: [1, 2, 3] -> [1, 2, 3, 1, 2, 3].
            - 'interleave': Repeat each prompt consecutively before moving to
              the next. Example: [1, 2, 3] -> [1, 1, 2, 2, 3, 3].

    Returns:
        A list of repeated prompts in the specified order.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    write_resp(f"Repeat mode:  {mode}\n")
    if mode == "random":
        repeated_prompts = prompts * repeat_count
        random.shuffle(repeated_prompts)
        return repeated_prompts
    elif mode == "tile":
        return prompts * repeat_count
    elif mode == "interleave":
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * repeat_count)
        return repeated_prompts
    else:
        raise ValueError(
            f"Invalid mode: {mode}, only support 'random', 'tile', 'interleave'"
        )


async def main(args):
    random.seed(args.shuffle_seed)

    # Create the OpenAI client
    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1", api_key="sk-dummy"
    )
    model = args.model

    pre_warmup_prompts = [str(i) + "xx" + " ".join(["hi"] * 1000) for i in range(5)]

    await test_long_document_qa(
        client=client,
        model=model,
        prompts=pre_warmup_prompts,
        output_len=args.output_len,
        max_inflight_requests=args.max_inflight_requests,
    )

    # Prepare the prompts:
    # we append the document id at the beginning to avoid any of the document
    # being the prefix of other documents
    warmup_prompts = [
        str(i) + " " + " ".join(["hi"] * args.document_length)
        for i in range(args.num_documents)
    ]

    prompts = repeat_prompts(warmup_prompts, args.repeat_count, mode=args.repeat_mode)

    write_resp("------warm up round------\n")
    warmup_start_time = time.time()
    warmup_ttfts = await test_long_document_qa(
        client=client,
        model=model,
        prompts=warmup_prompts,
        output_len=args.output_len,
        max_inflight_requests=args.max_inflight_requests,
    )
    warmup_end_time = time.time()
    write_resp("------query round------\n")

    sleep_time_after_warmup = args.sleep_time_after_warmup
    if sleep_time_after_warmup > 0:
        write_resp(f"Sleeping for {sleep_time_after_warmup} seconds after warmup...\n")
        time.sleep(sleep_time_after_warmup)

    benchmark_start_time = time.time()
    benchmark_ttfts = await test_long_document_qa(
        client=client,
        model=model,
        prompts=prompts,
        output_len=args.output_len,
        max_inflight_requests=args.max_inflight_requests,
    )
    benchmark_end_time = time.time()

    # Print results
    warmup_mean_ttft = sum(warmup_ttfts) / len(warmup_ttfts)
    query_mean_ttft = sum(benchmark_ttfts) / len(benchmark_ttfts)
    CSI = "\x1b["
    RESET = CSI + "0m"
    print(f"{CSI}36;1m\n=== BENCHMARK RESULTS ==={RESET}")
    print(f"{CSI}32mWarmup round mean TTFT: {warmup_mean_ttft:.3f}s{RESET}")
    print(
        f"{CSI}33mWarmup round time: {warmup_end_time - warmup_start_time:.3f}s{RESET}"
    )
    print(f"{CSI}35mWarmup round prompt count: {len(warmup_ttfts)}{RESET}")
    print(f"{CSI}32mQuery round mean TTFT: {query_mean_ttft:.3f}s{RESET}")
    print(
        f"{CSI}33mQuery round time: "
        f"{benchmark_end_time - benchmark_start_time:.3f}s{RESET}"
    )
    print(f"{CSI}35mQuery round prompt count: {len(benchmark_ttfts)}{RESET}")

    # Validate expected gains as multiplicative speed-ups
    if args.expected_ttft_gain is not None:
        actual_ttft_gain = (
            warmup_mean_ttft / query_mean_ttft if query_mean_ttft > 0 else float("inf")
        )
        print(f"{CSI}34mActual TTFT gain: {actual_ttft_gain:.2f}×{RESET}")
        if actual_ttft_gain < args.expected_ttft_gain:
            sys.exit(
                f"ERROR: TTFT gain {actual_ttft_gain:.2f}× < expected "
                f"{args.expected_ttft_gain:.2f}×"
            )

    if args.expected_latency_gain is not None:
        warmup_duration = warmup_end_time - warmup_start_time
        query_duration = benchmark_end_time - benchmark_start_time

        # compute per-prompt latency before comparing
        warmup_per_prompt = warmup_duration / len(warmup_ttfts)
        query_per_prompt = query_duration / len(benchmark_ttfts)
        actual_latency_gain = (
            warmup_per_prompt / query_per_prompt
            if query_per_prompt > 0
            else float("inf")
        )
        print(f"{CSI}34mActual latency gain: {actual_latency_gain:.2f}×{RESET}")
        if actual_latency_gain < args.expected_latency_gain:
            sys.exit(
                f"ERROR: latency gain {actual_latency_gain:.2f}× < expected "
                f"{args.expected_latency_gain:.2f}×"
            )


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark the performance with or "
        "without automatic prefix caching."
    )

    parser.add_argument(
        "--document-length",
        type=int,
        # Roughly the number of tokens for a system paper,
        # excluding images
        default=20000,
        help="Length of each document in tokens.",
    )

    parser.add_argument(
        "--num-documents",
        type=int,
        default=8,
        help="Number of documents to generate for testing.",
    )

    parser.add_argument(
        "--output-len",
        type=int,
        default=100,
        help="Maximum number of tokens to generate for each prompt.",
    )

    parser.add_argument(
        "--repeat-count",
        type=int,
        default=2,
        help="Number of times to repeat each prompt",
    )

    parser.add_argument(
        "--repeat-mode",
        type=str,
        default="random",
        help="The mode to repeat prompts. The supported "
        'modes are "random", "tile", and "interleave". '
        "See repeat_prompts() in the source code for details.",
    )

    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help='Random seed when the repeat mode is "random"',
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to query the vLLM server",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name",
    )

    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        default=20,
        help="Maximum number of concurrent inflight requests",
    )

    parser.add_argument(
        "--sleep-time-after-warmup",
        type=float,
        default=0.0,
        help="Sleep time after warm up iteration",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Filename to write all responses to; if omitted, writes to stdout.",
    )
    parser.add_argument(
        "--expected-ttft-gain",
        type=float,
        default=None,
        help=(
            "Expected minimum speed-up in time-to-first-token (warmup/query) "
            "as a factor, e.g. 4.3 for 4.3×. If actual gain is below this, exits."
        ),
    )
    parser.add_argument(
        "--expected-latency-gain",
        type=float,
        default=None,
        help=(
            "Expected minimum speed-up in total round time (warmup/query) "
            "as a factor, e.g. 4.5 for 4.5×. If actual gain is below this, exits."
        ),
    )

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    OUTPUT_FILE = args.output
    asyncio.run(main(args))
