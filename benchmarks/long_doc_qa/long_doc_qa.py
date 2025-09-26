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
        - 'tile': the entire prompt list is repeated in sequence.
        - 'interleave': each prompt is repeated consecutively before
                        moving to the next element.

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

    --expected-latency: Expected end to end latency for the first query round.
    --completions: Use completions API instead of chat completions API

    --visualize: Visualize the results

    --eos-token-id: EOS token id. we bias against this token id so we always
                   get the number of output tokens we specify

    --hit-miss-ratio: In query round, control how many of the prompts
    will miss the cache. For example, 3:1 means every fourth repeated prompt
"""

# Standard
from dataclasses import dataclass
import argparse
import asyncio
import random
import sys
import time

# Third Party
from openai import AsyncOpenAI
import pandas as pd

# Global output filename (set in __main__)
OUTPUT_FILE = None
completions_mode = False
visualize = False
eos_token_id = None


@dataclass
class RequestStats:
    prompt_id: int
    request_start: float
    ttft: float
    request_end: float


def has_content(chunk):
    """
    Check if the chunk has content in the choices.
    Args:
        chunk: The response chunk from OpenAI Chat Completions API.

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


def has_content_completions(chunk):
    """
    Completions streaming emits text at choices[0].text.
    """
    return bool(chunk.choices) and (chunk.choices[0].text is not None)


def extract_content(chunk):
    """
    Extract content from the response chunk.
    Args:
        chunk: The response chunk from OpenAI Chat Completions API.
    Returns:
        str: The content extracted from the chunk.
    """
    if chunk.choices[0].delta.content is not None:
        return chunk.choices[0].delta.content
    elif chunk.choices[0].delta.reasoning_content is not None:
        return chunk.choices[0].delta.reasoning_content
    else:
        return ""


def extract_content_completions(chunk):
    """
    Extract content from a Completions stream chunk.
    """
    return chunk.choices[0].text or ""


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
) -> RequestStats:
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
        RequestStats: RequestStats object containing the request stats
    """
    async with semaphore:  # Acquire semaphore to limit concurrent requests
        write_resp(f"\n--- Sending prompt {prompt_index + 1}/{total_prompts} ---\n")
        # a request starts once it acquires the semaphore
        start_time = time.time()
        first_token_time = None

        # add stop None so we always get the number of output tokens we specify
        if completions_mode:
            response = await client.completions.create(
                model=model,
                prompt=prompt,
                stream=True,
                max_tokens=output_len,
                temperature=0.0,
                stream_options={"include_usage": True},
                logit_bias={str(eos_token_id): -100}
                if eos_token_id is not None
                else None,
            )

            pieces = []
            async for chunk in response:
                if not has_content_completions(chunk):
                    continue

                content = extract_content_completions(chunk)
                if first_token_time is None and content.strip():
                    first_token_time = time.time()
                pieces.append(content)

            end_time = time.time()
            final_response = "".join(pieces)
            write_resp(f"\nResponse of request {prompt_index}: {final_response}\n")

            # ttft is never 0.0 so it is an immediate tell
            # that the request produced no output
            ttft = (
                (first_token_time - start_time) if first_token_time is not None else 0.0
            )
            return RequestStats(
                prompt_id=prompt_index,
                request_start=start_time,
                ttft=ttft,
                request_end=end_time,
            )

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=output_len,
            temperature=0.0,
            stream_options={"include_usage": True},
            logit_bias={str(eos_token_id): -100} if eos_token_id is not None else None,
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

        end_time = time.time()
        final_response = "".join(responses)
        write_resp(f"\nResponse of request {prompt_index}: {final_response}\n")

        # ttft is never 0.0 so it is an immediate tell
        # that the request produced no output
        ttft = (first_token_time - start_time) if first_token_time is not None else 0.0
        return RequestStats(
            prompt_id=prompt_index,
            request_start=start_time,
            ttft=ttft,
            request_end=end_time,
        )


async def test_long_document_qa(
    client, model, prompts=None, output_len=100, max_inflight_requests=10
) -> list[RequestStats]:
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
        list: request_stats - a list of RequestStats objects
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_inflight_requests)

    # Create tasks for all prompts
    tasks = [
        process_single_prompt(
            client=client,
            model=model,
            prompt=prompt,
            prompt_index=i,
            total_prompts=len(prompts),
            output_len=output_len,
            semaphore=semaphore,
        )
        for i, prompt in enumerate(prompts)
    ]
    # Execute all tasks concurrently and collect results
    # The semaphore will control max concurrent requests
    request_stats = await asyncio.gather(*tasks)

    return request_stats


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


def add_cache_misses(prompts, hit_miss_ratio):
    """
    Add cache misses to the prompts and return a boolean mask aligned with prompts.
    """
    if hit_miss_ratio is None:
        return prompts, [False] * len(prompts)

    hit, miss = map(int, hit_miss_ratio.split(":", 1))
    period = hit + miss
    miss_mask = [False] * len(prompts)

    for i in range(len(prompts)):
        # every (hit+miss) window: first `hit` are hits, rest are misses
        if period and (i % period) >= hit:
            miss_mask[i] = True
            prompts[i] = f"{random.randint(-10_000_000, 10_000_000)} {prompts[i]}"

    return prompts, miss_mask


def relative_time(df, start_time):
    """
    Relative time to the start of the benchmark.
    """
    df["request_start"] = df["request_start"] - start_time
    df["request_end"] = df["request_end"] - start_time
    df["ttft_time"] = df["request_start"] + df["ttft"]


def visualize_results(warmup_df, benchmark_df):
    def plot_bars(df, title, filename):
        plt.figure(figsize=(12, 6))

        if "is_miss" in df.columns:
            is_miss = df["is_miss"]
        else:
            is_miss = pd.Series(False, index=df.index)

        hits = df[~is_miss]
        misses = df[is_miss]

        # Prefill: dark blue (hit), dark orange (miss)
        if not hits.empty:
            plt.barh(
                hits["prompt_id"],
                hits["ttft_time"] - hits["request_start"],
                left=hits["request_start"],
                color="darkblue",
                label="Loading",  # prefill hits
            )
        if not misses.empty:
            plt.barh(
                misses["prompt_id"],
                misses["ttft_time"] - misses["request_start"],
                left=misses["request_start"],
                color="darkorange",
                label="Compute",  # prefill misses
            )

        # Decode: light blue (hit), light orange (miss)
        if not hits.empty:
            plt.barh(
                hits["prompt_id"],
                hits["request_end"] - hits["ttft_time"],
                left=hits["ttft_time"],
                color="skyblue",
                label="Decoding after loading",
            )
        if not misses.empty:
            plt.barh(
                misses["prompt_id"],
                misses["request_end"] - misses["ttft_time"],
                left=misses["ttft_time"],
                color="pink",
                label="Decoding after compute",
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Prompt ID")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_bars(warmup_df, "Warmup Round", "warmup_round.png")
    plot_bars(benchmark_df, "Query Round", "query_round.png")


async def main(args):
    random.seed(args.shuffle_seed)

    # Create the OpenAI client
    # No timeout: some benchmarks can take 4-5 minutes per request
    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="sk-dummy",
        timeout=None,
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
    prompts, miss_mask = add_cache_misses(prompts, args.hit_miss_ratio)

    write_resp("------warm up round------\n")
    warmup_start_time = time.time()
    warmup_request_stats = await test_long_document_qa(
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
    benchmark_request_stats = await test_long_document_qa(
        client=client,
        model=model,
        prompts=prompts,
        output_len=args.output_len,
        max_inflight_requests=args.max_inflight_requests,
    )
    benchmark_end_time = time.time()

    warmup_df = pd.DataFrame([stats.__dict__ for stats in warmup_request_stats])
    relative_time(warmup_df, warmup_start_time)
    warmup_df["is_miss"] = True
    benchmark_df = pd.DataFrame([stats.__dict__ for stats in benchmark_request_stats])
    benchmark_df["is_miss"] = miss_mask
    relative_time(benchmark_df, benchmark_start_time)

    warmup_df.to_csv("warmup_round.csv", index=False)
    benchmark_df.to_csv("query_round.csv", index=False)

    # Print results
    warmup_mean_ttft = warmup_df["ttft"].mean()
    query_mean_ttft = benchmark_df["ttft"].mean()
    CSI = "\x1b["
    RESET = CSI + "0m"
    print(f"{CSI}36;1m\n=== BENCHMARK RESULTS ==={RESET}")
    print(f"{CSI}32mWarmup round mean TTFT: {warmup_mean_ttft:.3f}s{RESET}")
    print(
        f"{CSI}33mWarmup round time: {warmup_end_time - warmup_start_time:.3f}s{RESET}"
    )
    print(f"{CSI}35mWarmup round prompt count: {len(warmup_df)}{RESET}")
    print(f"{CSI}32mQuery round mean TTFT: {query_mean_ttft:.3f}s{RESET}")
    print(
        f"{CSI}33mQuery round time: "
        f"{benchmark_end_time - benchmark_start_time:.3f}s{RESET}"
    )
    print(f"{CSI}35mQuery round prompt count: {len(benchmark_df)}{RESET}")

    if visualize:
        visualize_results(warmup_df, benchmark_df)

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
        warmup_per_prompt = warmup_duration / len(warmup_df)
        query_per_prompt = query_duration / len(benchmark_df)
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

    if args.expected_latency is not None:
        warmup_duration = warmup_end_time - warmup_start_time
        warmup_per_prompt = warmup_duration / len(warmup_df)
        print(f"{CSI}34mActual latency: {warmup_per_prompt:.2f}s{RESET}")
        if warmup_per_prompt > args.expected_latency:
            sys.exit(
                f"ERROR: latency {warmup_per_prompt:.2f}s > expected "
                f"{args.expected_latency:.2f}s"
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
        default=2,
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
    parser.add_argument(
        "--expected-latency",
        type=float,
        default=None,
        help="Expected end to end latency for the first query round.",
    )

    parser.add_argument(
        "--completions",
        action="store_true",
        help="Use completions API instead of chat completions API",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the results",
    )

    parser.add_argument(
        "--hit-miss-ratio",
        type=str,
        default=None,
        help=(
            "In query round, control how many of the prompts will miss the cache."
            "For example, 3:1 means every fourth repeated prompt will be randomized "
            "to force a cache miss. 2:2 means 2 hits and 2 misses"
        ),
    )

    parser.add_argument(
        "--eos-token-id",
        type=int,
        default=None,
        help=(
            "EOS token id. we bias against this token id so we always "
            "get the number of output tokens we specify"
        ),
    )

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    completions_mode = args.completions
    visualize = args.visualize
    if visualize:
        # Third Party
        import matplotlib.pyplot as plt
    if args.eos_token_id is not None:
        eos_token_id = args.eos_token_id
    OUTPUT_FILE = args.output
    asyncio.run(main(args))
