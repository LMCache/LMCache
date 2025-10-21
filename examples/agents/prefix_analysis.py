#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import OrderedDict
from typing import List, Optional, Tuple, Union
import argparse
import json

# Third Party
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Constants
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B"
DEFAULT_TOKENS_PER_GB = 8200  # Default for Llama-3.1; More details here: https://docs.lmcache.ai/getting_started/kv_cache_calculator.html
DEFAULT_POOL_SIZES_GB: List[Union[int, float, str]] = [
    1,
    2,
    4,
    8,
    16,
    32,
    50,
    100,
    200,
    500,
    "unlimited",
]


class LRUTokenPool:
    """
    Token pool with LRU eviction policy based on token count limit.
    For request i (1-indexed):
    y[i] = y[i-1] + (len(tokens[i]) - max_shared_prefix(tokens[i], any previous))
    """

    def __init__(self, max_tokens: float) -> None:
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.requests: OrderedDict[int, List[int]] = OrderedDict()

    def longest_prefix_len(self, tokens: List[int]) -> Tuple[int, int]:
        """
        Find longest prefix match and update LRU ordering.

        Returns:
            Tuple of (prefix_length, matching_request_id)
        """
        best_len = 0
        best_id = -1

        for req_id, req_tokens in self.requests.items():
            common_len = 0
            for i in range(min(len(tokens), len(req_tokens))):
                if tokens[i] == req_tokens[i]:
                    common_len += 1
                else:
                    break

            if common_len > best_len:
                best_len = common_len
                best_id = req_id

        # Update LRU ordering
        if best_id != -1:
            self.requests.move_to_end(best_id)

        return best_len, best_id

    def add_request(self, request_id: int, tokens: List[int]) -> None:
        """Add a request to the pool, evicting LRU entries if necessary."""
        # Evict until we have space
        while self.current_tokens + len(tokens) > self.max_tokens and self.requests:
            old_id, old_tokens = self.requests.popitem(last=False)
            self.current_tokens -= len(old_tokens)

        # Add new request
        self.requests[request_id] = tokens
        self.current_tokens += len(tokens)


def load_and_tokenize_inputs(
    jsonl_path: str, tokenizer_name: str = DEFAULT_TOKENIZER
) -> List[List[int]]:
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Reading and tokenizing inputs from: {jsonl_path}")
    tokenized_sequences = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Tokenizing"):
        try:
            data = json.loads(line.strip())
            input_text = data.get("input", "")
            tokens = tokenizer.encode(input_text)
            tokenized_sequences.append(tokens)
        except Exception as e:
            print(f"Warning: Failed to process line: {e}")
            tokenized_sequences.append([])

    return tokenized_sequences


def calculate_hit_rate(
    token_sequences: List[List[int]], pool_size: Optional[int] = None
) -> float:
    # Use float('inf') for unlimited case to avoid eviction
    max_tokens = float("inf") if pool_size is None else pool_size
    pool = LRUTokenPool(max_tokens)

    total_tokens = 0
    hit_tokens = 0

    for idx, tokens in enumerate(token_sequences):
        total_tokens += len(tokens)

        if idx > 0:
            common, _ = pool.longest_prefix_len(tokens)
            hit_tokens += common

        pool.add_request(idx, tokens)

    return hit_tokens / total_tokens if total_tokens > 0 else 0.0


def analyze_hit_rates_across_pool_sizes(
    token_sequences: List[List[int]],
    pool_sizes_gb: List[Union[int, float, str]],
    tokens_per_gb: int,
) -> Tuple[List[float], List[str]]:
    print("\nAnalyzing hit rates across pool sizes...")
    print("=" * 60)

    hit_rates = []
    x_labels = []

    for size_gb in pool_sizes_gb:
        if size_gb == "unlimited":
            size_tokens = None
            x_labels.append("∞")
            pool_desc = "unlimited"
            token_desc = ""
        else:
            size_tokens = int(size_gb * tokens_per_gb)
            x_labels.append(str(int(size_gb)))
            pool_desc = f"{size_gb}GB"
            token_desc = f" ({size_tokens:,} tokens)"

        print(f"Testing pool size: {pool_desc}{token_desc}")
        hit_rate = calculate_hit_rate(token_sequences, size_tokens)
        hit_rates.append(hit_rate)
        print(f"  Hit rate: {hit_rate:.4f} ({hit_rate * 100:.2f}%)\n")

    print("=" * 60)
    return hit_rates, x_labels


def plot_hit_rates(
    hit_rates: List[float], x_labels: List[str], output_path: str
) -> None:
    """
    Generate and save the hit rate vs pool size plot.

    Args:
        hit_rates: List of hit rates
        x_labels: X-axis labels (pool sizes)
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 7))
    plt.plot(
        range(len(hit_rates)),
        hit_rates,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
    )

    plt.xlabel("Pool Size (GB)", fontsize=12, fontweight="bold")
    plt.ylabel("Hit Rate", fontsize=12, fontweight="bold")
    plt.title("Prefix Cache Hit Rate vs Pool Size", fontsize=14, fontweight="bold")
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(0, min(1.0, max(hit_rates) * 1.1))

    for i, (rate, label) in enumerate(zip(hit_rates, x_labels, strict=False)):
        plt.annotate(
            f"{rate * 100:.1f}%",
            xy=(i, rate),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze prefix cache hit rates across different pool sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i trace.jsonl
  %(prog)s -i trace.jsonl -o custom_output.png
  %(prog)s -i trace.jsonl --pool-sizes 1 2 4 8 16 unlimited
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file (trace.jsonl)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="prefix_cache_hit_rate.png",
        help="Path to output plot file (PNG) (default: prefix_cache_hit_rate.png)",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"HuggingFace tokenizer model name (default: {DEFAULT_TOKENIZER})",
    )

    parser.add_argument(
        "--tokens-per-gb",
        type=int,
        default=DEFAULT_TOKENS_PER_GB,
        help=f"Conversion factor from GB to tokens "
        f"(default: {DEFAULT_TOKENS_PER_GB}). "
        "This should be adjusted when using a different tokenizer.",
    )

    parser.add_argument(
        "--pool-sizes",
        nargs="+",
        default=None,
        help='Pool sizes in GB to test (space-separated, can include "unlimited"). '
        f"Default: {' '.join(map(str, DEFAULT_POOL_SIZES_GB))}",
    )

    return parser.parse_args()


def parse_pool_sizes(
    pool_sizes_input: Optional[List[str]],
) -> List[Union[int, float, str]]:
    if pool_sizes_input is None:
        return DEFAULT_POOL_SIZES_GB

    parsed_sizes: List[Union[int, float, str]] = []
    for size in pool_sizes_input:
        if size.lower() == "unlimited":
            parsed_sizes.append("unlimited")
        else:
            try:
                parsed_sizes.append(float(size))
            except ValueError:
                raise ValueError(
                    f"Invalid pool size: {size}. Must be a number or 'unlimited'"
                ) from None

    return parsed_sizes


def main() -> None:
    args = parse_arguments()

    # Parse pool sizes
    pool_sizes_gb = parse_pool_sizes(args.pool_sizes)

    print("Configuration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Tokens per GB: {args.tokens_per_gb}")
    print(f"  Pool sizes: {pool_sizes_gb}\n")

    # Load and tokenize inputs
    token_sequences = load_and_tokenize_inputs(args.input, args.tokenizer)
    print(f"Loaded {len(token_sequences)} requests")

    # Analyze hit rates
    hit_rates, x_labels = analyze_hit_rates_across_pool_sizes(
        token_sequences, pool_sizes_gb, args.tokens_per_gb
    )

    # Generate plot
    plot_hit_rates(hit_rates, x_labels, args.output)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
