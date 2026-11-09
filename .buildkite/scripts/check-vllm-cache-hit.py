# SPDX-License-Identifier: Apache-2.0

"""Check that a warm vLLM response reports an LMCache cache hit."""

# Standard
import argparse
import json
from pathlib import Path


def cached_tokens(response_path: Path) -> int:
    """Read the LMCache cached-token count from a vLLM response.

    Args:
        response_path (Path): JSON response from a request with
            ``kv_transfer_params.cached_token_stats`` enabled.

    Returns:
        int: Non-negative number of tokens retrieved from LMCache.

    Raises:
        OSError: The response file cannot be read.
        json.JSONDecodeError: The response is not valid JSON.
        ValueError: The response has no valid LMCache cached-token count.
    """
    with response_path.open(encoding="utf-8") as response_file:
        response = json.load(response_file)

    try:
        count = response["kv_transfer_params"]["cached_token_stats"][
            "num_lmcache_cached_tokens"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{response_path}: missing LMCache cached-token count") from exc

    if type(count) is not int or count < 0:
        raise ValueError(f"{response_path}: invalid LMCache cached-token count: {count!r}")
    return count


def main() -> None:
    """Validate cold and warm response files supplied on the command line.

    Raises:
        SystemExit: An input response is invalid or the warm request did not
            report more LMCache cached tokens than the cold request.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cold_response", type=Path)
    parser.add_argument("warm_response", type=Path)
    args = parser.parse_args()

    try:
        cold_count = cached_tokens(args.cold_response)
        warm_count = cached_tokens(args.warm_response)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")

    if warm_count <= cold_count:
        parser.exit(
            1,
            f"FAIL: warm LMCache cached tokens ({warm_count}) must exceed "
            f"cold tokens ({cold_count})\n",
        )
    print(f"PASS: warm LMCache cached tokens {warm_count} > cold {cold_count}")


if __name__ == "__main__":
    main()
