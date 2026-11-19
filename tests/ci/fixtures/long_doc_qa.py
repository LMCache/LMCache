# SPDX-License-Identifier: Apache-2.0
"""Deterministic, CPU-only stand-in for the long-document QA benchmark."""

# Standard
from pathlib import Path
import argparse
import csv
import json
import os
import sys


def main() -> None:
    """Write phase-specific benchmark outputs using the real CLI contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, choices=(8000, 9000), required=True)
    parser.add_argument("--model", required=True)
    for option in (
        "--document-length",
        "--num-documents",
        "--output-len",
        "--repeat-count",
        "--shuffle-seed",
        "--max-inflight-requests",
    ):
        parser.add_argument(option, type=int, required=True)
    parser.add_argument(
        "--repeat-mode", choices=("tile", "random", "interleave"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", action="store_true", required=True)
    args = parser.parse_args()

    phase = "baseline" if args.port == 9000 else "lmcache"
    prompt_id_offset = 100 if phase == "baseline" else 200
    query_ttft = 1.0 if phase == "baseline" else float(os.environ["STUB_QUERY_TTFT"])
    query_round_time = (
        1.0 if phase == "baseline" else float(os.environ["STUB_QUERY_ROUND_TIME"])
    )
    for round_name, count, ttft, round_time in (
        ("warmup", args.num_documents, 0.6, 1.2),
        (
            "query",
            args.num_documents * args.repeat_count,
            query_ttft,
            query_round_time,
        ),
    ):
        with Path(f"{round_name}_round.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    "prompt_id",
                    "request_start",
                    "ttft",
                    "request_end",
                    "successful",
                    "ttft_time",
                    "is_miss",
                )
            )
            for index in range(count):
                start = index * round_time
                writer.writerow(
                    (
                        prompt_id_offset + index,
                        start,
                        ttft,
                        start + round_time,
                        True,
                        start + ttft,
                        round_name == "warmup",
                    )
                )

    with args.output.open("a", encoding="utf-8") as f:
        f.write(f"{phase} responses\n")
    print(f"{phase} stderr", file=sys.stderr)
    print(f"{phase} benchmark completed")
    print(
        json.dumps(
            {
                "query_ttft_per_prompt": query_ttft,
                "query_round_time_per_prompt": query_round_time,
                "warmup_round_time_per_prompt": 1.2,
            }
        )
    )


if __name__ == "__main__":
    main()
