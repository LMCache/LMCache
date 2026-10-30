# SPDX-License-Identifier: Apache-2.0
"""Stand-in for ``vllm bench serve`` used by CPU-only workflow tests.

The stub writes the same result schema the real benchmark emits, so the
workload scripts can be exercised without a GPU or a vLLM install.
"""

# Standard
import argparse
import json
import os
import pathlib


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--random-input-len", type=int, required=True)
    parser.add_argument("--random-output-len", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--result-filename", required=True)
    parser.add_argument("--save-result", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Write a deterministic result JSON and record the effective seed."""
    args = _parse_args()

    baseline_port = int(os.environ.get("STUB_BASELINE_PORT", "9000"))
    if args.port == baseline_port:
        side = "baseline"
        throughput = float(os.environ.get("STUB_BASELINE_THROUGHPUT", "100.0"))
    else:
        side = "lmcache"
        throughput = float(os.environ.get("STUB_LMCACHE_THROUGHPUT", "100.0"))

    result = {
        "total_input_tokens": args.num_prompts * args.random_input_len,
        "completed": args.num_prompts,
        "total_token_throughput": throughput,
    }

    result_dir = pathlib.Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / args.result_filename).write_text(json.dumps(result), encoding="utf-8")

    calls_log = os.environ.get("STUB_CALLS_LOG")
    if calls_log:
        with open(calls_log, "a", encoding="utf-8") as handle:
            handle.write(
                f"{side} seed={args.seed} port={args.port} "
                f"filename={args.result_filename}\n"
            )


if __name__ == "__main__":
    main()
