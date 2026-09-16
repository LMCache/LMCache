# SPDX-License-Identifier: Apache-2.0
"""Manually validate cross-server moves and restored GPU KV bytes."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import json
import time

# Third Party
import httpx

OUT: Path
client: httpx.Client
COORD: str
A: str
B: str
SOURCE: str
TARGET: str


def post(url: str, data: dict[str, Any] | None = None) -> Any:
    """Send a worker or move operation and fail on any HTTP error."""
    r = client.post(url, json=data or {})
    if not r.is_success:
        raise RuntimeError(f"{url} {r.status_code} {r.text}")
    return r.json()


def get(url: str) -> Any:
    """Read a JSON response, refusing HTTP errors."""
    r = client.get(url)
    r.raise_for_status()
    return r.json()


def count(node: str, model: str) -> int:
    """Count this case in a dedicated instance directory."""
    return sum(
        row["key"]["model_name"] == model
        for row in get(COORD + f"/directory/keys?instance_id={node}&limit=1000")["keys"]
    )


def counts(model: str, na: int, nb: int) -> None:
    """Wait for asynchronous directory events to reach expected counts."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        actual = (count(SOURCE, model), count(TARGET, model))
        if actual == (na, nb):
            return
        time.sleep(0.25)
    raise AssertionError(f"directory {actual} != {(na, nb)}")


def move(
    model: str,
    tokens: list[int],
    tp: int,
    src: str,
    dst: str,
    keep: bool,
    expected: tuple[int, int, int],
) -> dict[str, Any]:
    """Submit a move and verify counts and repeatable terminal reads."""
    reply = post(
        COORD + "/cache/moves",
        {
            "source_instance_id": src,
            "target_instance_id": dst,
            "model_name": model,
            "world_size": tp,
            "token_ids": tokens,
            "keep_source": keep,
        },
    )
    assert reply["requested"] == 6, reply
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        status = get(COORD + "/cache/moves/" + reply["move_id"])
        if status["status"] != "pending":
            break
        time.sleep(0.1)
    assert status["status"] == "completed", status
    assert tuple(status[k] for k in ("loaded", "missing", "deleted")) == expected, (
        status
    )
    assert status["skipped"] == 0, status
    assert get(COORD + "/cache/moves/" + reply["move_id"]) == status
    print("MOVE", status, flush=True)
    return status


def run() -> None:
    """Run both dtypes and rank counts, writing per-case JSON evidence."""
    results = []
    for dtype in ["float16", "bfloat16"]:
        for tp in [1, 2]:
            model = f"kv-bytecheck-{dtype}-ranks{tp}-{time.time_ns()}"
            tokens = list(range(1536))
            tokens[0] = int(time.time_ns() % 1000000000)
            spec = {"model": model, "dtype": dtype, "tp": tp, "tokens": tokens}
            print("CASE", model, flush=True)
            post(A + "/prepare", dict(spec, role="source"))
            post(B + "/prepare", dict(spec, role="target"))
            original = post(A + "/original")
            assert original["mismatched_bytes"] == 0
            assert len(set(original["checksums"].values())) == len(
                original["checksums"]
            )
            poisoned = post(B + "/poison")
            assert poisoned["mismatched_bytes"] > 0
            post(A + "/store")
            keys = 6 * tp
            counts(model, keys, 0)
            first = move(model, tokens, tp, SOURCE, TARGET, False, (keys, 0, keys))
            counts(model, 0, keys)
            retrieved = post(B + "/retrieve")
            assert retrieved["mismatched_bytes"] == 0, retrieved
            assert retrieved["checksums"] == original["checksums"]
            assert retrieved["untouched_blocks_preserved"]
            print(
                "BYTE_EQUAL B",
                retrieved["compared_bytes"],
                "bytes",
                len(retrieved["checksums"]),
                "rank/layer/chunk checks",
                flush=True,
            )
            copy = move(model, tokens, tp, TARGET, SOURCE, True, (keys, 0, 0))
            counts(model, keys, keys)
            assert post(A + "/poison")["mismatched_bytes"] > 0
            returned = post(A + "/retrieve")
            assert returned["mismatched_bytes"] == 0
            assert returned["checksums"] == original["checksums"]
            assert returned["untouched_blocks_preserved"]
            again = move(model, tokens, tp, SOURCE, TARGET, False, (0, keys, 0))
            counts(model, keys, keys)
            record = {
                "model": model,
                "dtype": dtype,
                "logical_ranks": tp,
                "physical_gpus_per_node": 1,
                "tokens": 1536,
                "layers": 28,
                "kv_heads_per_rank": 8,
                "head_size": 128,
                "source_blocks_permuted": True,
                "destination_blocks_differ": True,
                "original": original,
                "target": retrieved,
                "reverse_copy": returned,
                "negative_control_mismatched_bytes": poisoned["mismatched_bytes"],
                "moves": [first, copy, again],
            }
            (OUT / f"{dtype}-ranks{tp}.json").write_text(json.dumps(record, indent=2))
            results.append(
                {
                    k: record[k]
                    for k in [
                        "model",
                        "dtype",
                        "logical_ranks",
                        "physical_gpus_per_node",
                    ]
                }
            )
            post(A + "/close")
            post(B + "/close")
            print("CASE PASS", dtype, tp, flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print("RESULT PASS", flush=True)


def main() -> None:
    """Parse endpoints and close worker sessions even after a failed assertion."""
    global OUT, client, COORD, A, B, SOURCE, TARGET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--source-worker", required=True)
    parser.add_argument("--target-worker", required=True)
    parser.add_argument("--source-instance", default="move-a")
    parser.add_argument("--target-instance", default="move-b")
    parser.add_argument("--output", type=Path, default=Path("move-results"))
    args = parser.parse_args()
    COORD = args.coordinator.rstrip("/")
    A, B = args.source_worker.rstrip("/"), args.target_worker.rstrip("/")
    SOURCE, TARGET = args.source_instance, args.target_instance
    if A == B or SOURCE == TARGET:
        parser.error("Source and target must be distinct")
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=False)
    with httpx.Client(timeout=180, trust_env=False) as client:
        try:
            get(COORD + "/healthz")
            run()
        finally:
            for worker in (A, B):
                try:
                    post(worker + "/close")
                except Exception as exc:
                    print(f"Worker cleanup failed at {worker}: {exc}", flush=True)


if __name__ == "__main__":
    main()
