# SPDX-License-Identifier: Apache-2.0
"""Exercise unified MP instance cleanup through the real ZMQ wire path."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
import argparse
import json
import subprocess
import threading
import time
import uuid

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessageQueueServer
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.server import add_handler_helper


@dataclass
class RecordingStateOwner(InstanceLivenessTarget):
    """Thread-safe state owner used to observe lifecycle release effects."""

    name: str
    instances: set[int]
    released: list[int] = field(default_factory=list)
    drop_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def drop_instance_state(self, instance_id: int) -> None:
        """Release an instance once while counting every fanout attempt.

        Args:
            instance_id: The worker instance identifier.
        """
        with self.lock:
            self.drop_calls += 1
            if instance_id in self.instances:
                self.instances.remove(instance_id)
                self.released.append(instance_id)

    def tracked_instance_count(self) -> int:
        """Return the number of instance records still owned.

        Returns:
            Number of live records.
        """
        with self.lock:
            return len(self.instances)

    def snapshot(self) -> dict[str, int]:
        """Return stable counters for the final evidence document.

        Returns:
            Remaining, released, and fanout-call counts.
        """
        with self.lock:
            return {
                "remaining": len(self.instances),
                "released": len(self.released),
                "unique_released": len(set(self.released)),
                "drop_calls": self.drop_calls,
            }


def _git_head(repo_root: Path) -> str:
    """Return the tested repository revision."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _run_client(
    server_url: str,
    context: zmq.Context,
    instance_ids: list[int],
    duplicates: int,
) -> int:
    """Send one partition of unregister requests over a real MQ client."""
    client = MessageQueueClient(server_url, context)
    requests = 0
    try:
        for instance_id in instance_ids:
            for attempt in range(duplicates):
                request_type = (
                    RequestType.UNREGISTER_KV_CACHE
                    if attempt % 2 == 0
                    else RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT
                )
                assert (
                    client.submit_request(request_type, [instance_id]).result(
                        timeout=10.0
                    )
                    is None
                )
                requests += 1
    finally:
        client.close()
    return requests


def run_e2e(instances: int, workers: int, duplicates: int) -> dict[str, object]:
    """Run concurrent wire-level unregister requests and validate cleanup.

    Args:
        instances: Number of distinct worker instance IDs to register.
        workers: Number of concurrent ZMQ clients.
        duplicates: Requests sent per instance, alternating both wire types.

    Returns:
        Machine-readable environment, workload, owner counters, and invariants.

    Raises:
        ValueError: If any workload parameter is not positive.
        AssertionError: If cleanup or protocol invariants fail.
    """
    if min(instances, workers) <= 0 or duplicates < 2:
        raise ValueError(
            "instances and workers must be positive; duplicates must be at least 2"
        )

    repo_root = Path(__file__).resolve().parents[2]
    instance_ids = list(range(10_000, 10_000 + instances))
    owners = [
        RecordingStateOwner("kv", set(instance_ids)),
        RecordingStateOwner("q", set(instance_ids)),
        RecordingStateOwner("blend", set(instance_ids)),
    ]
    context = zmq.Context()
    server_url = f"inproc://lmcache-unregister-e2e-{uuid.uuid4().hex}"
    management = ManagementModule(
        cast(MPCacheServerContext, object()), liveness_targets=owners
    )
    server = MessageQueueServer(server_url, context)
    unregister_types = {
        RequestType.UNREGISTER_KV_CACHE,
        RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
    }
    for spec in management.get_handlers():
        if spec.request_type in unregister_types:
            add_handler_helper(server, spec.request_type, spec.handler)

    partitions = [instance_ids[offset::workers] for offset in range(workers)]
    started = time.perf_counter()
    server.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_run_client, server_url, context, partition, duplicates)
                for partition in partitions
            ]
            request_count = sum(future.result() for future in futures)
    finally:
        server.close()
        management.close()
        context.term()
    elapsed_seconds = time.perf_counter() - started

    owner_results = {owner.name: owner.snapshot() for owner in owners}
    expected_requests = instances * duplicates
    expected_drop_calls = expected_requests
    invariants = {
        "all_requests_completed": request_count == expected_requests,
        "all_owner_state_released": all(
            result["remaining"] == 0 for result in owner_results.values()
        ),
        "each_instance_released_once_per_owner": all(
            result["released"] == instances and result["unique_released"] == instances
            for result in owner_results.values()
        ),
        "each_request_fanned_out_once_per_owner": all(
            result["drop_calls"] == expected_drop_calls
            for result in owner_results.values()
        ),
        "both_wire_request_types_exercised": duplicates >= 2,
    }
    assert all(invariants.values()), {
        "invariants": invariants,
        "owners": owner_results,
    }
    return {
        "schema_version": 1,
        "git_head": _git_head(repo_root),
        "workload": {
            "instances": instances,
            "clients": workers,
            "requests_per_instance": duplicates,
            "requests": request_count,
        },
        "elapsed_seconds": elapsed_seconds,
        "owners": owner_results,
        "invariants": invariants,
    }


def main() -> int:
    """Parse arguments, run the E2E contract, and print JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--duplicates", type=int, default=4)
    args = parser.parse_args()
    evidence = run_e2e(args.instances, args.workers, args.duplicates)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
