#!/usr/bin/env python3
"""
LMCache Controller ZMQ Benchmark Tool - CLI Entry Point

This tool performs load testing on LMCache Controller using ZMQ interface
to measure message throughput, latency, and system performance.

Test operations:
- BatchedKVOperationMsg: admit/evict messages via PUSH socket
- RegisterMsg/DeRegisterMsg/HeartbeatMsg: worker lifecycle messages
"""

# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import asyncio
import json

# First Party
from lmcache.logging import init_logger
from lmcache.tools.controller_benchmark.benchmark import ZMQControllerBenchmark
from lmcache.tools.controller_benchmark.config import ZMQBenchmarkConfig

logger = init_logger(__name__)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="LMCache Controller ZMQ Benchmark Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--controller-host",
        type=str,
        default="127.0.0.1",
        help="Controller host address",
    )

    parser.add_argument(
        "--monitor-ports",
        type=str,
        default='{"pull":8100,"reply":8101}',
        help='Monitor ports in JSON format, e.g. {"pull":8100,"reply":8101}',
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of KV operations per batch message",
    )

    parser.add_argument(
        "--operations",
        type=str,
        default="admit:70,evict:25,heartbeat:5",
        help="Operation distribution (name:percentage comma-separated)",
    )

    parser.add_argument(
        "--num-instances",
        type=int,
        default=10,
        help="Number of instances to simulate",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers per instance",
    )

    parser.add_argument(
        "--num-locations",
        type=int,
        default=1,
        help="Number of storage locations",
    )

    parser.add_argument(
        "--num-keys",
        type=int,
        default=10000,
        help="Number of unique keys",
    )

    parser.add_argument(
        "--num-hashes",
        type=int,
        default=100,
        help="Number of hashes for P2P lookup operations",
    )

    parser.add_argument(
        "--no-register-first",
        action="store_true",
        help="Skip pre-registering workers before benchmark",
    )

    args = parser.parse_args()

    # Parse monitor ports from JSON
    try:
        monitor_ports = json.loads(args.monitor_ports)
        pull_port = monitor_ports.get("pull", 8100)
        reply_port = monitor_ports.get("reply")
    except json.JSONDecodeError as e:
        logger.error("Failed to parse monitor-ports JSON: %s", e)
        raise ValueError("Invalid monitor-ports format") from e

    # Convert 0.0.0.0 to 127.0.0.1 for client connections
    client_host = (
        "127.0.0.1" if args.controller_host == "0.0.0.0" else args.controller_host
    )
    controller_pull_url = f"{client_host}:{pull_port}"
    controller_reply_url = f"{client_host}:{reply_port}" if reply_port else None

    # Parse operations
    operations = {}
    for op_str in args.operations.split(","):
        if ":" in op_str:
            name, percentage = op_str.split(":", 1)
            operations[name.strip()] = float(percentage.strip())

    # Create config
    config = ZMQBenchmarkConfig(
        controller_pull_url=controller_pull_url,
        controller_reply_url=controller_reply_url,
        duration=args.duration,
        batch_size=args.batch_size,
        operations=operations,
        num_instances=args.num_instances,
        num_workers=args.num_workers,
        num_locations=args.num_locations,
        num_keys=args.num_keys,
        num_hashes=args.num_hashes,
        register_first=not args.no_register_first,
    )

    # Run benchmark
    benchmark = ZMQControllerBenchmark(config)

    try:
        asyncio.run(benchmark.run_benchmark())
        benchmark.print_results()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error("Benchmark failed: %s", e)
        raise e


if __name__ == "__main__":
    main()
