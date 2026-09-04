# SPDX-License-Identifier: Apache-2.0
"""Configuration and logical worker models for ``lmcache bench server``."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import argparse

# First Party
from lmcache.cli.commands.bench.server_bench.cases.base import BenchCase


@dataclass(frozen=True)
class BenchConfig:
    """Immutable client configuration for one server benchmark run.

    CLI-only concerns such as profiling stay with the command layer.

    Attributes:
        rpc_url: Multiprocess request endpoint of the LMCache server.
        http_url: HTTP endpoint used by management and checksum APIs.
        mode: ``"cpu"`` or ``"gpu"`` for mock Worker resources.
        transfer_mode: ``"auto"``, ``"engine_driven"``, or
            ``"lmcache_driven"``.
        tp_size: Number of simulated tensor-parallel Workers.
        use_mla: Whether the CLI explicitly requested MLA routing.
        num_tokens: Synthetic payload tokens per request, excluding the
            sequence token prepended by the benchmark.
        kvcache_shape_spec: LMCache KV shape specification string.
        num_blocks: Fallback number of paged KV blocks.
        block_size: Fallback number of tokens per paged KV block.
    """

    rpc_url: str
    http_url: str
    mode: str
    transfer_mode: str
    tp_size: int
    use_mla: bool
    num_tokens: int
    kvcache_shape_spec: str
    num_blocks: int
    block_size: int

    def __post_init__(self) -> None:
        if self.mode not in ("cpu", "gpu"):
            raise ValueError(f"unsupported device mode: {self.mode}")
        if self.transfer_mode not in (
            "auto",
            "engine_driven",
            "lmcache_driven",
        ):
            raise ValueError(f"unsupported transfer mode: {self.transfer_mode}")
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}")

    @property
    def is_gpu(self) -> bool:
        """Return whether mock Workers use GPU memory."""
        return self.mode == "gpu"

    @property
    def uses_handle_transfer(self) -> bool:
        """Return whether this run uses handle-based transfer.

        ``auto`` selects handles for GPU and engine-driven transfer for CPU.
        """
        if self.transfer_mode == "auto":
            return self.is_gpu
        return self.transfer_mode == "lmcache_driven"


@dataclass(frozen=True)
class WorkerSpec:
    """Logical identity and routing roles of one simulated Worker.

    Attributes:
        rank: Simulated tensor-parallel rank.
        instance_id: Server registration context identifier.
        kv_worker_id: Worker identifier stored in cache keys.
        kv_world_size: KV world size stored in cache keys.
        store_enabled: Whether STORE operations target this Worker.
        retrieve_enabled: Whether RETRIEVE operations target this Worker.
    """

    rank: int
    instance_id: int
    kv_worker_id: int
    kv_world_size: int
    store_enabled: bool
    retrieve_enabled: bool = True

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative, got {self.rank}")
        if self.instance_id < 0:
            raise ValueError(
                f"instance_id must be non-negative, got {self.instance_id}"
            )
        if self.kv_worker_id < 0:
            raise ValueError(
                f"kv_worker_id must be non-negative, got {self.kv_worker_id}"
            )
        if self.kv_world_size < 1:
            raise ValueError(
                f"kv_world_size must be positive, got {self.kv_world_size}"
            )


@dataclass(frozen=True)
class BenchRunSpec:
    """Bind one bench case to a concrete client configuration.

    Attributes:
        config: Client, Worker, and KV layout configuration.
        bench_case: Configured bench case to execute.
    """

    config: BenchConfig
    bench_case: BenchCase


def parse_args_to_config(args: argparse.Namespace) -> BenchConfig:
    """Build a config while preserving CLI defaults and TP clamping."""
    return BenchConfig(
        rpc_url=args.rpc_url,
        http_url=args.url,
        mode=args.mode,
        transfer_mode=getattr(args, "transfer_mode", "auto"),
        tp_size=max(1, int(getattr(args, "tp_size", 1))),
        use_mla=bool(getattr(args, "use_mla", False)),
        num_tokens=args.num_tokens,
        kvcache_shape_spec=args.kvcache_shape_spec,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
    )
