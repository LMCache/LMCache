# SPDX-License-Identifier: Apache-2.0
"""Workload definitions and factory for ``lmcache bench engine``.

Each workload module defines its own config dataclass and workload
class. The ``create_workload`` factory selects the right workload
based on ``EngineBenchConfig.workload``, resolves the workload-specific
config from CLI args, and returns the workload instance.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.bench.engine_bench.config import EngineBenchConfig
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import (
    RequestSender,
)
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload
from lmcache.cli.commands.bench.engine_bench.workloads.long_doc_qa import (
    LongDocQAConfig,
    LongDocQAWorkload,
)

__all__ = [
    "BaseWorkload",
    "LongDocQAConfig",
    "LongDocQAWorkload",
    "create_workload",
]

_WORKLOAD_NAMES = ("long-doc-qa",)


def create_workload(
    config: EngineBenchConfig,
    args: argparse.Namespace,
    request_sender: RequestSender,
    stats_collector: StatsCollector,
    progress_monitor: ProgressMonitor,
) -> BaseWorkload:
    """Resolve workload-specific config and create the workload instance.

    Dispatches on ``config.workload`` to the appropriate workload module,
    resolves the workload-specific config from ``args`` and ``config``,
    and returns the workload instance ready to ``run()``.

    Args:
        config: Fully-resolved general benchmark config.
        args: Raw CLI args namespace (contains workload-specific flags).
        request_sender: Shared request sender instance.
        stats_collector: Shared stats collector instance.
        progress_monitor: Shared progress monitor instance.

    Returns:
        A concrete BaseWorkload instance.

    Raises:
        ValueError: If the workload name is not recognized.
    """
    if config.workload == "long-doc-qa":
        workload_config = LongDocQAConfig.resolve(
            kv_cache_volume_gb=config.kv_cache_volume_gb,
            tokens_per_gb_kvcache=config.tokens_per_gb_kvcache,
            document_length=args.document_length,
            query_per_document=args.query_per_document,
            shuffle_policy=args.shuffle_policy,
            num_inflight_requests=args.num_inflight_requests,
        )
        return LongDocQAWorkload(
            config=workload_config,
            request_sender=request_sender,
            stats_collector=stats_collector,
            progress_monitor=progress_monitor,
            seed=config.seed,
        )

    raise ValueError(
        f"Unknown workload {config.workload!r}. Available: {', '.join(_WORKLOAD_NAMES)}"
    )
