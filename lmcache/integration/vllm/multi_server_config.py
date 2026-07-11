# SPDX-License-Identifier: Apache-2.0
"""Multi-server parallelism validation for the LMCache MP connector.

Extracted into a standalone module so it can be unit-tested without
importing vLLM or ZMQ — ``_validate_multi_server_config`` only reads
``vllm_config.parallel_config`` attributes and is independent of the
connector's runtime dependencies.
"""

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _validate_multi_server_config(
    vllm_config: "VllmConfig", n_servers: int
) -> None:
    """Validate parallelism constraints for a multi-server deployment.

    Checks:
      1. ``world_size`` is divisible by ``n_servers`` (so ranks split
         evenly into contiguous server blocks).
      2. Multi-server + data parallelism is not supported yet.

    Pipeline parallelism (PP) is supported in all modes — the old PP guard
    was removed because the root cause (``compute_extra_count`` inferring
    MLA from a ``tp > world_size`` heuristic) is now fixed by the explicit
    ``use_mla`` flag in ``IPCCacheServerKey``.

    Raises:
        AssertionError: If ``world_size`` is not divisible by ``n_servers``.
        ValueError: If multi-server + DP is requested, or if the MLA
            multi-server geometry is misaligned (``ranks_per_node`` is
            not a multiple of ``tp_size``).
    """
    pc = vllm_config.parallel_config
    assert pc.world_size % n_servers == 0, (
        f"world_size ({pc.world_size}) must be "
        f"divisible by n_servers ({n_servers})"
    )

    dp_size = getattr(pc, "data_parallel_size", 1)
    if n_servers > 1 and dp_size > 1:
        raise ValueError(
            "LMCacheMPConnector multi-server mode (n_servers > 1) does not "
            f"support data parallelism yet; got dp_size={dp_size}. "
            "DP across multiple LMCache servers will be "
            "supported in a follow-up PR."
        )

    # MLA multi-server alignment: the is_kv_writer formula
    # ``(rank % ranks_per_node) % tp_size == 0`` yields exactly one
    # writer per (server, pipeline-stage) only when
    # ``ranks_per_node % tp_size == 0``.  When it doesn't, some
    # (server, stage) pairs get zero writers (missing stores) and
    # others get multiple (double-stores).  Reject this config early
    # rather than silently corrupting the cache.
    use_mla = getattr(
        getattr(vllm_config, "model_config", None), "use_mla", False
    )
    if use_mla and n_servers > 1:
        ranks_per_node = pc.world_size // n_servers
        tp_size = pc.tensor_parallel_size
        if tp_size > 0 and ranks_per_node % tp_size != 0:
            raise ValueError(
                "LMCacheMPConnector multi-server MLA mode requires "
                f"ranks_per_node ({ranks_per_node}) to be a multiple of "
                f"tp_size ({tp_size}); got world_size={pc.world_size}, "
                f"n_servers={n_servers}.  This layout would cause "
                "misaligned store-writer selection."
            )
