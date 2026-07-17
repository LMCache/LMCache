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
    if n_servers <= 0:
        raise ValueError(
            f"n_servers must be >= 1, got {n_servers}. "
            "Check lmcache.mp.server_urls configuration."
        )
    assert pc.world_size % n_servers == 0, (
        f"world_size ({pc.world_size}) must be "
        f"divisible by n_servers ({n_servers})"
    )

    dp_size = getattr(pc, "data_parallel_size", 1)
    if n_servers > 1 and dp_size > 1:
        # DP is now supported: each DP replica gets n_servers // dp_size
        # servers.  Validate the split is even.
        if n_servers % dp_size != 0:
            raise ValueError(
                "LMCacheMPConnector multi-server + DP mode requires "
                f"n_servers ({n_servers}) to be divisible by dp_size ({dp_size}). "
                f"Each DP replica gets n_servers // dp_size servers."
            )

    # DCP awareness: decode context parallel splits a TP group into
    # tp_size // dcp_size DCP groups that share the same KV.  The
    # effective TP shard size for MLA alignment is tp_size (DCP workers
    # within a TP rank share the same MLA KV object).
    dcp_size = getattr(pc, "decode_context_parallel_size", 1)

    # MLA multi-server alignment: the is_kv_writer formula
    # ``(rank % ranks_per_node) % tp_size == 0`` yields exactly one
    # writer per (server, pipeline-stage) only when server blocks are
    # TP-aligned.  When ranks_per_node >= tp_size, require
    # ranks_per_node % tp_size == 0 (strict alignment).  When
    # ranks_per_node < tp_size (more servers than TP groups), the
    # min(tp, rpn) clamping in kv_tp_size makes the lock-balance work
    # as long as each server's ranks all fall in the same PP stage.
    use_mla = getattr(
        getattr(vllm_config, "model_config", None), "use_mla", False
    )
    if use_mla and n_servers > 1:
        ranks_per_node = pc.world_size // n_servers
        tp_size = pc.tensor_parallel_size
        if tp_size > 0 and ranks_per_node > 0:
            if ranks_per_node >= tp_size:
                # Strict alignment: server blocks must be TP-aligned
                if ranks_per_node % tp_size != 0:
                    raise ValueError(
                        "LMCacheMPConnector multi-server MLA mode requires "
                        f"ranks_per_node ({ranks_per_node}) to be a multiple of "
                        f"tp_size ({tp_size}); got world_size={pc.world_size}, "
                        f"n_servers={n_servers}.  This layout would cause "
                        "misaligned store-writer selection."
                    )
            # else: ranks_per_node < tp_size — the min(tp, rpn) clamping
            # in kv_tp_size handles this.  Each server has rpn ranks,
            # all in the same PP stage (since rpn < tp means the server
            # block is smaller than one TP group, and vLLM's TP-inner
            # rank layout puts consecutive ranks in the same stage).
            # This is balanced: locked = min(tp, rpn) = rpn = readers.
            # No rejection needed.
