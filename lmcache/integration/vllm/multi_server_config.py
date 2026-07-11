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
        ValueError: If multi-server + DP is requested.
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
