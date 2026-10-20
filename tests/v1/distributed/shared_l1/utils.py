# SPDX-License-Identifier: Apache-2.0
"""Shared L1 configurations for tests; the orchestrator harness is in
``tests.v1.memory_orchestrator.harness``."""

# Standard
from pathlib import Path
import json

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, parse_args
from tests.v1.memory_orchestrator.harness import REGION_ID

CAPACITY = 128 * 4096


def key(index: int) -> ObjectKey:
    """Return a deterministic key for lifecycle and overflow tests."""
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "shared-test", 0)


def shared_config(
    region_path: Path,
    endpoint: str,
    client_id: str,
    *,
    size_bytes: int,
    tag: str = "pool",
    rpc_timeout_seconds: float = 5.0,
    region_id: str = REGION_ID,
    l1_type: str = "DRAM",
) -> L1ManagerConfig:
    """Parse a shared L1 config the way ``--l1-manager`` does.

    The default maps ``region_path`` as a file (the DRAM medium); pass
    ``l1_type="DEVDAX"`` for a Device-DAX device.
    """
    return parse_l1_manager_spec(
        {
            "type": l1_type,
            "tag": tag,
            "size_gb": size_bytes / (1 << 30),
            "path": str(region_path),
            "shared": {
                "orchestrator": endpoint,
                "region_id": region_id,
                "client_id": client_id,
                "rpc_timeout_seconds": rpc_timeout_seconds,
            },
        }
    )


def parse_l1_manager_spec(spec: dict[str, object]) -> L1ManagerConfig:
    """Parse one ``--l1-manager`` JSON object through the CLI parser."""
    config = parse_args(["--eviction-policy", "noop", "--l1-manager", json.dumps(spec)])
    return config.l1_manager_configs[0]
