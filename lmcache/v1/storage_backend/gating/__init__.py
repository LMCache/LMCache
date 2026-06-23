# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Mapping
from typing import Any

# First Party
from lmcache.v1.storage_backend.gating.base_gate import BaseStorageGate, NullStorageGate
from lmcache.v1.storage_backend.gating.ssd_gate import (
    SsdStorageGate,
    SsdStorageGateStats,
)
from lmcache.v1.storage_backend.gating.write_veto import WriteVetoReason


def build_storage_gate_from_extra(extra: Mapping[str, Any]) -> BaseStorageGate:
    """
    Build a storage gate from ``LMCacheEngineConfig.extra_config``-style mapping.

    If both ``ssd_gate_min_size_bytes`` and ``ssd_gate_min_access_count`` are
    zero (or missing), returns :class:`NullStorageGate`.
    """
    min_sz = int(extra.get("ssd_gate_min_size_bytes", 0))
    min_acc = int(extra.get("ssd_gate_min_access_count", 0))
    if min_sz == 0 and min_acc == 0:
        return NullStorageGate()
    return SsdStorageGate(
        min_size_bytes=min_sz,
        min_read_count_before_write=min_acc,
    )


__all__ = [
    "BaseStorageGate",
    "NullStorageGate",
    "SsdStorageGate",
    "SsdStorageGateStats",
    "WriteVetoReason",
    "build_storage_gate_from_extra",
]
