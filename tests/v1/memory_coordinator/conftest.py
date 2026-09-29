# SPDX-License-Identifier: Apache-2.0
"""Common metadata and configuration for coordinator tests."""

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey
from lmcache.v1.memory_coordinator.api import (
    ReservationRef,
    WireLayout,
    WriteGrant,
    WriteReserveItem,
)
from lmcache.v1.memory_coordinator.config import MemoryCoordinatorConfig

TOKEN = "test-memory-coordinator-token"
CAPACITY = 64 * 1024


def key(seed: int) -> EncodedObjectKey:
    return EncodedObjectKey(f"{seed:08x}", "model", 0)


def item(seed: int, elements: int = 64) -> WriteReserveItem:
    return WriteReserveItem(
        key=key(seed), layout=WireLayout(shapes=((elements,),), dtypes=("float16",))
    )


def ref(grant: WriteGrant) -> ReservationRef:
    return ReservationRef(key=grant.key, token=grant.token)


def config(token_file: Path) -> MemoryCoordinatorConfig:
    return MemoryCoordinatorConfig(
        token_file=str(token_file),
        state_file=str(token_file.with_name("coordinator.state")),
        region_id="region",
        capacity_bytes=CAPACITY,
        alignment_bytes=4096,
        layout_id="layout",
    )


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    path = tmp_path / "token"
    path.write_text(TOKEN + "\n")
    return path


@pytest.fixture
def coordinator_config(token_file: Path) -> MemoryCoordinatorConfig:
    return config(token_file)
