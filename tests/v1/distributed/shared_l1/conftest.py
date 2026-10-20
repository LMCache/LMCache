# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures for shared ownership and storage integration."""

# Standard
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.shared_l1.manager import SharedL1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_orchestrator.api import DEFAULT_LAYOUT_FINGERPRINT, RegionUsage
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from lmcache.v1.platform import current_device_spec
from tests.v1.distributed.shared_l1.utils import CAPACITY, shared_config
from tests.v1.memory_orchestrator.harness import (
    REGION_ID,
    Orchestrator,
    running_orchestrator,
)
import lmcache.v1.memory_management as memory_management


@pytest.fixture(autouse=True)
def cpu_mapping_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace hardware allocation/registration, retaining real CPU views."""
    monkeypatch.setattr(current_device_spec, "pin_memory", lambda *args: True)
    monkeypatch.setattr(current_device_spec, "unpin_memory", lambda *args: True)
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        memory_management, "_free_cpu_memory", lambda *args, **kwargs: None
    )


@pytest.fixture(params=[CAPACITY])
def orchestrator(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[Orchestrator]:
    with running_orchestrator(tmp_path / "state", capacity_bytes=request.param) as orch:
        yield orch


@pytest.fixture
def make_manager(
    orchestrator: Orchestrator, tmp_path: Path
) -> Iterator[Callable[[str], SharedL1Manager]]:
    with ExitStack() as stack:

        def create(client_id: str) -> SharedL1Manager:
            manager = SharedL1Manager(
                shared_config(
                    tmp_path / "region",
                    orchestrator.endpoint,
                    client_id,
                    size_bytes=CAPACITY,
                )
            )
            stack.callback(manager.close)
            return manager

        yield create


@pytest.fixture
def make_store(
    orchestrator: Orchestrator, tmp_path: Path
) -> Iterator[Callable[[str], StorageManager]]:
    with ExitStack() as stack:

        def create(client_id: str) -> StorageManager:
            shared = shared_config(
                tmp_path / "region",
                orchestrator.endpoint,
                client_id,
                size_bytes=CAPACITY,
            )
            private = L1ManagerConfig(
                L1MemoryManagerConfig(8192, use_lazy=False, shm_name=""), tag="private"
            )
            store = StorageManager(
                StorageManagerConfig(
                    shared, EvictionConfig("noop"), l1_manager_configs=[shared, private]
                )
            )
            stack.callback(store.close)
            return store

        yield create


@pytest.fixture
def usage(orchestrator: Orchestrator) -> Iterator[Callable[[], RegionUsage]]:
    """Read uncached region counters, independent of L1's status cache."""
    client = MemoryOrchestratorClient(orchestrator.endpoint, REGION_ID, "probe")
    try:
        contract = client.describe_region()
        client.register(
            layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
            mapped_bytes=contract.capacity_bytes,
            visibility_mode=contract.visibility_mode,
        )
        yield client.usage
    finally:
        client.close()
