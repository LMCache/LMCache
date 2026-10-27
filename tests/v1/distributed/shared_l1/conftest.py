# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures: a real memory orchestrator and L1s mapping one region file."""

# Standard

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import current_device_spec
import lmcache.v1.memory_management as memory_management


@pytest.fixture(autouse=True)
def cpu_mapping_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace GPU registration and pinned allocation, keeping real CPU views."""
    monkeypatch.setattr(current_device_spec, "pin_memory", lambda *args: True)
    monkeypatch.setattr(current_device_spec, "unpin_memory", lambda *args: True)
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *_, **__: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *_, **__: None)
