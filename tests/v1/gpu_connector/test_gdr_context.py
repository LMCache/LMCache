# SPDX-License-Identifier: Apache-2.0
"""GDR landing ring: registration, slot layout, bank hand-off."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import gdr_context as mod
from lmcache.v1.gpu_connector.gdr_context import (
    DeviceRegionRegistrar,
    GdrLandingRing,
    create_landing_ring,
    get_device_region_registrar,
    set_device_region_registrar,
)

ALIGN = 4096


class _Registrar(DeviceRegionRegistrar):
    def __init__(self) -> None:
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []

    @property
    def align_bytes(self) -> int:
        return ALIGN

    def register(self, ptr: int, size: int, device_index: int) -> int:
        self.registered.append((ptr, size, device_index))
        return 11

    def unregister(self, region_id: int) -> None:
        self.unregistered.append(region_id)


@pytest.fixture
def registrar():
    reg = _Registrar()
    set_device_region_registrar(reg)
    try:
        yield reg
    finally:
        set_device_region_registrar(None)


def test_no_registrar_means_no_ring() -> None:
    assert get_device_region_registrar() is None
    assert create_landing_ring(torch.device("cpu"), 100, 4) is None


def test_ring_is_aligned_and_registered_once(registrar: _Registrar) -> None:
    ring = create_landing_ring(torch.device("cpu"), slot_bytes=100, slots_per_bank=3)
    assert ring is not None
    assert ring.slot_bytes == ALIGN
    assert ring.slots_per_bank == 3
    (ptr, size, device_index), *rest = registrar.registered
    assert not rest
    assert ptr % ALIGN == 0
    assert size == 2 * 3 * ALIGN
    assert device_index is None or device_index == torch.device("cpu").index

    slot = ring.slot(1, 2)
    assert slot.region_id == 11
    assert slot.offset == (3 + 2) * ALIGN
    assert slot.buffer.nbytes == ALIGN
    assert slot.buffer.data_ptr() == ptr + slot.offset
    with pytest.raises(IndexError):
        ring.slot(2, 0)
    with pytest.raises(IndexError):
        ring.slot(0, 3)

    ring.close()
    ring.close()
    assert registrar.unregistered == [11]


def test_acquire_waits_for_the_banks_last_kernel(
    registrar: _Registrar, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = MagicMock()
    event.is_complete.side_effect = [False, False, True]
    monkeypatch.setattr(mod, "record_completion_event", lambda device, stream: event)
    monkeypatch.setattr(mod, "_POLL_INTERVAL_S", 0)
    ring = GdrLandingRing(torch.device("cpu"), ALIGN, 1, registrar)
    ring.acquire(0)  # nothing recorded yet: returns at once
    ring.release(0, torch.device("cpu"), stream=None)
    ring.acquire(0)
    assert event.is_complete.call_count == 3
    ring.acquire(0)  # consumed: no further polling
    assert event.is_complete.call_count == 3
    ring.close()
