# SPDX-License-Identifier: Apache-2.0
"""Function-scoped CPU stand-ins for torch-rbln's HND prerequisites."""

# Standard
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import NamedTuple

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.rbln import staging


class DevicePlacement(NamedTuple):
    """Mirror of the torch-rbln placement record."""

    node_id: int
    chiplet_id: int
    slice_axes: tuple[tuple[int, int, int], ...]


@dataclass
class CpuRbln:
    """Placement log and allocation failure injection for CPU tensor tests."""

    chiplets: int = 4
    fail_chiplet: int | None = None
    byte_limit: int | None = None
    placed_bytes: int = 0
    placements: list[tuple[tuple[int, ...], list[DevicePlacement]]] = field(
        default_factory=list
    )

    def set_device_alloc(
        self, tensor: torch.Tensor, placements: list[DevicePlacement]
    ) -> None:
        """Record placement or fail before the caller can copy any bytes."""
        if any(p.chiplet_id == self.fail_chiplet for p in placements):
            raise RuntimeError("injected placement failure")
        self.placed_bytes += tensor.numel() * tensor.element_size()
        if self.byte_limit is not None and self.placed_bytes > self.byte_limit:
            raise RuntimeError("injected placement capacity exhausted")
        self.placements.append((tuple(tensor.shape), placements))

    def get_device_properties(self, device: torch.device) -> SimpleNamespace:
        """Report a stable chiplet count."""
        return SimpleNamespace(num_chiplet=self.chiplets)


@pytest.fixture
def cpu_rbln(monkeypatch: pytest.MonkeyPatch) -> Iterator[CpuRbln]:
    """Provide placement and a PrivateUse1 copy stand-in for one CPU test.

    The actual CPU swap uses tensor.copy_; this registration only satisfies
    the same prerequisite lookup as hardware. Libraries live for this fixture
    only, so hardware tests retain the real implementation. Reset the shared
    staging cache so allocation tests never inherit another test's buffers.
    """
    schema = implementation = None
    try:
        registered = torch._C._dispatch_has_kernel_for_dispatch_key(
            "torch_rbln::copy_strided_view", "PrivateUse1"
        )
    except RuntimeError:
        schema = torch.library.Library("torch_rbln", "FRAGMENT")
        schema.define("copy_strided_view(Tensor src, Tensor(a!) out) -> bool")
        registered = False
    if not registered:
        implementation = torch.library.Library("torch_rbln", "IMPL", "PrivateUse1")
        implementation.impl("copy_strided_view", lambda src, out: True)
    state = CpuRbln()
    monkeypatch.setattr(staging, "_STAGING", None)
    monkeypatch.setattr(staging, "_PLACEMENT_WARNING_LOGGED", False)
    monkeypatch.setattr(staging, "chiplet_count", lambda device: state.chiplets)
    monkeypatch.setattr(
        torch,
        "rbln",
        SimpleNamespace(
            DevicePlacement=DevicePlacement,
            set_device_alloc=state.set_device_alloc,
            get_device_properties=state.get_device_properties,
        ),
        raising=False,
    )
    yield state
    del implementation, schema
