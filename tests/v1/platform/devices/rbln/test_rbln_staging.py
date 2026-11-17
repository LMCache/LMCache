# SPDX-License-Identifier: Apache-2.0
"""Chiplet properties used to distribute HND staging."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.rbln.staging import chiplet_count


def test_chiplets_come_from_device_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reported property is cached separately for each device."""
    seen: list[str] = []

    def properties(device: torch.device) -> SimpleNamespace:
        seen.append(str(device))
        return SimpleNamespace(num_chiplet=4)

    monkeypatch.setattr(
        torch, "rbln", SimpleNamespace(get_device_properties=properties), raising=False
    )
    for _ in range(2):
        assert chiplet_count(torch.device("meta")) == 4
        assert chiplet_count(torch.device("cuda", 5)) == 4
    assert seen == ["meta", "cuda:5"]


def test_cpu_chiplet_count_needs_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU reference tensors use one shard without torch-rbln."""
    monkeypatch.delattr(torch, "rbln", raising=False)
    assert chiplet_count(torch.device("cpu")) == 1
