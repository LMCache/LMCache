# SPDX-License-Identifier: Apache-2.0
"""The MLA first-rank broadcast must use the current device, not worker_id."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1.cache_engine import LMCacheEngine
import lmcache.inprocess.v1.cache_engine as cache_engine_module

# Local
from .utils import create_test_memory_obj


def _engine(is_first_rank: bool, worker_id: int) -> LMCacheEngine:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.metadata = SimpleNamespace(
        is_first_rank=lambda: is_first_rank, first_rank=0, worker_id=worker_id
    )
    engine.broadcast_fn = MagicMock()
    engine.broadcast_object_fn = MagicMock()
    return engine


def _set_current_device(monkeypatch: pytest.MonkeyPatch, current: int) -> None:
    device = MagicMock()
    device.current_device.return_value = current
    monkeypatch.setattr(cache_engine_module, "torch_dev", device)
    monkeypatch.setattr(cache_engine_module, "torch_device_type", "cuda")


def test_first_rank_copies_to_current_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_current_device(monkeypatch, current=3)
    monkeypatch.setattr(cache_engine_module, "TensorMemoryObj", lambda **kw: kw)
    engine = _engine(is_first_rank=True, worker_id=0)
    raw = MagicMock()
    memory_obj = SimpleNamespace(
        metadata=create_test_memory_obj().metadata, raw_tensor=raw
    )

    engine._broadcast_or_receive_memory_objs([(None, memory_obj, 0, 4)], None)

    raw.to.assert_called_once_with("cuda:3")


def test_other_rank_receives_on_current_device(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = create_test_memory_obj().metadata.to_dict()
    _set_current_device(monkeypatch, current=5)
    devices = []
    real_empty = torch.empty

    def recording_empty(size, dtype=None, device=None):
        devices.append(device)
        return real_empty(size, dtype=dtype)

    monkeypatch.setattr(cache_engine_module.torch, "empty", recording_empty)
    engine = _engine(is_first_rank=False, worker_id=1)
    engine.broadcast_object_fn.side_effect = [1, (0, 4, metadata)]

    engine._broadcast_or_receive_memory_objs([], torch.zeros(8, dtype=torch.bool))

    assert devices == ["cuda:5"]
