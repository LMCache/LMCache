# SPDX-License-Identifier: Apache-2.0
"""The save_only_first_rank broadcast uses the worker's current device.

With vLLM data parallelism every DP engine's TP world starts at rank 0, so the
first rank of DP rank 1 has ``worker_id == 0`` while its KV cache is on another
GPU. Tensors exchanged by ``_broadcast_or_receive_memory_objs`` must be on the
device the inference engine made current, not ``cuda:{worker_id}``.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
import lmcache.v1.cache_engine as cache_engine_module

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


def _device(monkeypatch: pytest.MonkeyPatch, current: int) -> None:
    device = MagicMock()
    device.current_device.return_value = current
    device.device_count.return_value = 8
    monkeypatch.setattr(cache_engine_module, "torch_dev", device)
    monkeypatch.setattr(cache_engine_module, "torch_device_type", "cuda")


def test_first_rank_copies_to_current_device(monkeypatch: pytest.MonkeyPatch) -> None:
    # First rank of DP rank 1: worker_id 0, KV cache on GPU 3.
    _device(monkeypatch, current=3)
    monkeypatch.setattr(cache_engine_module, "TensorMemoryObj", lambda **kw: kw)
    engine = _engine(is_first_rank=True, worker_id=0)
    raw = MagicMock()
    memory_obj = SimpleNamespace(
        metadata=create_test_memory_obj().metadata, raw_tensor=raw
    )

    engine._broadcast_or_receive_memory_objs([(None, memory_obj, 0, 4)], None)

    raw.to.assert_called_once_with("cuda:3")
    engine.broadcast_fn.assert_called_once_with(raw.to.return_value, 0)
    assert engine._leader_gpu_substitute_objs[0]["raw_data"] is raw.to.return_value


def test_other_rank_receives_on_current_device(monkeypatch: pytest.MonkeyPatch) -> None:
    # TP rank 1 of DP rank 1 with TP=4 on one node: worker_id 1, GPU 5.
    metadata = create_test_memory_obj().metadata.to_dict()
    _device(monkeypatch, current=5)
    devices = []
    empty = torch.empty

    def fake_empty(size, dtype=None, device=None):
        devices.append(device)
        return empty(size, dtype=dtype)

    monkeypatch.setattr(cache_engine_module.torch, "empty", fake_empty)
    engine = _engine(is_first_rank=False, worker_id=1)
    engine.broadcast_object_fn.side_effect = [1, (0, 4, metadata)]
    reordered_chunks: list = []
    ret_mask = torch.zeros(8, dtype=torch.bool)

    engine._broadcast_or_receive_memory_objs(reordered_chunks, ret_mask)

    assert devices == ["cuda:5"]
    assert len(reordered_chunks) == 1 and ret_mask[:4].all()
