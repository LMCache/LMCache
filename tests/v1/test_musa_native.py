# SPDX-License-Identifier: Apache-2.0

# Standard
from types import ModuleType, SimpleNamespace
from typing import Any, cast
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import musa_native


def _make_native_module() -> ModuleType:
    """Create a fake native module with the Stage2 LMCache ABI."""
    module = ModuleType("musa_aiter")
    module.native_lmcache_kv_transfer_abi_version = lambda: 1  # type: ignore[attr-defined]
    module.lmcache_kv_paged_to_buffer = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    module.lmcache_kv_buffer_to_paged = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    module.lmcache_mla_paged_to_buffer = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    module.lmcache_mla_buffer_to_paged = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    return module


def test_native_transfer_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native MUSA transfer is opt-in so Stage1 behavior remains unchanged."""
    monkeypatch.delenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", raising=False)

    assert musa_native.is_native_musa_kv_transfer_enabled() is False


def test_native_transfer_enabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in environment variable enables native dispatch attempts."""
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    assert musa_native.is_native_musa_kv_transfer_enabled() is True


def test_load_native_module_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional ``musa_aiter`` must not break LMCache imports."""
    monkeypatch.setitem(cast(dict[str, Any], sys.modules), "musa_aiter", None)

    assert musa_native.load_native_musa_module() is None


def test_check_native_abi_requires_expected_symbols() -> None:
    """LMCache accepts only the Stage2 LMCache-compatible native ABI."""
    assert musa_native.check_native_abi(_make_native_module()) is True


def test_check_native_abi_rejects_missing_symbol() -> None:
    """The adapter falls back when the optional native module is incomplete."""
    module = SimpleNamespace(native_lmcache_kv_transfer_abi_version=lambda: 1)

    assert musa_native.check_native_abi(module) is False


def test_check_native_abi_rejects_non_callable_symbol() -> None:
    """Native symbols must be callable to be considered ABI-compatible."""
    module = _make_native_module()
    module.lmcache_kv_buffer_to_paged = object()  # type: ignore[attr-defined]

    assert musa_native.check_native_abi(module) is False


def test_try_native_to_gpu_calls_non_mla_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-MLA scatter uses the LMCache-compatible buffer-to-paged call."""
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    module = _make_native_module()

    def _buffer_to_paged(*args: Any, **kwargs: Any) -> bool:
        calls.append(("buffer_to_paged", args, kwargs))
        return True

    module.lmcache_kv_buffer_to_paged = _buffer_to_paged  # type: ignore[attr-defined]
    monkeypatch.setattr(musa_native, "load_native_musa_module", lambda: module)
    monkeypatch.setattr(musa_native, "_native_tensors_ready", lambda *args: True)
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    used = musa_native.try_native_to_gpu(
        use_mla=False,
        memory_tensor=torch.empty(2, 1, 4, 8),
        kvcaches=[torch.empty(2, 2, 2, 2, 4)],
        slot_mapping=torch.arange(4),
        start=0,
        end=4,
        skip_prefix_n_tokens=0,
        block_size=2,
        num_heads=2,
        head_size=4,
    )

    assert used is True
    assert calls[0][0] == "buffer_to_paged"
    assert torch.equal(calls[0][1][2], torch.arange(4))
    assert calls[0][1][3] == 0


def test_try_native_to_gpu_returns_true_for_empty_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty native transfer is a completed no-op after ABI is ready."""
    module = _make_native_module()
    monkeypatch.setattr(musa_native, "load_native_musa_module", lambda: module)
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    assert musa_native.try_native_to_gpu(
        use_mla=False,
        memory_tensor=torch.empty(2, 1, 4, 8),
        kvcaches=[torch.empty(2, 2, 2, 2, 4)],
        slot_mapping=torch.arange(4),
        start=0,
        end=4,
        skip_prefix_n_tokens=4,
        block_size=2,
        num_heads=2,
        head_size=4,
    )


def test_try_native_to_gpu_rejects_cpu_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native dispatch must not consume CPU tensors through musa_aiter fallback."""
    module = _make_native_module()
    monkeypatch.setattr(musa_native, "load_native_musa_module", lambda: module)
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    used = musa_native.try_native_to_gpu(
        use_mla=False,
        memory_tensor=torch.empty(2, 1, 4, 8),
        kvcaches=[torch.empty(2, 2, 2, 2, 4)],
        slot_mapping=torch.arange(4),
        start=0,
        end=4,
        skip_prefix_n_tokens=0,
        block_size=2,
        num_heads=2,
        head_size=4,
    )

    assert used is False


def test_try_native_from_gpu_falls_back_on_native_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native failures must return False so callers keep the torch fallback."""
    module = _make_native_module()

    def _paged_to_buffer(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("native boom")

    module.lmcache_kv_paged_to_buffer = _paged_to_buffer  # type: ignore[attr-defined]
    monkeypatch.setattr(musa_native, "load_native_musa_module", lambda: module)
    monkeypatch.setattr(musa_native, "_native_tensors_ready", lambda *args: True)
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    used = musa_native.try_native_from_gpu(
        use_mla=False,
        memory_tensor=torch.empty(2, 1, 4, 8),
        kvcaches=[torch.empty(2, 2, 2, 2, 4)],
        slot_mapping=torch.arange(4),
        start=0,
        end=4,
        block_size=2,
        num_heads=2,
        head_size=4,
    )

    assert used is False


def test_try_native_from_gpu_rejects_cpu_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native gather must not consume CPU tensors through musa_aiter fallback."""
    module = _make_native_module()
    monkeypatch.setattr(musa_native, "load_native_musa_module", lambda: module)
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    used = musa_native.try_native_from_gpu(
        use_mla=False,
        memory_tensor=torch.empty(2, 1, 4, 8),
        kvcaches=[torch.empty(2, 2, 2, 2, 4)],
        slot_mapping=torch.arange(4),
        start=0,
        end=4,
        block_size=2,
        num_heads=2,
        head_size=4,
    )

    assert used is False
