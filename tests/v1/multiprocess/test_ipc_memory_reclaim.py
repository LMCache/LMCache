# SPDX-License-Identifier: Apache-2.0
"""CUDA-IPC memory reclamation on instance release (LMCache#4014).

The MP server imports every client's KV cache via CUDA IPC. When a client
dies (reaper) or unregisters, dropping the tensor references and calling
``empty_cache()`` is NOT enough: IPC-imported segments live in the caching
allocator's IPC cache and are only unmapped by ``torch.cuda.ipc_collect()``
— and only once the last tensor reference is gone. Without it the server
retains the client's whole KV pool (observed: ~112 GB/GPU held after
``docker rm -f`` of the vLLM container, until a server restart).

These tests run the REAL registry/release flow with the device layer
stubbed: they assert every release path (explicit unregister, reaper,
server close) calls ``ipc_collect`` exactly once per batch, that the
entry references are actually dead by the time it fires (the load-bearing
ordering — a live reference turns ``ipc_collect`` into a silent no-op for
that entry's segments), and that device modules without ``ipc_collect``
(xpu / musa) degrade gracefully.
"""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

# Standard Library
import threading
import time
import weakref

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferModule,
)
import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as gpu_mod


class _FakeTorchDev:
    """Records the reclaim-call sequence; optionally omits ipc_collect."""

    empty_cache: MagicMock
    ipc_collect: MagicMock

    def __init__(self, with_ipc_collect: bool = True):
        self.calls: list[str] = []
        self.empty_cache = MagicMock(
            side_effect=lambda: self.calls.append("empty_cache")
        )
        if with_ipc_collect:
            self.ipc_collect = MagicMock(
                side_effect=lambda: self.calls.append("ipc_collect")
            )


def _bare_module() -> LMCacheDrivenTransferModule:
    """Module with only the registry state initialized (no GPU, no __init__)."""
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = MagicMock(name="ctx")
    module._cache_contexts = {}
    module._lock = threading.Lock()
    return module


def _entry(model: str = "m") -> ContextEntry:
    return ContextEntry(
        cache_context=MagicMock(name="cache_context"),
        model_name=model,
        world_size=1,
        last_seen=time.monotonic(),
        has_liveness_signal=True,
    )


def test_unregister_reclaims_ipc_memory(monkeypatch) -> None:
    """Explicit unregister closes the context AND runs empty_cache +
    ipc_collect (in that order)."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _bare_module()
    entry = _entry()
    ctx = cast(MagicMock, entry.cache_context)
    module._cache_contexts[7] = entry
    del entry

    module.unregister_kv_cache(7)

    ctx.close.assert_called_once()
    assert dev.calls == ["empty_cache", "ipc_collect"]
    assert module._cache_contexts == {}


def test_unregister_unknown_instance_does_not_reclaim(monkeypatch) -> None:
    """The warn path (already-reaped / never-registered id) must not touch
    the allocator — reclaim is tied to an actual release."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _bare_module()

    module.unregister_kv_cache(404)

    assert dev.calls == []


def test_unregister_entry_refs_dead_before_ipc_collect(monkeypatch) -> None:
    """THE load-bearing ordering: ipc_collect only frees segments whose
    tensors are unreferenced, so the entry must be garbage by the time it
    fires. Verified with a weakref probed from inside the fake collector."""
    module = _bare_module()
    entry = _entry()
    module._cache_contexts[1] = entry
    ref = weakref.ref(entry)
    del entry

    seen: dict = {}
    dev = SimpleNamespace(
        empty_cache=lambda: None,
        ipc_collect=lambda: seen.setdefault("entry_alive", ref() is not None),
    )
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)

    module.unregister_kv_cache(1)

    assert seen == {"entry_alive": False}


def test_reaper_reclaims_once_per_batch(monkeypatch) -> None:
    """Reaping N stale instances closes each context but batches the
    allocator reclaim into ONE empty_cache + ipc_collect."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _bare_module()
    stale_a, stale_b, fresh = _entry("a"), _entry("b"), _entry("c")
    stale_a.last_seen = stale_b.last_seen = time.monotonic() - 1000.0
    ctx_a, ctx_b, ctx_fresh = (
        cast(MagicMock, stale_a.cache_context),
        cast(MagicMock, stale_b.cache_context),
        cast(MagicMock, fresh.cache_context),
    )
    module._cache_contexts.update({1: stale_a, 2: stale_b, 3: fresh})
    del stale_a, stale_b, fresh

    reaped = module.reap_stale_instances(reap_timeout_s=60.0, registration_grace_s=60.0)

    assert sorted(reaped) == [1, 2]
    ctx_a.close.assert_called_once()
    ctx_b.close.assert_called_once()
    ctx_fresh.close.assert_not_called()
    assert dev.calls == ["empty_cache", "ipc_collect"]
    assert list(module._cache_contexts) == [3]


def test_reaper_noop_scan_does_not_reclaim(monkeypatch) -> None:
    """A scan that reaps nothing must not thrash the allocator."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _bare_module()
    module._cache_contexts[1] = _entry()

    reaped = module.reap_stale_instances(
        reap_timeout_s=3600.0, registration_grace_s=3600.0
    )

    assert reaped == []
    assert dev.calls == []


def test_reaper_entry_refs_dead_before_ipc_collect(monkeypatch) -> None:
    """Same ref-lifetime invariant on the reaper path."""
    module = _bare_module()
    entry = _entry()
    entry.last_seen = time.monotonic() - 1000.0
    module._cache_contexts[1] = entry
    ref = weakref.ref(entry)
    del entry

    seen: dict = {}
    dev = SimpleNamespace(
        empty_cache=lambda: None,
        ipc_collect=lambda: seen.setdefault("entry_alive", ref() is not None),
    )
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)

    module.reap_stale_instances(reap_timeout_s=60.0, registration_grace_s=60.0)

    assert seen == {"entry_alive": False}


def test_reclaim_degrades_without_ipc_collect(monkeypatch) -> None:
    """Device modules without ipc_collect (xpu / musa) must not raise —
    empty_cache still runs, the collect step is skipped."""
    dev = _FakeTorchDev(with_ipc_collect=False)
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _bare_module()
    module._cache_contexts[9] = _entry()

    module.unregister_kv_cache(9)

    assert dev.calls == ["empty_cache"]


def test_close_releases_all_and_reclaims_once(monkeypatch) -> None:
    """Server close() releases every remaining context and reclaims once."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _bare_module()
    module._device_host_func_dispatcher = MagicMock()
    cast(Any, module._ctx).storage_manager = MagicMock()
    e1, e2 = _entry("a"), _entry("b")
    c1, c2 = cast(MagicMock, e1.cache_context), cast(MagicMock, e2.cache_context)
    module._cache_contexts.update({1: e1, 2: e2})
    del e1, e2

    module.close()

    c1.close.assert_called_once()
    c2.close.assert_called_once()
    assert dev.calls == ["empty_cache", "ipc_collect"]
    assert module._cache_contexts == {}


def test_close_with_empty_registry_does_not_reclaim(monkeypatch) -> None:
    """close() on a server that never had clients skips the allocator."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _bare_module()
    module._device_host_func_dispatcher = MagicMock()
    cast(Any, module._ctx).storage_manager = MagicMock()

    module.close()

    assert dev.calls == []
