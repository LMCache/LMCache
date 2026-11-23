# SPDX-License-Identifier: Apache-2.0
"""A registration gets its own peer event backend, on the server and the engine.

Backends that keep the events imported from a peer (ROCm) hand out one view per
registration, so an unregistered or restarted peer's imports are dropped with
its registration.
"""

# Standard
from contextlib import nullcontext
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
from lmcache.v1.transport.futures import MessagingFuture
from tests.v1.multiprocess.test_event_ipc_handle_path import _FakeEventBackend
from tests.v1.multiprocess.test_rocm_host_event_wait import _build


class _PeerView:
    """One registration's backend: remembers what it imported."""

    def __init__(self, shared: "_SharedBackend") -> None:
        self._shared = shared
        self.imports: dict[bytes, object] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._shared, name)

    def import_event(self, handle: bytes, device: object) -> object:
        if handle not in self.imports:
            self.imports[handle] = self._shared.import_event(handle, device)
        return self.imports[handle]


class _SharedBackend(_FakeEventBackend):
    """A device backend that scopes its imports per peer."""

    def __init__(self) -> None:
        super().__init__()
        self.views: list[_PeerView] = []

    def for_peer(self) -> _PeerView:
        self.views.append(_PeerView(self))
        return self.views[-1]


def _imports(backend: _SharedBackend) -> list[bytes]:
    return [call[1] for call in backend.calls if call[0] == "import"]


def test_server_registration_owns_its_peer_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = _build(monkeypatch, enabled=False, hip=None, registry=True)
    module = t.module
    monkeypatch.setattr(lmcache_driven_transfer.torch_dev, "empty_cache", lambda: None)
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "ipc_collect", lambda: None, raising=False
    )
    shared = _SharedBackend()
    monkeypatch.setattr(
        module,
        "_create_cache_context",
        lambda *args, **kwargs: t.make_entry(shared).cache_context,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "get_event_ipc_backend", lambda device: shared
    )

    module.register_kv_cache(7, [], "model", 1, None, None, [])
    first = module._cache_contexts[7].event_backend
    assert first is shared.views[0]
    first.import_event(b"h", "d")
    first.import_event(b"h", "d")
    assert _imports(shared) == [b"h"]  # opened once for this registration

    # The engine goes away and a new one registers under the same instance
    # ID and issues the same handle bytes.
    module.unregister_kv_cache(7)
    module.register_kv_cache(7, [], "model", 1, None, None, [])
    second = module._cache_contexts[7].event_backend
    assert second is shared.views[1] and second is not first
    second.import_event(b"h", "d")
    assert _imports(shared) == [b"h", b"h"]  # not served from the old one

    # Another instance on the same device never shares imports with it.
    module.register_kv_cache(8, [], "model", 1, None, None, [])
    assert module._cache_contexts[8].event_backend is shared.views[2]


def test_worker_registration_owns_its_peer_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    from lmcache.v1.client import worker_transfer

    shared = _SharedBackend()
    monkeypatch.setattr(worker_transfer, "get_event_ipc_backend", lambda d: shared)
    monkeypatch.setattr(
        worker_transfer, "wrap_kv_caches", lambda kv_caches: list(kv_caches.values())
    )
    monkeypatch.setattr(
        worker_transfer,
        "kv_wrapper_fds_delivered",
        lambda wrappers, fd_channel_path, timeout: nullcontext(),
    )
    done: MessagingFuture[object] = MessagingFuture()
    done.set_result(True)
    client = MagicMock()
    client.register_kv_cache.return_value = done

    context = worker_transfer.LMCacheDrivenTransferContext(1, client)
    kv_caches = {"layer_0": torch.empty(1)}
    context.register(kv_caches, "model", 1, 1, 1.0)
    assert context._event_backend is shared.views[0]

    # Heartbeat recovery registers again after a server restart.
    context.register(kv_caches, "model", 1, 1, 1.0)
    assert context._event_backend is shared.views[1]
    assert len(shared.views) == 2
