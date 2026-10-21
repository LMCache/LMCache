# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the multiprocess extension hooks.

Plugin packages (loaded with ``--server-module`` or a custom vLLM
connector) customize the LMCache-driven path by overriding these methods
rather than copying the code around them:

* ``BaseCacheContext.transfer_object_group`` -- every object-group copy in
  ``store`` and ``retrieve``, owned by the context whose staging layout it
  writes.
* ``LMCacheDrivenTransferModule._create_cache_context`` -- the context built by
  ``register_kv_cache``.
* ``GPUCacheContext._make_temp_buffer`` -- the staging buffer (CUDA-only
  runtime test lives in ``test_gpu_cache_context.py``).
* ``LMCacheMPWorkerAdapter._create_transfer_context`` -- the worker-side
  transfer context.
* ``LMCacheMPConnector.create_lmcache_mp_worker_adapter`` -- the worker
  adapter.

Subclasses of ``LMCacheDrivenTransferContext`` also rely on the state its
constructor sets, so that is pinned too.

Two things are pinned: every call site really goes through its hook (an
override that is silently bypassed is the failure that tests elsewhere would
not notice), and the hook signatures stay stable. If a test here fails,
change the hook deliberately and treat it as an extension-API change.
"""

# Standard
from types import SimpleNamespace
from typing import Any, Callable, cast
from unittest.mock import MagicMock
import inspect
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
)
from lmcache.v1.multiprocess import object_group_transfer
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    LMCacheDrivenTransferContext,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from tests.v1.multiprocess.test_lmcache_driven_transfer_skip import (
    _make_checkpoint_module,
    _make_module,
)
import lmcache.lmcache_native as lmcache_native

_H2D = lmcache_native.TransferDirection.H2D
_D2H = lmcache_native.TransferDirection.D2H


class _StopHere(Exception):
    """Raised by a test override to end the call right after the hook."""


def _shape(fn: Callable[..., Any]) -> str:
    """Parameter names in order, with ``*`` before the keyword-only ones."""
    parts: list[str] = []
    for param in inspect.signature(fn).parameters.values():
        if param.kind is param.KEYWORD_ONLY and "*" not in parts:
            parts.append("*")
        parts.append(param.name)
    return ", ".join(parts)


# ------------------------------------------------------------------ #
#  BaseCacheContext.transfer_object_group                              #
# ------------------------------------------------------------------ #


def _copies(cache_context: MagicMock) -> list[tuple[int, Any]]:
    return [
        (call.kwargs["object_group_id"], call.kwargs["direction"])
        for call in cache_context.transfer_object_group.call_args_list
    ]


def test_retrieve_copies_only_through_the_cache_context(monkeypatch):
    module, _reads, _copied = _make_module(monkeypatch, 3, num_chunks_in_sw=[-1, 1])
    cache_context = module.get_and_touch_context_entry(1).cache_context

    _handle, ok = module.retrieve(
        key=SimpleNamespace(request_id="req", cache_salt="salt"),
        instance_id=1,
        gpu_block_ids=[[1, 2, 3], [0, 0, 9]],
        event_ipc_handle=b"x",
    )

    assert ok is True
    assert _copies(cache_context) == [(0, _H2D), (1, _H2D)]


def test_store_copies_only_through_the_cache_context(monkeypatch):
    module, cache_context, _reads, _copied = _make_checkpoint_module(monkeypatch)

    _handle, ok = module.store(
        SimpleNamespace(request_id="req", worker_id=1),
        1,
        [[0, 1, 2, 3], [-1, -1, 0, 1], [-1, -1, 2, 3]],
        b"producer",
    )

    assert ok
    assert _copies(cache_context) == [(0, _D2H), (1, _D2H)]


def test_transfer_module_has_no_direct_copy_path():
    # A direct import would let a new copy site skip the context's override.
    assert not hasattr(mod, "transfer_kv_per_object_group")


def test_default_context_copy_forwards_every_argument(monkeypatch):
    copy = MagicMock()
    monkeypatch.setattr(object_group_transfer, "transfer_kv_per_object_group", copy)
    cache_context = cast(BaseCacheContext, object())
    block_ids, memory_objs = [object()], [None]

    BaseCacheContext.transfer_object_group(
        cache_context,
        cast(Any, block_ids),
        memory_objs,
        object_group_id=2,
        batch_size=4,
        skip_first_n_tokens=16,
        direction=_H2D,
        transfer_key="key",
        block_ids_host=[[1]],
    )

    copy.assert_called_once_with(
        cache_context,
        block_ids,
        memory_objs,
        object_group_id=2,
        batch_size=4,
        skip_first_n_tokens=16,
        direction=_H2D,
        transfer_key="key",
        block_ids_host=[[1]],
    )


# ------------------------------------------------------------------ #
#  LMCacheDrivenTransferModule._create_cache_context                   #
# ------------------------------------------------------------------ #


def test_register_builds_the_context_through_the_hook():
    seen: dict[str, Any] = {}

    class _Module(LMCacheDrivenTransferModule):
        def _create_cache_context(
            self,
            kv_caches,
            *,
            instance_id,
            engine_type,
            layout_hints,
            engine_group_infos,
        ):
            seen.update(
                kv_caches=kv_caches,
                instance_id=instance_id,
                engine_type=engine_type,
                layout_hints=layout_hints,
                engine_group_infos=engine_group_infos,
            )
            raise _StopHere

    module = _Module.__new__(_Module)
    module._lock = threading.Lock()
    module._cache_contexts = {}
    kv_caches, infos = object(), [object()]
    engine_type = next(iter(mod.EngineType))

    with pytest.raises(_StopHere):
        module.register_kv_cache(7, kv_caches, "m", 1, engine_type, {}, infos)

    # Empty hints are normalized to None before the hook, as before.
    assert seen == {
        "kv_caches": kv_caches,
        "instance_id": 7,
        "engine_type": engine_type,
        "layout_hints": None,
        "engine_group_infos": infos,
    }


def test_default_cache_context_hook_applies_server_settings(monkeypatch):
    create = MagicMock(return_value="context")
    monkeypatch.setattr(mod, "create_cache_context", create)
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = SimpleNamespace(
        chunk_size=128, separate_object_groups=True, full_sw_kv=True
    )
    kv_caches, infos = object(), [object()]
    engine_type = next(iter(mod.EngineType))

    result = module._create_cache_context(
        kv_caches,
        instance_id=7,
        engine_type=engine_type,
        layout_hints=None,
        engine_group_infos=infos,
    )

    assert result == "context"
    create.assert_called_once_with(
        kv_caches,
        128,
        layout_hints=None,
        engine_group_infos=infos,
        engine_type=engine_type,
        separate_object_groups=True,
        full_sw_kv=True,
    )


# ------------------------------------------------------------------ #
#  LMCacheMPWorkerAdapter._create_transfer_context                     #
# ------------------------------------------------------------------ #


def test_registration_builds_the_transfer_context_through_the_hook():
    seen: list[Any] = []

    class _Adapter(LMCacheMPWorkerAdapter):
        def _create_transfer_context(self, kv_caches):
            seen.append(kv_caches)
            raise _StopHere

    adapter = _Adapter.__new__(_Adapter)
    kv_caches = {"layer.0": object()}

    # Shared by register_kv_caches and heartbeat re-registration.
    with pytest.raises(_StopHere):
        adapter._send_register_kv_caches_request(kv_caches)

    assert seen == [kv_caches]


def test_default_transfer_context_hook_uses_adapter_settings(monkeypatch):
    create = MagicMock(return_value="transfer-context")
    monkeypatch.setattr(adapter_mod, "create_transfer_context", create)
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter.instance_id = 42
    adapter.req_client = object()
    adapter._mp_transfer_mode = "lmcache_driven"
    kv_caches = {"layer.0": object()}

    assert adapter._create_transfer_context(kv_caches) == "transfer-context"
    create.assert_called_once_with(
        kv_caches,
        instance_id=42,
        req_client=adapter.req_client,
        mode="lmcache_driven",
    )


def test_transfer_context_state_for_subclasses_is_stable():
    req_client = object()

    context = LMCacheDrivenTransferContext(5, cast(Any, req_client))

    # Read by subclasses that add requests (for example, an extra
    # registration round trip); renaming any of these breaks them silently.
    assert context._instance_id == 5
    assert context._req_client is req_client
    assert context._device is None
    assert context._event_backend is None


# ------------------------------------------------------------------ #
#  LMCacheMPConnector.create_lmcache_mp_worker_adapter                 #
# ------------------------------------------------------------------ #


def test_connector_builds_the_stock_worker_adapter(monkeypatch):
    pytest.importorskip("vllm")
    # First Party
    from lmcache.integration.vllm import lmcache_mp_connector as connector_mod

    # Resolved at call time, so tests patching the module-level name still
    # take effect.
    built = MagicMock()
    monkeypatch.setattr(connector_mod, "LMCacheMPWorkerAdapter", built)
    connector = connector_mod.LMCacheMPConnector.__new__(
        connector_mod.LMCacheMPConnector
    )
    adapter = connector.create_lmcache_mp_worker_adapter(server_url="u", x=1)
    built.assert_called_once_with(server_url="u", x=1)
    assert adapter is built.return_value


def test_connector_worker_role_routes_through_the_adapter_hook():
    pytest.importorskip("vllm")
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    source = inspect.getsource(LMCacheMPConnector.__init__)
    assert "self.create_lmcache_mp_worker_adapter(" in source
    assert "LMCacheMPWorkerAdapter(" not in source


# ------------------------------------------------------------------ #
#  Signatures                                                          #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("hook", "expected"),
    [
        (
            BaseCacheContext.transfer_object_group,
            "self, block_ids_gpu, memory_objs, *, object_group_id, batch_size, "
            "skip_first_n_tokens, direction, transfer_key, block_ids_host",
        ),
        (
            LMCacheDrivenTransferModule._create_cache_context,
            "self, kv_caches, *, instance_id, engine_type, layout_hints, "
            "engine_group_infos",
        ),
        (
            LMCacheMPWorkerAdapter._create_transfer_context,
            "self, kv_caches",
        ),
    ],
    ids=lambda value: getattr(value, "__qualname__", ""),
)
def test_hook_signature_is_stable(hook, expected):
    assert _shape(hook) == expected


def test_temp_buffer_hook_signature_is_stable():
    pytest.importorskip("cupy", reason="the CUDA cache context requires cupy")
    # First Party
    from lmcache.v1.platform.devices.cuda.cache_context import GPUCacheContext

    assert _shape(GPUCacheContext._make_temp_buffer) == (
        "self, kv_layer_groups_manager, lmcache_tokens_per_chunk, device, "
        "max_batch_size, full_sw_kv"
    )
