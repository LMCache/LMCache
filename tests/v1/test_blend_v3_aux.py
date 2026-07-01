# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the CB-v3 AUX wire protocol and its two handlers.

Covers (CPU-only):
  * the ``AUX_PUT`` / ``AUX_GET_BY_HASH_IPC`` protocol definitions and their
    framework consistency (payload/response classes, blocking type,
    enum/name/definition cross-check);
  * ``BlendV3Module.store_aux`` — the blob-view + delegation contract;
  * ``BlendV3Module.retrieve_aux_by_hashes_ipc`` — the unregistered-instance
    guard, and the success-path orchestration with the CUDA-IPC surface mocked.

The handlers are exercised on a bare instance (``object.__new__``) with only the
two attributes they touch set, so no ``MPCacheServerContext`` / daemon thread /
GPU is constructed.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import contextlib

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    DeviceIPCWrapper,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod
from lmcache.v1.multiprocess.protocols import blend_v3 as proto
from lmcache.v1.multiprocess.protocols import initialize_protocols
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType

GROUP = 7000


def _key() -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        token_ids=[1, 2, 3, 4],
        start=0,
        end=4,
        request_id="req-0",
    )


# --------------------------------------------------------------------------- #
# Protocol definitions
# --------------------------------------------------------------------------- #
def test_aux_request_types_and_framework_consistency():
    # initialize_protocols() cross-checks enum <-> REQUEST_NAMES <-> definitions
    # and raises ProtocolInitializationError on any mismatch.
    defs = initialize_protocols()
    assert RequestType.AUX_PUT in defs
    assert RequestType.AUX_GET_BY_HASH_IPC in defs


def test_aux_names_registered_in_blend_v3_module():
    names = proto.REQUEST_NAMES
    module_defs = proto.get_protocol_definitions()
    for name in ("AUX_PUT", "AUX_GET_BY_HASH_IPC"):
        assert name in names
        assert name in module_defs


def test_aux_put_protocol_definition():
    d = proto.get_protocol_definitions()["AUX_PUT"]
    assert d.payload_classes == [IPCCacheServerKey, int, list[int], DeviceIPCWrapper]
    assert d.response_class is bool
    assert d.handler_type is HandlerType.BLOCKING


def test_aux_get_by_hash_ipc_protocol_definition():
    d = proto.get_protocol_definitions()["AUX_GET_BY_HASH_IPC"]
    assert d.payload_classes == [
        IPCCacheServerKey,
        int,
        list[bytes],
        list[int],
        DeviceIPCWrapper,
        int,
        bytes,
    ]
    assert d.response_class == tuple[bytes, bool]
    assert d.handler_type is HandlerType.BLOCKING


# --------------------------------------------------------------------------- #
# Handler harness
# --------------------------------------------------------------------------- #
def _bare_module():
    """A BlendV3Module instance with only the attrs the AUX handlers touch."""
    eng = object.__new__(v3_mod.BlendV3Module)
    eng._aux_store = MagicMock()
    eng._transfer_module = MagicMock()
    return eng


# --------------------------------------------------------------------------- #
# store_aux
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("store_ret", [True, False])
def test_store_aux_delegates_and_returns_bool(store_ret):
    eng = _bare_module()
    eng._aux_store.store.return_value = store_ret
    blob_ipc = SimpleNamespace(to_tensor=lambda: torch.zeros(16, dtype=torch.uint8))

    out = v3_mod.BlendV3Module.store_aux(eng, _key(), GROUP, [8, 8], blob_ipc)

    assert out is store_ret  # bool returned verbatim
    eng._aux_store.store.assert_called_once()
    args = eng._aux_store.store.call_args.args
    assert args[1] == GROUP and args[2] == [8, 8]
    passed_blob = args[3]
    assert passed_blob.dtype == torch.uint8 and passed_blob.dim() == 1


def test_store_aux_views_blob_as_uint8_widening_byte_count():
    eng = _bare_module()
    eng._aux_store.store.return_value = True
    # 8 bf16 elems == 16 bytes; the handler must hand the store a uint8 view.
    blob_ipc = SimpleNamespace(to_tensor=lambda: torch.zeros(8, dtype=torch.bfloat16))

    v3_mod.BlendV3Module.store_aux(eng, _key(), GROUP, [16], blob_ipc)

    passed_blob = eng._aux_store.store.call_args.args[3]
    assert passed_blob.dtype == torch.uint8
    assert passed_blob.numel() == 16


# --------------------------------------------------------------------------- #
# retrieve_aux_by_hashes_ipc
# --------------------------------------------------------------------------- #
def test_retrieve_unregistered_instance_raises_and_skips_fetch():
    eng = _bare_module()
    eng._transfer_module.get_and_touch_context_entry.return_value = None

    with pytest.raises(ValueError, match="not registered"):
        v3_mod.BlendV3Module.retrieve_aux_by_hashes_ipc(
            eng, _key(), GROUP, [], [], None, 5, b"evt"
        )

    eng._transfer_module.get_and_touch_context_entry.assert_called_once_with(5)
    eng._aux_store.fetch_into_ipc.assert_not_called()


def test_retrieve_success_path_returns_completion_event_and_ok(monkeypatch):
    """Orchestration test: resolves obj_keys, maps dst, delegates to
    fetch_into_ipc, and returns (server completion-event handle, ok). The
    CUDA-IPC surface is mocked so it runs on CPU."""
    eng = _bare_module()
    entry = SimpleNamespace(
        cache_context=SimpleNamespace(device="cuda:0", stream=object())
    )
    eng._transfer_module.get_and_touch_context_entry.return_value = entry
    eng._aux_store.fetch_into_ipc.return_value = True

    class _FakeEvent:
        def __init__(self, *a, **k):
            pass

        def record(self):
            pass

        def wait(self, stream=None):
            pass

        def ipc_handle(self):
            return b"server-evt"

        @classmethod
        def from_ipc_handle(cls, device, handle):
            return cls()

    fake_torch_dev = SimpleNamespace(
        Event=_FakeEvent,
        device=lambda d: contextlib.nullcontext(),
        stream=lambda s: contextlib.nullcontext(),
    )
    monkeypatch.setattr(v3_mod, "torch_dev", fake_torch_dev)
    monkeypatch.setattr(v3_mod, "check_interprocess_event_support", lambda: None)
    monkeypatch.setattr(
        v3_mod,
        "ipc_key_to_object_keys",
        lambda key, hashes, groups: [["k0", "k1"]],
    )
    dst_ipc = SimpleNamespace(to_tensor=lambda: torch.zeros(16, dtype=torch.uint8))

    handle, ok = v3_mod.BlendV3Module.retrieve_aux_by_hashes_ipc(
        eng, _key(), GROUP, [b"h0", b"h1"], [8, 8], dst_ipc, 5, b"worker-evt"
    )

    assert (handle, ok) == (b"server-evt", True)
    eng._aux_store.fetch_into_ipc.assert_called_once()
    fargs = eng._aux_store.fetch_into_ipc.call_args.args
    assert fargs[0] == ["k0", "k1"] and fargs[1] == [8, 8]
    assert fargs[2].dtype == torch.uint8  # the mapped dst buffer
