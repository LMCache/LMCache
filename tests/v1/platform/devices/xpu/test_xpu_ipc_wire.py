# SPDX-License-Identifier: Apache-2.0
"""XPU IPC wrappers must cross the MP wire as typed descriptors.

The descriptor tests need no accelerator. The end-to-end test uses a real
XPU tensor with stand-ins for torch's storage IPC methods, so it checks the
wrapper and wire plumbing rather than the driver import.
"""

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.platform.base.ipc_wire import TensorMetadata, XpuIPCData
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.devices.xpu.ipc_wrapper import (
    ENV_XPU_HANDLE_TRANSFER,
    XpuIPCWrapper,
)

pytestmark = pytest.mark.no_shared_allocator

METADATA = TensorMetadata("float32", (2, 3), (3, 1), 0, "device-uuid")
HANDLE = (0, b"h" * 64, 24, 0, "/refcounter", 0, b"e" * 64, False)


def _descriptor(**overrides: object) -> dict:
    # A plain msgpack round trip keeps bytes, unlike msgspec.to_builtins.
    descriptor = msgspec.msgpack.decode(
        msgspec.msgpack.encode(XpuIPCData(version=1, metadata=METADATA, handle=HANDLE))
    )
    descriptor.update(overrides)
    return descriptor


def test_descriptor_round_trips_through_zmq_codec() -> None:
    wrapper = DeviceIPCWrapper.Deserialize(msgspec.msgpack.encode(_descriptor()))
    assert type(wrapper) is XpuIPCWrapper
    encoded = get_customized_encoder(list[DeviceIPCWrapper]).encode([wrapper])
    restored = get_customized_decoder(list[DeviceIPCWrapper]).decode(encoded)[0]
    assert type(restored) is XpuIPCWrapper
    assert restored.to_wire() == XpuIPCData(version=1, metadata=METADATA, handle=HANDLE)


@pytest.mark.parametrize(
    "handle",
    [
        # A bytes ref-counter handle is the CUDA shape, not the XPU one.
        (0, b"h" * 64, 24, 0, b"/refcounter", 0, b"e" * 64, False),
        # The view needs 24 bytes; the storage only has 8.
        (0, b"h" * 64, 8, 0, "/refcounter", 0, b"e" * 64, False),
        (0, b"h" * 64, -1, 0, "/refcounter", 0, b"e" * 64, False),
        HANDLE[:7],
    ],
)
def test_invalid_handle_rejected(handle: tuple) -> None:
    with pytest.raises(msgspec.DecodeError):
        DeviceIPCWrapper.Deserialize(
            msgspec.msgpack.encode(_descriptor(handle=list(handle)))
        )


def test_from_wire_rejects_other_descriptors() -> None:
    with pytest.raises(ValueError, match="XPU"):
        XpuIPCWrapper.from_wire(object())  # type: ignore[arg-type]


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="requires an XPU device",
)
def test_serialized_wrapper_reopens_same_allocation(monkeypatch) -> None:
    """Serialize, decode and reopen a real XPU view through the wrapper."""
    shared: dict[bytes, torch.UntypedStorage] = {}

    def share_xpu(storage: torch.UntypedStorage) -> tuple:
        token = str(storage.data_ptr()).encode()
        shared[token] = storage
        return (0, token, storage.nbytes(), 0, "/refcounter", 0, None, False)

    def new_shared_xpu(device, handle, size, offset, *rest) -> torch.UntypedStorage:
        storage = shared[handle]
        assert (size, offset, rest) == (
            storage.nbytes(),
            0,
            ("/refcounter", 0, None, False),
        )
        return storage

    monkeypatch.setenv(ENV_XPU_HANDLE_TRANSFER, "1")
    monkeypatch.setattr(torch.UntypedStorage, "_share_xpu_", share_xpu, raising=False)
    monkeypatch.setattr(
        torch.UntypedStorage,
        "_new_shared_xpu",
        staticmethod(new_shared_xpu),
        raising=False,
    )

    base = torch.arange(32, dtype=torch.float16, device="xpu")
    view = base[4:].view(4, 7)
    wrapper = XpuIPCWrapper.wrap(view)

    restored = DeviceIPCWrapper.Deserialize(DeviceIPCWrapper.Serialize(wrapper))
    assert type(restored) is XpuIPCWrapper
    tensor = restored.to_tensor()
    assert tensor.data_ptr() == view.data_ptr()
    torch.testing.assert_close(tensor, view)
    tensor[0, 0] = -1
    assert base[4].item() == -1
