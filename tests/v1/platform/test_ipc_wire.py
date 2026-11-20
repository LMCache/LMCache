# SPDX-License-Identifier: Apache-2.0
"""IPC wire security and compatibility tests; no accelerator is required."""

# Standard
from pathlib import Path
from typing import Callable
import pickle

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import common_pb2
from lmcache.v1.multiprocess.transport.grpc_impl.codecs.common import get_message_codecs
from lmcache.v1.platform.base.ipc_wire import (
    CpuShmIPCData,
    CudaIPCData,
    MusaIPCData,
    NpuIPCData,
    NpuPlaneData,
    RawCudaIPCData,
    TensorMetadata,
    VmmCudaIPCData,
    WireIPCData,
)
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.devices.cpu.shm import CpuShmTensorWrapper

pytestmark = pytest.mark.no_shared_allocator

METADATA = TensorMetadata("float32", (2, 3), (3, 1), 0, "device-uuid")
DESCRIPTORS = [
    NpuIPCData(version=1, bare=True, planes=(NpuPlaneData((0, b"npu", 24), METADATA),)),
    NpuIPCData(
        version=1, bare=False, planes=(NpuPlaneData((0, b"npu", 24), METADATA),) * 2
    ),
    CudaIPCData(
        version=1,
        metadata=METADATA,
        handle=(0, b"h" * 64, 24, 0, b"refcounter", 0, b"e" * 64, False),
    ),
    RawCudaIPCData(
        version=1,
        metadata=METADATA,
        ipc_handle=b"h" * 64,
        alloc_offset=16,
        nbytes=24,
    ),
    VmmCudaIPCData(
        version=1,
        metadata=METADATA,
        export_id=b"i" * 16,
        kind="posix_fd",
        fabric_blob=None,
        alloc_size=4096,
        alloc_offset=16,
        nbytes=24,
    ),
    VmmCudaIPCData(
        version=1,
        metadata=METADATA,
        export_id=b"i" * 16,
        kind="fabric",
        fabric_blob=b"f" * 64,
        alloc_size=4096,
        alloc_offset=16,
        nbytes=24,
    ),
    CpuShmIPCData(version=1, metadata=METADATA, shm_name="/lmcache_test", nbytes=24),
    MusaIPCData(
        version=1, metadata=METADATA, handle={"token": b"handle", "dims": [2, 3]}
    ),
]


def _decode_zmq(payload: bytes) -> list[DeviceIPCWrapper]:
    frame = msgspec.msgpack.encode([msgspec.msgpack.Ext(1, payload)])
    return get_customized_decoder(list[DeviceIPCWrapper]).decode(frame)


def _decode_grpc(payload: bytes) -> DeviceIPCWrapper:
    return get_message_codecs()[0].reader(
        common_pb2.DeviceIpcWrapper(pickled_payload=payload)
    )


class _ExecutionMarker:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __reduce__(self) -> tuple[Callable[..., None], tuple[Path]]:
        return Path.touch, (self.path,)


@pytest.mark.parametrize(
    "decode", [DeviceIPCWrapper.Deserialize, _decode_zmq, _decode_grpc]
)
@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_pickle_rejected_before_execution(
    tmp_path: Path, decode, protocol: int
) -> None:
    """Both real transport codecs reject pickle before its callback can run."""
    marker = tmp_path / "executed"
    payload = pickle.dumps(_ExecutionMarker(marker), protocol=protocol)
    with pytest.raises(msgspec.DecodeError):
        decode(payload)
    assert not marker.exists()


@pytest.mark.parametrize("descriptor", DESCRIPTORS)
def test_descriptor_roundtrip_through_both_transports(descriptor: WireIPCData) -> None:
    """Concrete wrapper types and all transmitted fields survive both codecs."""
    wrapper = DeviceIPCWrapper.Deserialize(msgspec.msgpack.encode(descriptor))
    encoded = get_customized_encoder(list[DeviceIPCWrapper]).encode([wrapper])
    zmq_copy = get_customized_decoder(list[DeviceIPCWrapper]).decode(encoded)[0]
    message = common_pb2.DeviceIpcWrapper()
    codec = get_message_codecs()[0]
    codec.writer(message, wrapper)
    grpc_copy = codec.reader(message)
    for restored in (zmq_copy, grpc_copy):
        assert type(restored) is type(wrapper)
        assert restored.to_wire() == descriptor


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("type", "os.system"),
        ("_fd", 17),
        ("nbytes", -1),
        ("nbytes", 1),
    ],
)
def test_invalid_descriptor_rejected(field: str, value: object) -> None:
    descriptor = msgspec.to_builtins(DESCRIPTORS[6])
    descriptor[field] = value
    with pytest.raises(msgspec.DecodeError):
        DeviceIPCWrapper.Deserialize(msgspec.msgpack.encode(descriptor))


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "load"),
        ("shape", [-1, 3]),
        ("stride", [1]),
        ("storage_offset", -1),
        ("storage_offset", 999),
    ],
)
def test_invalid_metadata_rejected(field: str, value: object) -> None:
    descriptor = msgspec.to_builtins(DESCRIPTORS[6])
    descriptor["metadata"][field] = value
    with pytest.raises(msgspec.DecodeError):
        DeviceIPCWrapper.Deserialize(msgspec.msgpack.encode(descriptor))


def test_musa_opaque_handle_and_nested_extension_rejected() -> None:
    with pytest.raises(ValueError, match="primitive"):
        MusaIPCData(version=1, metadata=METADATA, handle=object())
    descriptor = msgspec.to_builtins(DESCRIPTORS[7])
    descriptor["handle"] = {"nested": [msgspec.msgpack.Ext(1, b"untrusted")]}
    with pytest.raises(msgspec.DecodeError, match="primitive"):
        DeviceIPCWrapper.Deserialize(msgspec.msgpack.encode(descriptor))


def test_cpu_shm_wire_preserves_shared_storage() -> None:
    """The new descriptor still reconstructs the actual shared tensor view."""
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    wrapper = CpuShmTensorWrapper.wrap(tensor)
    restored = DeviceIPCWrapper.Deserialize(DeviceIPCWrapper.Serialize(wrapper))
    view = restored.to_tensor()
    torch.testing.assert_close(view, tensor)
    view[0, 0] = 42
    assert tensor[0, 0].item() == 42
