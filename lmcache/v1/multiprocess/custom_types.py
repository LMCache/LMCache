# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Any, Callable
import pickle
import threading

# Third Party
import msgspec
import torch

"""
Defines the types and the customized encoder/decoders for inter-process 
communications.
"""


class CudaIPCWrapper:
    _discovered_device_mapping: dict[str, int] = {}
    _device_mapping_lock = threading.Lock()

    @staticmethod
    def _get_device_uuid(device_index: int) -> str:
        """Get the UUID of a GPU device given its index."""
        return str(torch.cuda.get_device_properties(device_index).uuid)

    @staticmethod
    def _discover_gpu_devices():
        """Discover all available GPU devices and map their UUIDs to
        the physical device ordinals.
        """
        if not torch.cuda.is_available():
            return

        num_devices = torch.cuda.device_count()
        with CudaIPCWrapper._device_mapping_lock:
            if CudaIPCWrapper._discovered_device_mapping:
                return  # Already discovered

            for i in range(num_devices):
                device_uuid = CudaIPCWrapper._get_device_uuid(i)
                CudaIPCWrapper._discovered_device_mapping[device_uuid] = i

    @staticmethod
    def _get_device_index_from_uuid(device_uuid: str) -> int:
        """Get the physical device ordinal from its UUID."""
        CudaIPCWrapper._discover_gpu_devices()

        with CudaIPCWrapper._device_mapping_lock:
            device_index = CudaIPCWrapper._discovered_device_mapping.get(
                device_uuid, None
            )

        if device_index is None:
            raise RuntimeError(
                f"Device UUID {device_uuid} not found in the discovered devices."
                "Please make sure the process can see all the GPU devices"
            )
        return device_index

    def __init__(self, tensor: torch.Tensor):
        assert tensor.storage_offset() == 0
        assert tensor.is_contiguous()
        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        device_index = tensor.device.index
        self.device_uuid = CudaIPCWrapper._get_device_uuid(device_index)

    def to_tensor(self):
        """
        Note:
            This function may break if torch cuda is not initialized.
            We should call `torch.cuda.init()` before using this function.
        """
        device = CudaIPCWrapper._get_device_index_from_uuid(self.device_uuid)
        storage = torch.UntypedStorage._new_shared_cuda(device, *self.handle[1:])
        t = torch.tensor(0, device=device, dtype=self.dtype)
        t.set_(storage)
        return t.view(self.shape)

    def __eq__(self, other):
        if not isinstance(other, CudaIPCWrapper):
            return False
        return (
            self.handle == other.handle
            and self.dtype == other.dtype
            and self.shape == other.shape
            and self.device_uuid == other.device_uuid
        )

    @staticmethod
    def Serialize(obj: "CudaIPCWrapper") -> bytes:
        return pickle.dumps(obj)

    @staticmethod
    def Deserialize(data: bytes) -> "CudaIPCWrapper":
        return pickle.loads(data)


@dataclass(order=True, frozen=True)
class IPCCacheEngineKey:
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: bytes

    @staticmethod
    def IntHash2Bytes(chunk_hash: int) -> bytes:
        # NOTE: this is only used by tests
        return chunk_hash.to_bytes(4, byteorder="big")

    @staticmethod
    def Bytes2IntHash(chunk_hash: bytes) -> int:
        # NOTE: this is only used by tests
        return int.from_bytes(chunk_hash, byteorder="big") & ((1 << 64) - 1)

    @classmethod
    def from_int_hash(
        cls, model_name: str, world_size: int, worker_id: int, chunk_hash: int
    ) -> "IPCCacheEngineKey":
        # NOTE: this is only used by tests
        return cls(
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            chunk_hash=cls.IntHash2Bytes(chunk_hash),
        )

    @staticmethod
    def Serialize(obj: "IPCCacheEngineKey") -> bytes:
        return msgspec.msgpack.encode(obj)

    @staticmethod
    def Deserialize(data: bytes) -> "IPCCacheEngineKey":
        return msgspec.msgpack.decode(data, type=IPCCacheEngineKey)


# Type exports
KVCache = list[CudaIPCWrapper]


@dataclass
class CustomizedSerdeConfig:
    serializer: Callable[[Any], bytes]
    deserializer: Callable[[bytes], Any]
    code: int


_CUSTOMERIZED_SERIALIZERS = {
    CudaIPCWrapper: CustomizedSerdeConfig(
        serializer=CudaIPCWrapper.Serialize,
        deserializer=CudaIPCWrapper.Deserialize,
        code=1,
    ),
}


def get_customized_encoder(type: Any) -> msgspec.msgpack.Encoder:
    # TODO: `type` is not used here
    def enc_hook(obj: Any) -> Any:
        for supported_type, cfg in _CUSTOMERIZED_SERIALIZERS.items():
            if isinstance(obj, supported_type):
                data = cfg.serializer(obj)
                return msgspec.msgpack.Ext(cfg.code, data)
        raise TypeError(f"Unsupported type for serialization: {type(obj)}")

    return msgspec.msgpack.Encoder(enc_hook=enc_hook)


def get_customized_decoder(type: Any) -> msgspec.msgpack.Decoder:
    def ext_hook(code: int, data: bytes) -> Any:
        for cfg in _CUSTOMERIZED_SERIALIZERS.values():
            if cfg.code == code:
                return cfg.deserializer(data)
        raise TypeError(f"Unsupported ext code for deserialization: {code}")

    return msgspec.msgpack.Decoder(ext_hook=ext_hook, type=type)
