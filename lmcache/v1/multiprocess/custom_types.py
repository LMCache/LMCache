# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
import pickle
import threading

# Third Party
import msgspec
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.token_hasher import TokenHasher

"""
Defines the types and the customized encoder/decoders for inter-process
communications.

Key Types:
- IPCCacheEngineKey: Token-based cache key
  - Contains token_ids, start, end, request_id (all required)
  - chunk_hash is optionally set after hashing by the server
  - Converted to ObjectKey for storage operations via ipc_keys_to_object_keys()
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

    @staticmethod
    def _validate_tensor_for_ipc(tensor: torch.Tensor) -> None:
        if not tensor.is_cuda:
            raise ValueError("CudaIPCWrapper only supports CUDA tensors.")
        if tensor.is_sparse:
            raise ValueError("sparse tensors are not supported for CUDA IPC sharing.")
        # disallow negative strides (possible via as_strided)
        if any(s < 0 for s in tensor.stride()):
            raise ValueError(
                "negative strides are not supported for IPC reconstruction."
            )

        storage = tensor.untyped_storage()
        if storage.device.type != "cuda":
            raise ValueError("tensor storage is not CUDA.")

        offset_elems = tensor.storage_offset()
        sizes = tensor.size()
        strides = tensor.stride()
        itemsize = tensor.element_size()

        # edge case: if the tensor is empty, return
        if tensor.numel() == 0:
            return

        # compute max idx (in elements) for the strided view
        # start at the offset
        max_index = offset_elems
        for sz, st in zip(sizes, strides, strict=False):
            # edge case: if any size is 0, then whole tensor is empty, and return
            if sz == 0:
                return
            max_index += (sz - 1) * st

        required_bytes = (max_index + 1) * itemsize
        if required_bytes > storage.nbytes():
            raise ValueError(
                f"tensor view exceeds underlying storage: need {required_bytes} bytes, "
                f"storage has {storage.nbytes()} bytes."
            )

    def __init__(self, tensor: torch.Tensor):
        self._validate_tensor_for_ipc(tensor)

        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = CudaIPCWrapper._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """
        Note:
            This function may break if torch cuda is not initialized.
            We should call `torch.cuda.init()` before using this function.
        """
        device_index = CudaIPCWrapper._get_device_index_from_uuid(self.device_uuid)

        storage = torch.UntypedStorage._new_shared_cuda(  # noqa: SLF001
            device_index, *self.handle[1:]
        )

        t = torch.empty((), device=f"cuda:{device_index}", dtype=self.dtype)
        t.set_(storage, self.storage_offset, self.shape, self.stride)
        return t

    def __eq__(self, other):
        if not isinstance(other, CudaIPCWrapper):
            return False
        return (
            self.handle == other.handle
            and self.dtype == other.dtype
            and self.shape == other.shape
            and self.stride == other.stride
            and self.storage_offset == other.storage_offset
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
    """Cache key for the IPC (multiprocess) protocol.

    This key type is sent by the client over ZMQ (serialized via msgspec).

    The client sends token_ids, start, end, and request_id (all required).
    The server computes chunk hashes via TokenHasher and converts to
    ObjectKey for storage operations.

    The request_id field is for session tracking and is NOT included
    in equality/hash comparisons (two keys with same content but different
    request_ids are considered equal for cache purposes).
    """

    model_name: str
    world_size: int
    worker_id: int | None

    token_ids: tuple[int, ...]  # frozen tuple for hashability
    start: int
    end: int

    # === Session tracking (not part of cache identity) ===
    request_id: str = field(compare=False)

    chunk_hash: bytes | None = None

    def to_hash_keys(
        self,
        hasher: "TokenHasher",
        full_chunk_only: bool = True,
        prefix_hash: int | None = None,
    ) -> list["IPCCacheEngineKey"]:
        """Compute chunk hashes and return one IPCCacheEngineKey per chunk.

        Preserves all fields in generated keys.

        Args:
            hasher: TokenHasher instance to compute chunk hashes
            full_chunk_only: If True, only return keys for full chunks .
                Else, return keys for all chunks (including partial ones).
            prefix_hash: Optional int hash to combine with token_ids.
        """
        chunk_hashes = hasher.compute_chunk_hashes(
            list(self.token_ids), full_chunk_only, prefix_hash
        )
        return [
            IPCCacheEngineKey(
                model_name=self.model_name,
                world_size=self.world_size,
                worker_id=self.worker_id,
                token_ids=self.token_ids,
                start=self.start,
                end=self.end,
                request_id=self.request_id,
                chunk_hash=hasher.hash_to_bytes(h),
            )
            for h in chunk_hashes
        ]

    # Helper function for unit tests only
    @classmethod
    def from_token_ids(
        cls,
        model_name: str,
        world_size: int,
        worker_id: int | None,
        token_ids: list[int],
        start: int = 0,
        end: int = 0,
        request_id: str = "",
    ) -> "IPCCacheEngineKey":
        """Create a key from token ids. Only used by the tests."""
        return cls(
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
        )

    def no_worker_id_version(self) -> "IPCCacheEngineKey":
        """Create a copy with worker_id=None for lookup requests."""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=self.token_ids,
            start=self.start,
            end=self.end,
            chunk_hash=self.chunk_hash,
            request_id=self.request_id,
        )


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


@dataclass
class CBMatchResult:
    """Result of a sub-sequence match from BlendTokenRangeMatcher.

    Attributes:
        old_st: Start position in the originally registered (stored) sequence.
        old_ed: End position in the originally registered (stored) sequence.
        cur_st: Start position in the query sequence where the match was found.
        cur_ed: End position in the query sequence where the match was found.
        hash: Token hash bytes (from registration) used as the storage key.
    """

    old_st: int
    old_ed: int
    cur_st: int
    cur_ed: int
    hash: bytes
