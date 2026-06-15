# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Any, Callable
import pickle
import threading

# Third Party
import msgspec
import torch

# First Party
from lmcache import torch_dev, torch_device_type

"""
Defines the types and the customized encoder/decoders for inter-process
communications.

Key Types:
- IPCCacheServerKey: Token-based cache key
  - Contains token_ids, start, end, request_id (all required)
  - Converted to ObjectKey for storage operations via ipc_key_to_object_keys()
"""


class CudaIPCWrapper:
    _discovered_device_mapping: dict[str, int] = {}
    _device_mapping_lock = threading.Lock()

    @staticmethod
    def _get_device_uuid(device_index: int) -> str:
        """Get the UUID of a GPU device given its index."""
        return str(torch_dev.get_device_properties(device_index).uuid)

    @staticmethod
    def _discover_gpu_devices():
        """Discover all available GPU devices and map their UUIDs to
        the physical device ordinals.
        """
        if not torch_dev.is_available():
            return

        num_devices = torch_dev.device_count()
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
        # First Party
        from lmcache.v1.gpu_connector.utils import attempt_permute_to_contiguous_view

        # Permute any non-contiguous view (e.g. vLLM's NHD-over-HND) so the
        # shape/stride we encode across IPC reflects the physical layout.
        # Offset is preserved by the wrapper's storage_offset field.
        tensor = attempt_permute_to_contiguous_view(tensor)

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
            This function may break if the accelerator is not initialized.
            We should call `torch_dev.init()` before using this function
            (guarded by hasattr since not all backends expose init()).
        """
        device_index = CudaIPCWrapper._get_device_index_from_uuid(self.device_uuid)

        storage = torch.UntypedStorage._new_shared_cuda(  # noqa: SLF001
            device_index, *self.handle[1:]
        )

        t = torch.empty(
            (), device=f"{torch_device_type}:{device_index}", dtype=self.dtype
        )
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


class RawCudaIPCWrapper(CudaIPCWrapper):
    """IPC wrapper for CUDA tensors allocated outside PyTorch's caching
    allocator.

    PyTorch's ``UntypedStorage._share_cuda_()`` only works for tensors
    backed by its own caching allocator. TRT-LLM publishes its KV pool
    via ``at::for_blob`` over a ``cudaMalloc``'d buffer, which raises in
    ``_share_cuda_()``. This subclass bypasses that path: it calls
    ``cudaIpcGetMemHandle`` on the raw data pointer, then reconstructs
    the tensor on the receiving side via ``cudaIpcOpenMemHandle`` plus
    a CuPy ``UnownedMemory`` → DLPack → ``torch`` round-trip.

    Subclassing (rather than introducing a parallel class with its own
    msgspec ext code) is load-bearing — msgspec does not support unions
    of custom ext-encoded types. With subclassing, ``KVCache =
    list[CudaIPCWrapper]`` continues to type-check, the existing ext
    code 1 round-trips both wrappers, and pickle preserves the subclass
    identity through the wire so ``to_tensor`` dispatches correctly.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import assert_contiguous

        assert_contiguous(tensor)

        try:
            # Third Party
            from cuda.bindings import runtime as cudart
        except ImportError:
            # Third Party
            from cuda import cudart

        data_ptr = tensor.data_ptr()
        err, ipc_handle = cudart.cudaIpcGetMemHandle(data_ptr)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaIpcGetMemHandle failed: {err} (ptr=0x{data_ptr:x})"
            )

        # Store only what's needed for reconstruction.
        self._ipc_handle_reserved = bytes(ipc_handle.reserved)
        self._nbytes = tensor.untyped_storage().nbytes()

        # CudaIPCWrapper interface fields. ``handle`` is unused —
        # ``to_tensor`` is overridden to bypass it — but kept for
        # equality/identity checks against the parent class.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = CudaIPCWrapper._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process via raw CUDA IPC."""
        # Third Party
        import cupy

        try:
            # Third Party
            from cuda.bindings import runtime as cudart
        except ImportError:
            # Third Party
            from cuda import cudart

        device_index = CudaIPCWrapper._get_device_index_from_uuid(self.device_uuid)

        handle = cudart.cudaIpcMemHandle_t()
        handle.reserved = self._ipc_handle_reserved
        err, ptr = cudart.cudaIpcOpenMemHandle(
            handle, cudart.cudaIpcMemLazyEnablePeerAccess
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed: {err}")

        # Wrap as a flat ``uint8`` CuPy array, DLPack to torch, then view
        # as the original dtype/shape. ``uint8`` avoids dtype-conversion
        # gaps (bfloat16, fp8 have no direct CuPy/NumPy equivalent without
        # ml_dtypes).
        with cupy.cuda.Device(device_index):
            mem = cupy.cuda.UnownedMemory(ptr, self._nbytes, owner=self)
            memptr = cupy.cuda.MemoryPointer(mem, 0)
            cp_flat = cupy.ndarray(self._nbytes, dtype=cupy.uint8, memptr=memptr)

        raw = torch.from_dlpack(cp_flat)
        return raw.view(self.dtype).reshape(self.shape)


@dataclass(order=True, frozen=True)
class IPCCacheServerKey:
    """Cache key for the IPC (multiprocess) protocol.

    This key type is sent by the client over ZMQ (serialized via msgspec).

    The client sends token_ids, start, end, and request_id (all required).
    The server computes chunk hashes via TokenHasher and converts to
    ObjectKey for storage operations using ipc_key_to_object_keys().

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

    # === Per-user isolation salt (part of cache identity) ===
    # msgspec encodes dataclasses as maps, so forward wire compatibility
    # works by field name: an old payload without ``cache_salt`` decodes
    # on new code using the default "". Placing the field last is a style
    # choice — all defaulted fields must come after non-defaulted ones.
    #
    # Invariant: must not contain ``@``, ``/``, ``\``, or NUL, and
    # must be <= 128 chars — same rationale as ObjectKey (see
    # ObjectKey.cache_salt). Validated in __post_init__.
    cache_salt: str = ""

    # Duplicated from ObjectKey — cannot import ObjectKey here due to
    # circular dependency (api.py imports IPCCacheServerKey).
    _SALT_FORBIDDEN_CHARS = frozenset("@/\\\x00")
    _SALT_MAX_LEN = 128

    def __post_init__(self) -> None:
        bad = self._SALT_FORBIDDEN_CHARS & set(self.cache_salt)
        if bad:
            raise ValueError(
                f"cache_salt must not contain {bad!r} (got {self.cache_salt!r})"
            )
        if len(self.cache_salt) > self._SALT_MAX_LEN:
            raise ValueError(
                f"cache_salt exceeds max length {self._SALT_MAX_LEN} "
                f"(got {len(self.cache_salt)})"
            )

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
        cache_salt: str = "",
    ) -> "IPCCacheServerKey":
        """Create a key from token ids. Only used by the tests."""
        return cls(
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        )

    def no_worker_id_version(self) -> "IPCCacheServerKey":
        """Create a copy with worker_id=None for lookup requests."""
        return IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=self.token_ids,
            start=self.start,
            end=self.end,
            request_id=self.request_id,
            cache_salt=self.cache_salt,
        )


# Type exports
KVCache = list[CudaIPCWrapper]


class RegisterNonGpuContextPayload(msgspec.Struct):
    """Payload for the REGISTER_KV_CACHE_NON_GPU_CONTEXT protocol message.

    Attributes:
        instance_id: Worker instance identifier (typically PID).
        model_name: Model name associated with this worker.
        world_size: Worker world size used in cache keys.
        block_size: Tokens per paged block.
        num_layers: Number of model layers.
        hidden_dim_size: Flattened hidden dimension per token.
        dtype_str: Torch dtype name (e.g. ``"float16"``).
        use_mla: Whether the worker KV format is MLA.
    """

    instance_id: int
    model_name: str
    world_size: int
    block_size: int
    num_layers: int
    hidden_dim_size: int
    dtype_str: str
    use_mla: bool


@dataclass
class CustomizedSerdeConfig:
    serializer: Callable[[Any], bytes]
    deserializer: Callable[[bytes], Any]
    code: int


@dataclass
class BlockAllocationRecord:
    """A single per-request GPU block allocation delta from vLLM."""

    req_id: str
    new_block_ids: list[int]
    new_token_ids: list[int]


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


@dataclass
class CBUnifiedLookupResult:
    """Resolved payload of ``CB_UNIFIED_LOOKUP``: prefix lookup + non-prefix
    fingerprint match, reconciled in one RPC. The RPC returns ``None`` (not this)
    while either leg's KV is still loading into L1; this type is sent only once
    both are resident.

    Attributes:
        prefix_coverage_tokens: Contiguous prefix-cache coverage (L1+L2) in
            tokens — what the standard LOOKUP would report.
        non_prefix_segments: Fingerprint matches outside the prefix coverage
            (cur_st order), each carrying ``(old_st, old_ed, cur_st, cur_ed,
            hash)``. Already sparse-prefetched, so the retrieve set equals the
            prefetched set. Includes fleet-coordinator (shared-L2) matches:
            those are merged in before the sparse prefetch -- prefix-covered and
            locally-duplicated ones dropped -- so they ride the identical
            prefetch + retrieve path and need no separate handling.
    """

    prefix_coverage_tokens: int
    non_prefix_segments: list[CBMatchResult]


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
