# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias
import pickle
import threading

# Third Party
import msgspec
import torch

# First Party
from lmcache.utils import EngineType

"""
Defines the types and the customized encoder/decoders for inter-process
communications.

Key Types:
- IPCCacheEngineKey: Token-based cache key
  - Contains token_ids, start, end, request_id (all required)
  - Converted to ObjectKey for storage operations via ipc_key_to_object_keys()
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


DeviceBufferView: TypeAlias = torch.Tensor


@dataclass(frozen=True)
class DeviceBufferDescriptor:
    """Backend-neutral descriptor for a registered serving-engine KV buffer.

    The descriptor is the Modular MAX / LMCache boundary object. CUDA IPC is one
    supported backend payload, but callers must still identify the backend and
    handle type explicitly so unsupported backends fail before touching CUDA
    code.
    """

    engine_type: EngineType
    backend: str
    handle_type: str
    handle: Any
    device_id: int
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    nbytes: int
    storage_offset_bytes: int
    layout_format: str
    layout_hints: dict[str, Any]


@dataclass(frozen=True)
class DeviceEventDescriptor:
    """Backend-neutral event descriptor for future async synchronization."""

    backend: str
    handle_type: str
    handle: bytes | object | None


class DeviceBufferImporter:
    """Import backend-neutral buffer descriptors into local tensor views."""

    _MAX_LAYOUT_FORMAT = "NB_KV_NL_BS_NH_HS"

    @staticmethod
    def from_descriptor(
        desc: DeviceBufferDescriptor,
        *,
        allow_host_staging: bool = False,
    ) -> DeviceBufferView:
        """Create a tensor view from ``desc``.

        Args:
            desc: Backend-neutral buffer descriptor.
            allow_host_staging: Whether explicit host staging fallback may be
                used for CPU/host descriptors.

        Returns:
            A tensor view over the registered buffer.

        Raises:
            RuntimeError: If the backend is unsupported or host staging was not
                explicitly enabled.
            TypeError: If the descriptor handle type is invalid.
        """
        engine_type = DeviceBufferImporter._normalize_engine_type(desc.engine_type)
        DeviceBufferImporter._validate_descriptor_metadata(desc, engine_type)
        backend = desc.backend.lower()
        handle_type = desc.handle_type.lower()

        if backend == "cuda":
            if handle_type != "cuda_ipc":
                raise RuntimeError(
                    "LMCache MP for Modular MAX CUDA backend requires "
                    f"handle_type='cuda_ipc', got {desc.handle_type!r}."
                )
            if isinstance(desc.handle, CudaIPCWrapper):
                tensor = desc.handle.to_tensor()
            elif isinstance(desc.handle, bytes):
                tensor = CudaIPCWrapper.Deserialize(desc.handle).to_tensor()
            else:
                raise TypeError(
                    "CUDA DeviceBufferDescriptor handle must be bytes or "
                    f"CudaIPCWrapper, got {type(desc.handle)!r}."
                )
            DeviceBufferImporter._validate_tensor_metadata(desc, tensor)
            return tensor

        if backend in ("rocm", "hip", "amd"):
            raise RuntimeError(
                "LMCache MP for Modular MAX does not yet support backend "
                f"{desc.backend!r}. Implement HipIpcBufferImporter or enable "
                "explicit host staging fallback."
            )

        if backend in ("cpu", "host"):
            if not allow_host_staging:
                raise RuntimeError(
                    "LMCache MP for Modular MAX received a host buffer descriptor, "
                    "but host staging fallback is disabled. Set "
                    "lmcache_allow_host_staging=true to opt in."
                )
            raise RuntimeError(
                "LMCache MP host staging descriptors are not implemented yet."
            )

        raise RuntimeError(
            f"LMCache MP for Modular MAX does not yet support backend {desc.backend!r}."
        )

    @staticmethod
    def _validate_tensor_metadata(
        desc: DeviceBufferDescriptor,
        tensor: torch.Tensor,
    ) -> None:
        if str(tensor.dtype) != desc.dtype:
            raise RuntimeError(
                "Imported device buffer dtype mismatch: descriptor "
                f"{desc.dtype!r}, tensor {str(tensor.dtype)!r}."
            )
        if tuple(tensor.shape) != tuple(desc.shape):
            raise RuntimeError(
                "Imported device buffer shape mismatch: descriptor "
                f"{desc.shape}, tensor {tuple(tensor.shape)}."
            )
        if desc.strides is not None and tuple(tensor.stride()) != tuple(desc.strides):
            raise RuntimeError(
                "Imported device buffer stride mismatch: descriptor "
                f"{desc.strides}, tensor {tuple(tensor.stride())}."
            )
        nbytes = tensor.untyped_storage().nbytes()
        if nbytes != desc.nbytes:
            raise RuntimeError(
                "Imported device buffer nbytes mismatch: descriptor "
                f"{desc.nbytes}, tensor {nbytes}."
            )
        storage_offset_bytes = tensor.storage_offset() * tensor.element_size()
        if storage_offset_bytes != desc.storage_offset_bytes:
            raise RuntimeError(
                "Imported device buffer storage offset mismatch: descriptor "
                f"{desc.storage_offset_bytes}, tensor {storage_offset_bytes}."
            )
        if tensor.device.type == "cuda":
            device_id = tensor.device.index or 0
            if device_id != desc.device_id:
                raise RuntimeError(
                    "Imported device buffer device mismatch: descriptor "
                    f"{desc.device_id}, tensor {device_id}."
                )

    @staticmethod
    def _normalize_engine_type(engine_type: EngineType | str) -> EngineType:
        if isinstance(engine_type, EngineType):
            return engine_type
        try:
            return EngineType(engine_type)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unsupported DeviceBufferDescriptor engine_type {engine_type!r}."
            ) from exc

    @staticmethod
    def _validate_descriptor_metadata(
        desc: DeviceBufferDescriptor,
        engine_type: EngineType,
    ) -> None:
        if not desc.backend:
            raise RuntimeError("DeviceBufferDescriptor backend must be non-empty.")
        if not desc.handle_type:
            raise RuntimeError("DeviceBufferDescriptor handle_type must be non-empty.")
        if not desc.dtype:
            raise RuntimeError("DeviceBufferDescriptor dtype must be non-empty.")
        if not desc.shape:
            raise RuntimeError("DeviceBufferDescriptor shape must be non-empty.")
        if desc.strides is not None and len(desc.strides) != len(desc.shape):
            raise RuntimeError(
                "DeviceBufferDescriptor strides must be None or match shape rank."
            )
        if desc.nbytes <= 0:
            raise RuntimeError("DeviceBufferDescriptor nbytes must be positive.")
        if desc.storage_offset_bytes < 0:
            raise RuntimeError(
                "DeviceBufferDescriptor storage_offset_bytes must be non-negative."
            )
        if not desc.layout_format:
            raise RuntimeError(
                "DeviceBufferDescriptor layout_format must be non-empty."
            )
        if not isinstance(desc.layout_hints, dict):
            raise TypeError(
                "DeviceBufferDescriptor layout_hints must be a dict, got "
                f"{type(desc.layout_hints)!r}."
            )
        if (
            engine_type == EngineType.MAX
            and desc.layout_format != DeviceBufferImporter._MAX_LAYOUT_FORMAT
        ):
            raise RuntimeError(
                "Modular MAX DeviceBufferDescriptor layout_format must be "
                f"{DeviceBufferImporter._MAX_LAYOUT_FORMAT!r}, got "
                f"{desc.layout_format!r}."
            )


@dataclass(order=True, frozen=True)
class IPCCacheEngineKey:
    """Cache key for the IPC (multiprocess) protocol.

    This key type is sent by the client over ZMQ (serialized via msgspec).

    The sender provides ``token_ids``, ``start``, and ``end``; the server
    computes chunk hashes via TokenHasher.

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
    # circular dependency (api.py imports IPCCacheEngineKey).
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
        if not self.token_ids:
            raise ValueError("IPCCacheEngineKey requires non-empty token_ids")

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
    ) -> "IPCCacheEngineKey":
        """Create a token-mode key for MP serving-engine adapters."""
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

    def no_worker_id_version(self) -> "IPCCacheEngineKey":
        """Create a copy with worker_id=None for lookup requests."""
        return IPCCacheEngineKey(
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
KVCache = list[Any]


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
