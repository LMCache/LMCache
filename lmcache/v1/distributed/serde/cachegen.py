# SPDX-License-Identifier: Apache-2.0
"""CacheGen serde for multiprocess distributed L2 adapters."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import math

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata


def _parse_required_str(kwargs: dict[str, object], name: str) -> str:
    value = kwargs.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"cachegen serde requires non-empty {name!r}")
    return value


def _parse_positive_int(
    kwargs: dict[str, object],
    name: str,
    default: int | None = None,
) -> int:
    value = kwargs.get(name, default)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"cachegen serde requires positive integer {name!r}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    else:
        raise ValueError(f"cachegen serde requires positive integer {name!r}")
    if parsed <= 0:
        raise ValueError(f"cachegen serde requires positive integer {name!r}")
    return parsed


def _parse_dtype(kwargs: dict[str, object]) -> torch.dtype:
    dtype_name = _parse_required_str(kwargs, "dtype")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {dtype_name!r}")
    return dtype


def _metadata(model_name: str, dtype: torch.dtype) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name=model_name,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=None,  # type: ignore[arg-type]
    )


def mp_layout_to_cachegen_tensor(
    tensor: torch.Tensor,
    *,
    num_heads: int,
    head_size: int,
    chunk_size: int,
) -> torch.Tensor:
    """Convert MP KV layout to CacheGen KV layout.

    Args:
        tensor: MP-layout tensor with shape
            ``[2, num_layers, num_tokens, hidden_dim]``.
        num_heads: Number of KV heads.
        head_size: Per-head KV dimension.
        chunk_size: Maximum supported token count.

    Returns:
        CacheGen-layout tensor with shape
        ``[num_layers, 2, num_tokens, num_heads, head_size]``.

    Raises:
        ValueError: If the tensor shape is not a valid MP KV layout.
    """
    if tensor.ndim != 4:
        raise ValueError(
            "CacheGen MP layout expects shape [2, num_layers, num_tokens, hidden_dim]"
        )
    if tensor.shape[0] != 2:
        raise ValueError("CacheGen MP layout expects K/V dimension size 2")
    if tensor.shape[2] > chunk_size:
        raise ValueError(
            f"CacheGen MP layout got {tensor.shape[2]} tokens, "
            f"exceeding chunk_size {chunk_size}"
        )
    hidden_dim = int(tensor.shape[3])
    expected_hidden_dim = num_heads * head_size
    if hidden_dim != expected_hidden_dim:
        raise ValueError(
            f"CacheGen MP layout hidden_dim {hidden_dim} does not match "
            f"num_heads * head_size {expected_hidden_dim}"
        )
    return tensor.reshape(
        2,
        int(tensor.shape[1]),
        int(tensor.shape[2]),
        num_heads,
        head_size,
    ).permute(1, 0, 2, 3, 4)


def cachegen_layout_to_mp_tensor(
    tensor: torch.Tensor,
    dst_shape: torch.Size,
    *,
    num_heads: int,
    head_size: int,
) -> torch.Tensor:
    """Convert CacheGen KV layout to MP KV layout.

    Args:
        tensor: CacheGen-layout tensor with shape
            ``[num_layers, 2, num_tokens, num_heads, head_size]``.
        dst_shape: Target MP shape ``[2, num_layers, num_tokens, hidden_dim]``.
        num_heads: Number of KV heads.
        head_size: Per-head KV dimension.

    Returns:
        MP-layout tensor with shape ``dst_shape``.

    Raises:
        ValueError: If the decoded tensor and target shape are inconsistent.
    """
    if len(dst_shape) != 4:
        raise ValueError(
            "CacheGen MP destination expects shape "
            "[2, num_layers, num_tokens, hidden_dim]"
        )
    if dst_shape[0] != 2:
        raise ValueError("CacheGen MP destination expects K/V dimension size 2")
    expected_hidden_dim = num_heads * head_size
    if int(dst_shape[3]) != expected_hidden_dim:
        raise ValueError(
            f"CacheGen MP destination hidden_dim {int(dst_shape[3])} does not "
            f"match num_heads * head_size {expected_hidden_dim}"
        )
    expected_cachegen_shape = torch.Size(
        [int(dst_shape[1]), 2, int(dst_shape[2]), num_heads, head_size]
    )
    if tensor.shape != expected_cachegen_shape:
        raise ValueError(
            f"CacheGen decoded shape {tuple(tensor.shape)} does not match "
            f"destination-derived shape {tuple(expected_cachegen_shape)}"
        )
    return tensor.permute(1, 0, 2, 3, 4).reshape(dst_shape)


def _cachegen_gpu_max_tokens_per_chunk() -> int:
    # First Party
    from lmcache.storage_backend.serde.cachegen_basics import (  # noqa: PLC0415
        CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK,
    )

    return int(CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK)


def _create_inner_serializer(
    model_name: str,
    chunk_size: int,
    dtype: torch.dtype,
) -> Any:
    # First Party
    from lmcache.storage_backend.serde.cachegen_encoder import (  # noqa: PLC0415
        CacheGenSerializer,
    )

    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    return CacheGenSerializer(config, _metadata(model_name, dtype))


def _create_inner_deserializer(
    model_name: str,
    chunk_size: int,
    dtype: torch.dtype,
) -> Any:
    # First Party
    from lmcache.storage_backend.serde.cachegen_decoder import (  # noqa: PLC0415
        CacheGenDeserializer,
    )

    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    return CacheGenDeserializer(config, _metadata(model_name, dtype), dtype)


class CacheGenMpSerializer(Serializer):
    """Serialize KV tensors to CacheGen bytes for MP distributed L2 storage."""

    def __init__(
        self,
        model_name: str,
        chunk_size: int,
        num_heads: int | None = None,
        head_size: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Create a CacheGen MP serializer.

        Args:
            model_name: Model identifier used by CacheGen to select codec bins.
            chunk_size: Maximum number of tokens in one serialized KV object.
            num_heads: Number of KV heads in MP ``[2, L, T, D]`` tensors.
            head_size: Per-head KV dimension in MP ``[2, L, T, D]`` tensors.
            dtype: Source KV dtype used for CacheGen metadata.
        """
        self._model_name = model_name
        self._chunk_size = chunk_size
        self._num_heads = num_heads
        self._head_size = head_size
        self._inner = _create_inner_serializer(
            model_name=model_name,
            chunk_size=chunk_size,
            dtype=dtype,
        )
        self._cachegen_config = self._inner.cachegen_config

    def serialize(self, src: MemoryObj, dst: MemoryObj) -> int:
        """Encode ``src.tensor`` into ``dst`` and return bytes written.

        Args:
            src: Source MemoryObj containing an MP ``[2, L, T, D]`` tensor
                or a CacheGen-shaped ``[L, 2, T, H, HS]`` tensor.
            dst: Destination byte-buffer MemoryObj.

        Returns:
            Number of bytes written into ``dst``.

        Raises:
            ValueError: If tensors are missing, the source shape is invalid,
                or the destination buffer is too small.
        """
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None:
            raise ValueError("CacheGenMpSerializer requires src.tensor")
        if dst_tensor is None:
            raise ValueError("CacheGenMpSerializer requires dst.tensor")
        cachegen_tensor = self._to_cachegen_tensor(src_tensor)

        encode_tensor = (
            cachegen_tensor
            if cachegen_tensor.device.type == torch_device_type
            else cachegen_tensor.to(torch_device_type)
        )
        blob = self._inner.to_bytes(encode_tensor)
        n = len(blob)
        dst_view = dst_tensor.flatten().view(torch.uint8)
        if dst_view.numel() < n:
            raise ValueError(
                f"CacheGenMpSerializer destination capacity {dst_view.numel()} "
                f"is smaller than payload {n}"
            )
        dst_view[:n].copy_(torch.frombuffer(bytearray(blob), dtype=torch.uint8))
        return n

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return a conservative CacheGen payload capacity.

        Args:
            layout_desc: Source KV memory layout.

        Returns:
            Upper-bound byte capacity for the encoded CacheGen payload.
        """
        total = 0
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
            total += self._estimate_layout_size(shape, dtype)
        return int(total)

    def _estimate_layout_size(self, shape: torch.Size, dtype: torch.dtype) -> int:
        cachegen_shape = self._cachegen_shape_for_layout(shape)
        if cachegen_shape is None:
            return math.prod(shape) * dtype.itemsize

        num_layers, kv_dim, num_tokens, num_heads, head_size = (
            int(cachegen_shape[0]),
            int(cachegen_shape[1]),
            int(cachegen_shape[2]),
            int(cachegen_shape[3]),
            int(cachegen_shape[4]),
        )
        n_channels = num_heads * head_size
        source_bytes = math.prod(shape) * dtype.itemsize
        max_bins = max(
            [spec.bins for spec in self._cachegen_config.kspecs]
            + [spec.bins for spec in self._cachegen_config.vspecs]
        )
        cdf_bytes = kv_dim * num_layers * n_channels * (max_bins + 1) * 2
        max_tensors_bytes = kv_dim * num_layers * num_tokens * 4
        max_tokens_per_chunk = _cachegen_gpu_max_tokens_per_chunk()
        num_chunks = max(1, math.ceil(num_tokens / max_tokens_per_chunk))
        bytestream_bytes = (
            num_chunks * kv_dim * num_layers * n_channels * max_tokens_per_chunk
        )
        length_bytes = num_chunks * kv_dim * num_layers * n_channels * 4
        pickle_overhead_bytes = 4 << 20
        return max(source_bytes, bytestream_bytes) + (
            cdf_bytes
            + max_tensors_bytes
            + length_bytes
            + (pickle_overhead_bytes * num_chunks)
        )

    def _cachegen_shape_for_layout(self, shape: torch.Size) -> torch.Size | None:
        if len(shape) == 5 and shape[1] == 2:
            return shape
        if len(shape) == 4 and shape[0] == 2:
            self._require_mp_head_config()
            assert self._num_heads is not None
            assert self._head_size is not None
            hidden_dim = int(shape[3])
            expected_hidden_dim = self._num_heads * self._head_size
            if hidden_dim != expected_hidden_dim:
                raise ValueError(
                    f"CacheGen MP layout hidden_dim {hidden_dim} does not match "
                    f"num_heads * head_size {expected_hidden_dim}"
                )
            return torch.Size(
                [int(shape[1]), 2, int(shape[2]), self._num_heads, self._head_size]
            )
        return None

    def _to_cachegen_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 5:
            self._validate_cachegen_tensor(tensor)
            return tensor
        if tensor.ndim == 4:
            return self._mp_to_cachegen_tensor(tensor)
        raise ValueError(
            "CacheGenMpSerializer expects MP shape [2, num_layers, num_tokens, "
            "hidden_dim] or CacheGen shape [num_layers, 2, num_tokens, "
            "num_heads, head_size]"
        )

    def _mp_to_cachegen_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        self._require_mp_head_config()
        assert self._num_heads is not None
        assert self._head_size is not None
        return mp_layout_to_cachegen_tensor(
            tensor,
            num_heads=self._num_heads,
            head_size=self._head_size,
            chunk_size=self._chunk_size,
        )

    def _validate_cachegen_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.shape[1] != 2:
            raise ValueError("CacheGenMpSerializer expects K/V dimension size 2")
        if tensor.shape[2] > self._chunk_size:
            raise ValueError(
                f"CacheGenMpSerializer got {tensor.shape[2]} tokens, "
                f"exceeding chunk_size {self._chunk_size}"
            )
        if self._num_heads is not None and tensor.shape[3] != self._num_heads:
            raise ValueError(
                f"CacheGenMpSerializer got {tensor.shape[3]} heads, "
                f"expected {self._num_heads}"
            )
        if self._head_size is not None and tensor.shape[4] != self._head_size:
            raise ValueError(
                f"CacheGenMpSerializer got head_size {tensor.shape[4]}, "
                f"expected {self._head_size}"
            )

    def _require_mp_head_config(self) -> None:
        if self._num_heads is None or self._head_size is None:
            raise ValueError(
                "CacheGen MP layout requires num_heads and head_size for "
                "[2, num_layers, num_tokens, hidden_dim] tensors"
            )


class CacheGenMpDeserializer(Deserializer):
    """Deserialize CacheGen bytes back into caller-provided KV tensors."""

    def __init__(
        self,
        model_name: str,
        chunk_size: int,
        dtype: torch.dtype,
        num_heads: int | None = None,
        head_size: int | None = None,
    ) -> None:
        """Create a CacheGen MP deserializer.

        Args:
            model_name: Model identifier used by CacheGen to select codec bins.
            chunk_size: Maximum number of tokens in one serialized KV object.
            dtype: Expected destination KV dtype.
            num_heads: Number of KV heads in MP ``[2, L, T, D]`` tensors.
            head_size: Per-head KV dimension in MP ``[2, L, T, D]`` tensors.
        """
        self._dtype = dtype
        self._chunk_size = chunk_size
        self._num_heads = num_heads
        self._head_size = head_size
        self._inner = _create_inner_deserializer(
            model_name=model_name,
            chunk_size=chunk_size,
            dtype=dtype,
        )

    def deserialize(self, src: MemoryObj, dst: MemoryObj) -> None:
        """Decode CacheGen bytes from ``src`` into ``dst.tensor``.

        Args:
            src: Source serialized byte-buffer MemoryObj.
            dst: Destination KV MemoryObj.

        Raises:
            ValueError: If destination tensor is missing, dtype mismatches,
                source bytes are unavailable, or decoded element count does
                not match the destination.
        """
        dst_tensor = dst.tensor
        if dst_tensor is None:
            raise ValueError("CacheGenMpDeserializer requires dst.tensor")
        if dst_tensor.dtype != self._dtype:
            raise ValueError(
                f"CacheGenMpDeserializer configured dtype {self._dtype} "
                f"does not match destination dtype {dst_tensor.dtype}"
            )

        blob = self._src_bytes(src)
        decoded = self._inner.from_bytes(blob)
        if decoded.numel() != dst_tensor.numel():
            raise ValueError(
                f"CacheGenMpDeserializer decoded {decoded.numel()} elements, "
                f"destination expects {dst_tensor.numel()}"
            )
        decoded = self._to_destination_layout(decoded, dst_tensor).to(
            device=dst_tensor.device,
            dtype=dst_tensor.dtype,
        )
        dst_tensor.copy_(decoded)

    def _to_destination_layout(
        self,
        decoded: torch.Tensor,
        dst_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if dst_tensor.ndim == 5:
            if decoded.shape != dst_tensor.shape:
                decoded = decoded.reshape(dst_tensor.shape)
            return decoded
        if dst_tensor.ndim == 4:
            return self._cachegen_to_mp_tensor(decoded, dst_tensor)
        raise ValueError(
            "CacheGenMpDeserializer expects MP shape [2, num_layers, num_tokens, "
            "hidden_dim] or CacheGen shape [num_layers, 2, num_tokens, "
            "num_heads, head_size]"
        )

    def _cachegen_to_mp_tensor(
        self,
        decoded: torch.Tensor,
        dst_tensor: torch.Tensor,
    ) -> torch.Tensor:
        self._require_mp_head_config()
        assert self._num_heads is not None
        assert self._head_size is not None
        if decoded.ndim != 5:
            raise ValueError(
                "CacheGenMpDeserializer decoded tensor must have shape "
                "[num_layers, 2, num_tokens, num_heads, head_size]"
            )
        return cachegen_layout_to_mp_tensor(
            decoded,
            dst_tensor.shape,
            num_heads=self._num_heads,
            head_size=self._head_size,
        )

    def _require_mp_head_config(self) -> None:
        if self._num_heads is None or self._head_size is None:
            raise ValueError(
                "CacheGen MP layout requires num_heads and head_size for "
                "[2, num_layers, num_tokens, hidden_dim] tensors"
            )

    def _src_bytes(self, src: MemoryObj) -> bytes:
        try:
            return memoryview(src.byte_array).cast("B").tobytes()
        except Exception:
            src_tensor = src.tensor
            if src_tensor is None:
                raise ValueError("CacheGenMpDeserializer requires src bytes") from None
            n = src.get_size()
            return (
                src_tensor.flatten()
                .view(torch.uint8)[:n]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            )


def _create_cachegen_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    model_name = _parse_required_str(kwargs, "model_name")
    chunk_size = _parse_positive_int(kwargs, "chunk_size")
    dtype = _parse_dtype(kwargs)
    num_heads = _parse_positive_int(kwargs, "num_heads")
    head_size = _parse_positive_int(kwargs, "head_size")
    max_workers = _parse_positive_int(kwargs, "max_workers", default=1)
    return AsyncSerdeProcessor(
        CacheGenMpSerializer(
            model_name=model_name,
            chunk_size=chunk_size,
            num_heads=num_heads,
            head_size=head_size,
            dtype=dtype,
        ),
        CacheGenMpDeserializer(
            model_name=model_name,
            chunk_size=chunk_size,
            dtype=dtype,
            num_heads=num_heads,
            head_size=head_size,
        ),
        max_workers=max_workers,
    )
