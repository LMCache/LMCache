# SPDX-License-Identifier: Apache-2.0
"""
TurboQuant serde backend for LMCache.

This backend is intended to compress LMCache KV tensors before L2 store and
decompress them after L2 load.

Initial scope:
- First preset: turboquant_k8v4
- Input KV layout: [2, num_layers, num_tokens, hidden_dim]
- Serialized layout: [num_layers, num_blocks, block_size, num_heads, slot_size]
"""

# Standard
from dataclasses import dataclass
import math

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.memory_management import MemoryObj


@dataclass(frozen=True)
class TurboQuantSerdeConfig:
    """Configuration for TurboQuant serde."""

    preset: str = "turboquant_k8v4"
    head_dim: int = 128
    block_size: int = 16

    @property
    def key_fp8(self) -> bool:
        return self.preset == "turboquant_k8v4"

    @property
    def key_quant_bits(self) -> int:
        if self.preset == "turboquant_k8v4":
            return 8
        raise NotImplementedError(f"Unsupported TurboQuant preset: {self.preset}")

    @property
    def key_mse_bits(self) -> int:
        if self.key_fp8:
            return 0
        return self.key_quant_bits

    @property
    def value_quant_bits(self) -> int:
        if self.preset == "turboquant_k8v4":
            return 4
        raise NotImplementedError(f"Unsupported TurboQuant preset: {self.preset}")

    @property
    def key_packed_size(self) -> int:
        if self.key_fp8:
            return self.head_dim
        mse_bytes = math.ceil(self.head_dim * self.key_mse_bits / 8)
        norm_bytes = 2
        return mse_bytes + norm_bytes

    @property
    def value_packed_size(self) -> int:
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4  # scale fp16 + zero fp16

    @property
    def slot_size(self) -> int:
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        s = self.slot_size
        return s + (s % 2)


def _validate_layout_shape(
    shape: torch.Size, cfg: TurboQuantSerdeConfig
) -> tuple[int, int, int, int]:
    """Validate LMCache KV layout and return L, T, H, D.

    Expected input shape:
        [2, num_layers, num_tokens, hidden_dim]

    Returns:
        num_layers, num_tokens, num_heads, head_dim
    """
    if len(shape) != 4:
        raise ValueError(
            "TurboQuant serde expects 4D KV tensor "
            f"[2, L, T, hidden_dim], got {tuple(shape)}"
        )
    if int(shape[0]) != 2:
        raise ValueError(
            f"TurboQuant serde expects first dim kv_size=2, got {int(shape[0])}"
        )

    num_layers = int(shape[1])
    num_tokens = int(shape[2])
    hidden_dim = int(shape[3])
    head_dim = cfg.head_dim

    if hidden_dim % head_dim != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by head_dim={head_dim}"
        )

    num_heads = hidden_dim // head_dim
    return num_layers, num_tokens, num_heads, head_dim


def _serialized_nbytes_for_shape(shape: torch.Size, cfg: TurboQuantSerdeConfig) -> int:
    """Return serialized size in bytes for one LMCache KV tensor."""
    num_layers, num_tokens, num_heads, _ = _validate_layout_shape(shape, cfg)
    num_blocks = math.ceil(num_tokens / cfg.block_size)
    return (
        num_layers
        * num_blocks
        * cfg.block_size
        * num_heads
        * cfg.slot_size_aligned
    )


def _compressed_layout_for_shape(
    shape: torch.Size, cfg: TurboQuantSerdeConfig
) -> tuple[int, int, int, int, int]:
    """Return compressed layout [L, num_blocks, block_size, H, slot_size]."""
    num_layers, num_tokens, num_heads, _ = _validate_layout_shape(shape, cfg)
    num_blocks = math.ceil(num_tokens / cfg.block_size)
    return (
        num_layers,
        num_blocks,
        cfg.block_size,
        num_heads,
        cfg.slot_size_aligned,
    )


def _make_slot_mapping(num_tokens: int, device: torch.device) -> torch.Tensor:
    """Sequential slot mapping: token i -> slot i."""
    return torch.arange(num_tokens, device=device, dtype=torch.int32)


def _make_block_table(num_blocks: int, device: torch.device) -> torch.Tensor:
    """Sequential block table: logical block i -> physical block i."""
    return torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, num_blocks)


def _select_cuda_device(*tensors: torch.Tensor) -> torch.device:
    """Select a CUDA device for Triton work.

    If any tensor is already on CUDA, reuse its device. Otherwise use the
    current CUDA device. This allows StorageManager E2E paths whose L1
    MemoryObj tensors are CPU / pinned-memory tensors.
    """
    for tensor in tensors:
        if tensor.is_cuda:
            return tensor.device
    if not torch.cuda.is_available():
        raise RuntimeError("TurboQuant Triton serde requires CUDA")
    return torch.device("cuda", torch.cuda.current_device())


def _make_dummy_tq_tensors(
    cfg: TurboQuantSerdeConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create dummy PiT, midpoints, and centroids for k8v4.

    For turboquant_k8v4, the store path uses FP8 keys and does not use PiT
    or midpoints. The full-dequant path also does not use centroids when
    KEY_FP8=True. We still pass valid tensors to match kernel signatures.
    """
    if cfg.preset != "turboquant_k8v4":
        raise NotImplementedError(
            "Only turboquant_k8v4 dummy tensors are supported for now"
        )

    pi_t = torch.empty((cfg.head_dim, cfg.head_dim), device=device, dtype=torch.float32)
    midpoints = torch.empty((0,), device=device, dtype=torch.float32)
    centroids = torch.empty((1,), device=device, dtype=torch.float32)
    return pi_t, midpoints, centroids


class TurboQuantSerializer(Serializer):
    """TurboQuant serializer skeleton."""

    def __init__(self, cfg: TurboQuantSerdeConfig):
        self._cfg = cfg

    def serialize(self, src: MemoryObj, dst: MemoryObj) -> int:
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("TurboQuant serde requires src and dst to have tensors")

        n_bytes = _serialized_nbytes_for_shape(src_tensor.shape, self._cfg)
        if dst_tensor.numel() < n_bytes:
            raise ValueError(
                f"Destination buffer too small: got {dst_tensor.numel()} bytes, "
                f"need {n_bytes}"
            )

        if dst_tensor.dtype != torch.uint8:
            raise ValueError(
                f"TurboQuant serialized destination must be torch.uint8, got {dst_tensor.dtype}"
            )
        cuda_device = _select_cuda_device(src_tensor, dst_tensor)

        # StorageManager may provide CPU / pinned-memory MemoryObjs. Triton
        # kernels require CUDA tensors, so use temporary CUDA buffers when
        # necessary and copy the serialized bytes back to the original dst.
        src_work = src_tensor if src_tensor.is_cuda else src_tensor.to(cuda_device)
        dst_work = (
            dst_tensor
            if dst_tensor.is_cuda
            else torch.empty(n_bytes, dtype=torch.uint8, device=cuda_device)
        )

        from lmcache.v1.distributed.serde.turboquant_store_kernel import (
            triton_turboquant_store,
        )

        cfg = self._cfg
        num_layers, num_tokens, num_heads, head_dim = _validate_layout_shape(
            src_work.shape, cfg
        )
        compressed_shape = _compressed_layout_for_shape(src_work.shape, cfg)
        dst_view = dst_work.flatten()[:n_bytes].view(*compressed_shape)

        slot_mapping = _make_slot_mapping(num_tokens, cuda_device)
        pi_t, midpoints, _ = _make_dummy_tq_tensors(cfg, cuda_device)

        # LMCache layout: [2, L, T, hidden_dim]
        # Kernel input layout per layer: key/value [T, H, D]
        for layer_idx in range(num_layers):
            key = src_work[0, layer_idx].view(
                num_tokens, num_heads, head_dim
            ).contiguous()
            value = src_work[1, layer_idx].view(
                num_tokens, num_heads, head_dim
            ).contiguous()
            kv_cache_layer = dst_view[layer_idx]

            triton_turboquant_store(
                key,
                value,
                kv_cache_layer,
                slot_mapping,
                pi_t,
                midpoints,
                mse_bits=cfg.key_mse_bits,
                key_packed_size=cfg.key_packed_size,
                value_quant_bits=cfg.value_quant_bits,
                key_fp8=cfg.key_fp8,
            )

        if not dst_tensor.is_cuda:
            dst_tensor.flatten()[:n_bytes].copy_(dst_work.cpu().flatten()[:n_bytes])

        return n_bytes

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        total = 0
        for shape in layout_desc.shapes:
            total += _serialized_nbytes_for_shape(shape, self._cfg)
        return total


class TurboQuantDeserializer(Deserializer):
    """TurboQuant deserializer skeleton."""

    def __init__(self, cfg: TurboQuantSerdeConfig):
        self._cfg = cfg

    def deserialize(self, src: MemoryObj, dst: MemoryObj) -> None:
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("TurboQuant serde requires src and dst to have tensors")

        n_bytes = _serialized_nbytes_for_shape(dst_tensor.shape, self._cfg)
        if src_tensor.numel() < n_bytes:
            raise ValueError(
                f"Source buffer too small: got {src_tensor.numel()} bytes, "
                f"need {n_bytes}"
            )

        if src_tensor.dtype != torch.uint8:
            raise ValueError(
                f"TurboQuant serialized source must be torch.uint8, got {src_tensor.dtype}"
            )
        cuda_device = _select_cuda_device(src_tensor, dst_tensor)

        # StorageManager may provide CPU / pinned-memory MemoryObjs. Triton
        # kernels require CUDA tensors, so copy compressed bytes to CUDA and
        # dequantize into a CUDA temporary when the destination is CPU.
        src_work = (
            src_tensor
            if src_tensor.is_cuda
            else src_tensor.flatten()[:n_bytes].to(cuda_device)
        )
        dst_work = (
            dst_tensor
            if dst_tensor.is_cuda
            else torch.empty(dst_tensor.shape, dtype=dst_tensor.dtype, device=cuda_device)
        )

        from lmcache.v1.distributed.serde.turboquant_decode_kernel import (
            _tq_full_dequant_kv,
            _use_fp8_e4b15,
        )

        cfg = self._cfg
        num_layers, num_tokens, num_heads, head_dim = _validate_layout_shape(
            dst_work.shape, cfg
        )
        compressed_shape = _compressed_layout_for_shape(dst_work.shape, cfg)
        src_view = src_work.flatten()[:n_bytes].view(*compressed_shape)

        num_blocks = compressed_shape[1]
        alloc_len = num_blocks * cfg.block_size
        block_table = _make_block_table(num_blocks, cuda_device)
        _, _, centroids = _make_dummy_tq_tensors(cfg, cuda_device)

        block_d = 1 << (head_dim - 1).bit_length()
        val_data_bytes = math.ceil(head_dim * cfg.value_quant_bits / 8)
        mse_bytes = (
            math.ceil(head_dim * cfg.key_mse_bits / 8)
            if not cfg.key_fp8
            else head_dim
        )

        for layer_idx in range(num_layers):
            kv_cache_layer = src_view[layer_idx]

            k_out = torch.empty(
                (1, num_heads, alloc_len, head_dim),
                dtype=torch.float16,
                device=cuda_device,
            )
            v_out = torch.empty(
                (1, num_heads, alloc_len, head_dim),
                dtype=torch.float16,
                device=cuda_device,
            )

            grid = (alloc_len, num_heads)
            _tq_full_dequant_kv[grid](
                kv_cache_layer,
                block_table,
                centroids,
                k_out,
                v_out,
                k_out.stride(0),
                k_out.stride(1),
                k_out.stride(2),
                v_out.stride(0),
                v_out.stride(1),
                v_out.stride(2),
                kv_cache_layer.stride(0),
                kv_cache_layer.stride(1),
                kv_cache_layer.stride(2),
                block_table.stride(0),
                HEAD_DIM=head_dim,
                BLOCK_SIZE=cfg.block_size,
                NUM_KV_HEADS=num_heads,
                MSE_BYTES=mse_bytes,
                KPS=cfg.key_packed_size,
                VQB=cfg.value_quant_bits,
                VAL_DATA_BYTES=val_data_bytes,
                MSE_BITS=cfg.key_mse_bits,
                KEY_FP8=1 if cfg.key_fp8 else 0,
                BLOCK_D=block_d,
                NORM_CORRECTION=0,
                FP8_E4B15=_use_fp8_e4b15(cuda_device.index or 0),
                num_warps=4,
            )

            key = (
                k_out[0, :, :num_tokens, :]
                .transpose(0, 1)
                .contiguous()
                .view(num_tokens, num_heads * head_dim)
            )
            value = (
                v_out[0, :, :num_tokens, :]
                .transpose(0, 1)
                .contiguous()
                .view(num_tokens, num_heads * head_dim)
            )

            dst_work[0, layer_idx].copy_(key.to(dst_work.dtype))
            dst_work[1, layer_idx].copy_(value.to(dst_work.dtype))

        if not dst_tensor.is_cuda:
            dst_tensor.copy_(dst_work.cpu())


def _create_turboquant_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    preset = str(kwargs.get("preset", "turboquant_k8v4"))
    head_dim = int(kwargs.get("head_dim", 128))  # type: ignore[arg-type]
    block_size = int(kwargs.get("block_size", 16))  # type: ignore[arg-type]
    max_workers = int(kwargs.get("max_workers", 1))  # type: ignore[arg-type]

    cfg = TurboQuantSerdeConfig(
        preset=preset,
        head_dim=head_dim,
        block_size=block_size,
    )

    if cfg.preset != "turboquant_k8v4":
        raise NotImplementedError(
            "Initial TurboQuant serde only supports preset='turboquant_k8v4'"
        )

    return AsyncSerdeProcessor(
        TurboQuantSerializer(cfg),
        TurboQuantDeserializer(cfg),
        max_workers=max_workers,
    )


register_serde_factory("turboquant", _create_turboquant_serde)
