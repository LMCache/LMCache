# SPDX-License-Identifier: Apache-2.0
"""KVWeave attention codec for LMCache L2 quantization."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, Optional, cast
import struct

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.common.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    KVWeaveCodecConfig,
)

try:
    # First Party
    import lmcache.kvweave_quant as kvweave_quant
except ImportError:  # pragma: no cover
    kvweave_quant = None

try:
    # First Party
    import lmcache.kvweave_quant_xpu as kvweave_quant_xpu
except ImportError:  # pragma: no cover
    kvweave_quant_xpu = None

logger = init_logger(__name__)

_BytesLike = bytes | memoryview


def _cpu_thread_kwargs(device: str, num_threads: int) -> dict:
    """``num_threads`` kwarg for the CPU (OpenMP) native module only.

    Without it the CPU kernels use ``omp_get_max_threads()`` (torch's thread
    count), which oversubscribes against torch's own pool; the SYCL module
    has no such argument.
    """
    return {"num_threads": num_threads} if device == "cpu" and num_threads > 0 else {}


def _resolve_native(device: str):
    """Return the native quantization module for ``device`` ('cpu' or 'xpu').

    Single canonical native-module resolver used everywhere in this file --
    both modules expose identical kvweave_serialize_chunk*/
    kvweave_dequantize_chunk* signatures (the XPU ones added a native
    ``num_layers`` axis and fused K/V entry points in the SYCL kernels,
    mirroring the CPU wrapper's fused chunk entry points -- see
    quant_sycl.cpp / kvweave_quant_xpu_wrapper.cpp), so every call site here
    just picks a module and calls the same function name/arguments
    regardless of device.
    """
    if device == "xpu":
        if kvweave_quant_xpu is None:
            raise RuntimeError(
                "KVWeave native XPU quantization extension is unavailable; "
                "KVWeave is only built for Intel (SYCL) with BUILD_KVWEAVE=1"
            )
        return kvweave_quant_xpu
    if kvweave_quant is None:
        raise RuntimeError(
            "KVWeave native quantization extension is unavailable; "
            "KVWeave is only built for Intel (SYCL, x86-64) with BUILD_KVWEAVE=1"
        )
    return kvweave_quant


@dataclass
class _KVShape:
    tensor4d: torch.Tensor
    header_shape: tuple[int, ...]
    num_layers: int
    chunk_tokens: int
    hidden_dim: int
    original_ndim: int


class _KVWeaveCodec:
    """Self-describing attention KVWeave codec."""

    def __init__(self, kwargs: dict[str, Any] | None = None):
        kwargs = kwargs or {}
        config = KVWeaveCodecConfig(
            qbit=int(kwargs.get("qbit", 4)),
            scaling_method=str(kwargs.get("scaling_method", "per_channel")),
            rh=bool(kwargs.get("rh", True)),
            asym=bool(kwargs.get("asym", True)),
            precond=bool(kwargs.get("precond", False)),
            num_threads=int(kwargs.get("num_threads", 8)),
        )
        self._config = config
        self.qbit = config.qbit
        self.scaling_method = config.scaling_method
        self.rh = config.rh
        self.asym = config.asym
        self.precond = config.precond
        self.num_threads = config.num_threads
        self.device = str(kwargs.get("device", "cpu")).strip().lower()
        if self.device not in {"cpu", "xpu"}:
            raise ValueError(f"device={self.device!r} is not one of ['cpu', 'xpu']")
        if self.device == "xpu" and not (
            torch_device_type == "xpu" and torch_dev.is_available()
        ):
            raise RuntimeError("device='xpu' but no XPU device is available")
        self.native = _resolve_native(self.device)
        # fp8 KV cache: the engine hands over uint8 tensors; this says which fp8
        # format they hold.
        self.kv_fp8_dtype: torch.dtype | None = kwargs.get("kv_fp8_dtype")  # type: ignore[assignment]

    def supports_fp8_attention(self) -> bool:
        """Whether this codec can quantize an fp8 KV cache (fused-K/V attention).

        Both the XPU and the CPU kernels read and write fp8 natively.
        """
        return self.kv_fp8_dtype is not None and self.scaling_method in {
            "per_channel",
            "per_token",
        }

    def _kv_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Map the engine's uint8 fp8-KV storage dtype to the configured fp8 format."""
        if dtype == torch.uint8 and self.kv_fp8_dtype is not None:
            return self.kv_fp8_dtype
        return dtype

    def _kv_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reinterpret a uint8 fp8-KV tensor as the fp8 dtype (same memory)."""
        if tensor.dtype == torch.uint8 and self.kv_fp8_dtype is not None:
            return tensor.view(self.kv_fp8_dtype)
        return tensor

    def supports_fused_direct_quant(self) -> bool:
        """Whether fused-K/V attention quantizes without a separate head split."""
        return self.scaling_method == "per_channel"

    def fused_quant_uses_direct(
        self, hidden: int, scaling_method: str | None = None
    ) -> bool:
        """Whether ``serialize_fused_tensor`` takes the direct kernel for this
        hidden size, in which case the heads are never split on the host."""
        method = scaling_method or self.scaling_method
        per_head_scales = method != "per_tensor" and self._fused_head_num(hidden) > 1
        return per_head_scales and self.supports_fused_direct_quant()

    def supports_fused_direct_dequant(self) -> bool:
        """Whether fused-K/V attention decodes straight into the fused layout."""
        # CPU only: on XPU the wide-stride fused writes measured ~15-20% slower
        # than dequantizing to the split layout and merging heads on the device.
        return self.device == "cpu"

    @staticmethod
    def _payload_u8(tensor: torch.Tensor) -> torch.Tensor:
        payload = tensor.detach()
        if payload.dtype != torch.uint8:
            raise ValueError("KVWeave payload tensor must have dtype torch.uint8")
        if payload.device.type != "cpu":
            payload = payload.cpu()
        if not payload.is_contiguous():
            payload = payload.contiguous()
        return payload.view(torch.uint8)

    @staticmethod
    def _payload_buffer(payload: torch.Tensor) -> memoryview:
        return memoryview(payload.numpy())

    @staticmethod
    def _payload_tail(
        payload: torch.Tensor, offset: int, dtype: torch.dtype
    ) -> torch.Tensor:
        if dtype == torch.int8:
            return payload[offset:].view(torch.int8)
        if dtype == torch.uint8:
            return payload[offset:]
        if dtype == torch.int16:
            if offset % 2:
                raise ValueError("KVWeave int16 q_data is not 2-byte aligned")
            return payload[offset:].view(torch.int16)
        if dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        }:
            itemsize = dtype.itemsize
            tail = payload[offset:]
            if offset % itemsize:
                return torch.frombuffer(
                    bytearray(tail.contiguous().numpy().tobytes()), dtype=dtype
                )
            return tail.view(dtype)
        raise ValueError(f"unsupported KVWeave payload view dtype: {dtype}")

    def estimate_serialized_size(
        self, layout_desc: MemoryLayoutDesc, scaling_method: Optional[str] = None
    ) -> int:
        """Estimate the upper bound needed for one serialized KV buffer."""
        return sum(
            self._estimate_shape(
                tuple(int(dim) for dim in shape), dtype, scaling_method
            )
            for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
        )

    def estimate_fused_serialized_size(
        self, layout_desc: MemoryLayoutDesc, scaling_method: Optional[str] = None
    ) -> int:
        """Estimate the upper bound for one serialized fused-K/V KV buffer."""
        return sum(
            self._estimate_fused_shape(
                tuple(int(dim) for dim in shape), dtype, scaling_method
            )
            for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
        )

    def _estimate_fused_shape(
        self, shape: tuple[int, ...], dtype: torch.dtype, method: str | None
    ) -> int:
        """Exact serialized size of one fused-K/V ``[L, T, H]`` chunk.

        Layout: ``magic(4) + 5 flag bytes + 4*ndim shape + 4 scale_len``, then
        the scale blob (``12`` bytes per scale plus ``4`` per native layer),
        then the packed q data. Independent of ``rh``/``asym``/``precond``.
        """
        if len(shape) != 3:
            raise ValueError(
                f"KVWeave fused-K/V estimate expects [L, T, H], got {shape}"
            )
        layers, tokens, hidden = (int(dim) for dim in shape)
        method = method or self.scaling_method
        elements = layers * tokens * hidden
        scales = self._fused_scale_count(tokens, hidden, layers, method)
        fused_heads = self._fused_head_num(hidden)
        native_layers = (
            layers * fused_heads
            if method != "per_tensor" and fused_heads > 1
            else layers
        )
        header = 4 + 5 + 4 * len(shape) + 4
        return (
            header
            + scales * 12
            + 4 * native_layers
            + KVWeaveCodecConfig.quantized_bytes(elements, self.qbit)
        )

    def _estimate_shape(
        self, shape: tuple[int, ...], dtype: torch.dtype, method: str | None
    ) -> int:
        """Exact serialized size of one split-K/V ``[2, L, T, H]`` chunk.

        Layout: ``magic(4) + 8 flag bytes + 4*ndim shape``, then two scale
        blobs (``4`` length + ``12`` per scale + ``4`` per layer each), then
        the packed q data of both planes. Independent of ``rh``/``asym``.
        """
        if len(shape) == 3:
            tokens, kv_size, hidden = shape
            layers = 1
        elif len(shape) == 4:
            kv_size, layers, tokens, hidden = shape
        else:
            return 64 + dtype.itemsize * int(torch.tensor(shape).prod().item())
        if kv_size != 2:
            raise ValueError(f"KVWeave expects K/V dimension 2, got {shape}")
        method = method or self.scaling_method
        elements = int(layers) * int(tokens) * int(hidden)
        scales = self._scale_count(tokens, hidden, layers, method)
        header = 4 + 8 + 4 * len(shape)
        return (
            header
            + 2 * (4 + scales * 12 + 4 * int(layers))
            + 2 * KVWeaveCodecConfig.quantized_bytes(elements, self.qbit)
        )

    def serialize_tensor(
        self,
        tensor: torch.Tensor,
        scaling_method: str | None = None,
        out: torch.Tensor | None = None,
    ) -> bytes | int:
        """Normalize a split-K/V tensor and serialize its quantized payload.

        With ``out`` (CPU codec only) the payload is written into it and its
        length returned instead of a ``bytes`` object.
        """
        work = self._kv_view(tensor.detach().to("cpu").contiguous())
        shape = self._normalize(work)
        method = scaling_method or self.scaling_method
        if shape.hidden_dim <= 0:
            raise ValueError(
                "KVWeave attention quantization needs a non-empty hidden axis"
            )
        rh, asym, precond = (
            (False, False, False)
            if method == "per_tensor"
            else (self.rh, self.asym, self.precond)
        )
        header = self._quant_header(shape, work.dtype, method, rh, asym, precond)
        ids = KVWeaveCodecConfig.next_scale_ids(2)
        signs, perm = self._attention_precond(
            method, rh, precond, shape.chunk_tokens, shape.hidden_dim
        )
        out_kwargs = {"out": out} if out is not None else {}
        payload = self.native.kvweave_serialize_chunk(
            shape.tensor4d,
            header,
            ids[0],
            ids[1],
            qbit=self.qbit,
            blocks_num=1,
            block_size=shape.chunk_tokens,
            head_num=self._head_num(shape.hidden_dim),
            head_dim=self._head_dim(shape.hidden_dim),
            num_layers=shape.num_layers,
            rh=rh,
            asym=asym,
            scaling_method=method,
            signs=signs,
            perm=perm,
            num_threads=self.num_threads,
            **out_kwargs,
        )
        payload_bytes = payload if isinstance(payload, int) else len(payload)
        raw_bytes = work.numel() * work.element_size()
        logger.debug(
            "KVWeave quantize store shape=%s dtype=%s qbit=%d scaling=%s "
            "raw_bytes=%d payload_bytes=%d ratio=%.4f",
            tuple(work.shape),
            work.dtype,
            self.qbit,
            method,
            raw_bytes,
            payload_bytes,
            payload_bytes / raw_bytes if raw_bytes else 0.0,
        )
        return payload

    def deserialize_tensor(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Decode a payload and restore it into the destination KV tensor."""
        dst = self._kv_view(dst)
        payload = self._payload_u8(src)
        parsed = self._read_header(payload)
        shape = parsed["shape4d"]
        q_dtype = torch.int8 if parsed["qbit"] <= 8 else torch.int16
        q = self._payload_tail(payload, parsed["q_offset"], q_dtype)
        signs, perm = self._attention_precond(
            parsed["scaling"], parsed["rh"], parsed["precond"], shape[2], shape[3]
        )
        kwargs = dict(
            qbit=parsed["qbit"],
            blocks_num=1,
            block_size=parsed["tokens"],
            head_num=parsed["head_num"],
            head_dim=parsed["head_dim"],
            rh=parsed["rh"],
            asym=parsed["asym"],
            scaling_method=parsed["scaling"],
            output_dtype=dst.dtype,
            signs=signs,
            perm=perm,
            num_threads=self.num_threads,
        )
        if dst.dim() == 4 and dst.device.type == "cpu":
            self.native.kvweave_dequantize_chunk_into_4d(
                q,
                parsed["k_scales"],
                parsed["v_scales"],
                dst,
                shape[1],
                shape[2],
                shape[3],
                **kwargs,
            )
            return
        restored = self.native.kvweave_dequantize_chunk(
            q,
            parsed["k_scales"],
            parsed["v_scales"],
            shape[1],
            shape[2],
            shape[3],
            **kwargs,
        )
        if dst.dim() == 3:
            restored = restored.squeeze(1).permute(1, 0, 2)
        dst.copy_(restored.to(dtype=dst.dtype, device=dst.device))

    def _fused_quant_params(
        self,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        method: str,
        per_head_scales: bool,
    ) -> tuple[bytes, bool, bool, torch.Tensor | None, torch.Tensor | None]:
        """Header and RH/asym/precond parameters for quantizing a fused chunk.

        Returns:
            ``(header, rh, asym, signs, perm)``; ``signs``/``perm`` are ``None``
            unless preconditioning is on. Shared by every fused encode entry so
            their payloads cannot drift apart.
        """
        _, tokens, hidden = shape
        rh, asym = (False, False) if method == "per_tensor" else (self.rh, self.asym)
        precond = rh and self.precond
        flags = (
            (1 if rh else 0)
            | (2 if asym else 0)
            | (4 if precond else 0)
            | (8 if per_head_scales else 0)
        )
        header = self._config.MAGIC_QUANT_FUSED + struct.pack(
            ">BBBBB" + "i" * len(shape),
            self.qbit,
            self._config.DTYPE_TO_CODE.get(dtype, 0),
            flags,
            self._config.SCALING_TO_CODE.get(method, 1),
            len(shape),
            *shape,
        )
        signs = perm = None
        if precond:
            transform_size = self._fused_rh_transform_size(
                tokens, hidden, self._fused_head_dim(hidden), method, per_head_scales
            )
            signs, perm = self._config.rh_precond_tensors(transform_size)
        return header, rh, asym, signs, perm

    def serialize_fused_tensor(
        self,
        tensor: torch.Tensor,
        scaling_method: str | None = None,
        out: torch.Tensor | None = None,
    ) -> bytes | int:
        """Serialize a fused-K/V attention chunk (no leading K/V axis).

        Returns the payload as ``bytes``. If ``out`` (a contiguous ``uint8``
        CPU tensor) is given and the native module can write in place (the
        direct CPU kernel, or ``kvweave_serialize_chunk_state_into`` on XPU),
        the payload is written straight into ``out`` and its length returned.
        """
        if tensor.dim() != 3:
            raise ValueError(
                f"KVWeave fused-K/V tensor expects [L, T, H], got {tuple(tensor.shape)}"
            )
        work = self._kv_view(tensor.detach().to("cpu").contiguous())
        shape = tuple(int(dim) for dim in work.shape)
        layers, tokens, hidden = shape
        method = scaling_method or self.scaling_method
        head_num = self._fused_head_num(hidden)
        head_dim = self._fused_head_dim(hidden)
        per_head_scales = method != "per_tensor" and head_num > 1
        header, rh, asym, signs, perm = self._fused_quant_params(
            work.dtype, shape, method, per_head_scales
        )
        if self.fused_quant_uses_direct(hidden, method):
            result = self.native.kvweave_serialize_fused_chunk_direct(
                work,
                header,
                self.qbit,
                head_num,
                head_dim,
                rh=rh,
                asym=asym,
                scaling_method=method,
                signs=signs,
                perm=perm,
                num_threads=self.num_threads,
                out=out,
            )
            return result if out is not None else bytes(result)
        native_src = work
        native_layers = layers
        if per_head_scales:
            native_src = self.split_fused_attention_heads(work, head_num, head_dim)
            native_layers *= head_num
        native_kwargs = dict(
            qbit=self.qbit,
            blocks_num=1,
            block_size=tokens,
            head_num=head_num,
            head_dim=head_dim,
            num_layers=native_layers,
            rh=rh,
            asym=asym,
            scaling_method=method,
            signs=signs,
            perm=perm,
            num_threads=self.num_threads,
        )
        if out is not None and self.device == "xpu":
            return int(
                self.native.kvweave_serialize_chunk_state_into(
                    native_src.view(-1),
                    out.view(-1),
                    header,
                    KVWeaveCodecConfig.next_scale_id(),
                    **native_kwargs,
                )
            )
        return bytes(
            self.native.kvweave_serialize_chunk_state(
                native_src.view(-1),
                header,
                KVWeaveCodecConfig.next_scale_id(),
                **native_kwargs,
            )
        )

    @staticmethod
    def split_fused_attention_heads(
        source: torch.Tensor,
        head_num: int,
        head_dim: int,
        num_threads: int = 0,
    ) -> torch.Tensor:
        """Reorder fused ``[L,T,H*D]`` attention into contiguous head rows."""
        layers, tokens, hidden = map(int, source.shape)
        if hidden != head_num * head_dim:
            raise ValueError("fused hidden size does not match head layout")
        if kvweave_quant is not None:
            return kvweave_quant.kvweave_split_fused_heads(
                source.contiguous(), head_num, head_dim, num_threads, None
            )
        return (
            source.reshape(layers, tokens, head_num, head_dim)
            .permute(0, 2, 1, 3)
            .reshape(layers * head_num, tokens, head_dim)
            .contiguous()
        )

    def deserialize_fused_tensor(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Decode a fused-K/V payload and restore it into the destination tensor."""
        dst = self._kv_view(dst)
        payload = self._payload_u8(src)
        parsed = self._read_fused_header(payload)
        layers, tokens, hidden = parsed["shape"]
        q_dtype = torch.int8 if parsed["qbit"] <= 8 else torch.int16
        q = self._payload_tail(payload, parsed["q_offset"], q_dtype)
        signs = perm = None
        if parsed["precond"]:
            transform_size = self._fused_rh_transform_size(
                tokens,
                hidden,
                parsed["head_dim"],
                parsed["scaling"],
                parsed["per_head_scales"],
            )
            signs, perm = self._config.rh_precond_tensors(transform_size)
        native_layers = (
            layers * parsed["head_num"] if parsed["per_head_scales"] else layers
        )
        native_hidden = parsed["head_dim"] if parsed["per_head_scales"] else hidden
        if (
            parsed["per_head_scales"]
            and parsed["scaling"] == "per_channel"
            and parsed["qbit"] <= 8
            and dst.device.type == "cpu"
            and dst.is_contiguous()
            and self.supports_fused_direct_dequant()
        ):
            self.native.kvweave_dequantize_fused_chunk_direct(
                q,
                parsed["scales"],
                dst,
                layers,
                tokens,
                parsed["head_num"],
                parsed["head_dim"],
                parsed["qbit"],
                rh=parsed["rh"],
                asym=parsed["asym"],
                scaling_method=parsed["scaling"],
                signs=signs,
                perm=perm,
                num_threads=self.num_threads,
            )
            return
        native_kwargs = dict(
            qbit=parsed["qbit"],
            blocks_num=1,
            block_size=tokens,
            head_num=parsed["head_num"],
            head_dim=parsed["head_dim"],
            rh=parsed["rh"],
            asym=parsed["asym"],
            scaling_method=parsed["scaling"],
            output_dtype=dst.dtype,
            signs=signs,
            perm=perm,
            num_threads=self.num_threads,
        )
        if self.device == "xpu" and dst.device.type == "cpu" and dst.is_contiguous():
            # XPU module: DMA straight into dst, head merge done on the device.
            self.native.kvweave_dequantize_chunk_state_into(
                q,
                parsed["scales"],
                dst.view(-1),
                native_layers,
                tokens,
                native_hidden,
                merge_head_num=parsed["head_num"] if parsed["per_head_scales"] else 0,
                **native_kwargs,
            )
            return
        restored = self.native.kvweave_dequantize_chunk_state(
            q,
            parsed["scales"],
            native_layers,
            tokens,
            native_hidden,
            **native_kwargs,
        )
        if parsed["per_head_scales"]:
            if kvweave_quant is not None:
                kvweave_quant.kvweave_merge_fused_heads(
                    restored,
                    layers,
                    parsed["head_num"],
                    native_hidden,
                    self.num_threads,
                    dst.reshape(layers, tokens, hidden),
                )
                return
            else:
                restored = (
                    restored.reshape(layers, parsed["head_num"], tokens, native_hidden)
                    .permute(0, 2, 1, 3)
                    .reshape(layers, tokens, hidden)
                    .contiguous()
                )
        dst.copy_(restored.reshape(dst.shape).to(dtype=dst.dtype, device=dst.device))

    def _read_fused_header(self, payload: torch.Tensor) -> dict[str, Any]:
        """Read a fused-K/V header without copying the quantized payload."""
        raw = self._payload_buffer(payload)
        magic = bytes(raw[:4])
        if magic != self._config.MAGIC_QUANT_FUSED:
            raise ValueError(f"invalid KVWeave fused payload magic: {magic!r}")
        offset = 4
        qbit, dtype_code, flags, scaling_code, ndim = struct.unpack_from(
            ">BBBBB", raw, offset
        )
        offset += 5
        shape = tuple(struct.unpack_from(">" + "i" * ndim, raw, offset))
        offset += 4 * ndim
        if len(shape) != 3:
            raise ValueError(f"KVWeave fused payload expects 3-D shape, got {shape}")
        (scale_len,) = struct.unpack_from(">I", raw, offset)
        offset += 4
        scales = bytes(raw[offset : offset + scale_len])
        offset += scale_len
        method = {
            value: key for key, value in self._config.SCALING_TO_CODE.items()
        }.get(scaling_code, self.scaling_method)
        hidden = shape[2]
        return {
            "shape": shape,
            "qbit": qbit,
            "rh": bool(flags & 1),
            "asym": bool(flags & 2),
            "precond": bool(flags & 4),
            "per_head_scales": bool(flags & 8),
            "scaling": method,
            "dtype": self._config.CODE_TO_DTYPE.get(dtype_code, torch.float16),
            "scales": scales,
            "q_offset": offset,
            "head_num": self._fused_head_num(hidden),
            "head_dim": self._fused_head_dim(hidden),
        }

    @staticmethod
    def _fused_rh_transform_size(
        tokens: int,
        hidden: int,
        head_dim: int,
        method: str,
        per_head_scales: bool = False,
    ) -> int:
        if per_head_scales:
            return head_dim if method == "per_token" else tokens
        if method == "per_token":
            return hidden
        if method == "per_channel":
            if hidden % head_dim:
                raise ValueError("fused hidden size must divide by head_dim")
            return tokens * (hidden // head_dim)
        return tokens * hidden

    def _normalize(self, tensor: torch.Tensor) -> _KVShape:
        """Convert supported 3D/4D KV layouts to canonical 4D metadata."""
        if tensor.dim() == 3:
            tokens, kv_size, hidden = map(int, tensor.shape)
            if kv_size != 2:
                raise ValueError("KVWeave expects [T, 2, H]")
            return _KVShape(
                tensor.permute(1, 0, 2).unsqueeze(1).contiguous(),
                tuple(tensor.shape),
                1,
                tokens,
                hidden,
                3,
            )
        if tensor.dim() == 4:
            kv_size, layers, tokens, hidden = map(int, tensor.shape)
            if kv_size != 2:
                raise ValueError("KVWeave expects [2, L, T, H]")
            return _KVShape(tensor, tuple(tensor.shape), layers, tokens, hidden, 4)
        raise ValueError(f"unsupported KVWeave tensor shape: {tuple(tensor.shape)}")

    def _quant_header(
        self,
        shape: _KVShape,
        dtype: torch.dtype,
        method: str,
        rh: bool,
        asym: bool,
        precond: bool,
    ) -> bytes:
        return struct.pack(
            ">4sBBBBBBBB" + "i" * shape.original_ndim,
            self._config.MAGIC_QUANT,
            self.qbit,
            int(rh),
            int(asym),
            int(precond),
            shape.original_ndim,
            1,
            self._config.SCALING_TO_CODE.get(method, 1),
            self._config.DTYPE_TO_CODE.get(dtype, 0),
            *shape.header_shape,
        )

    def _read_header(self, payload: torch.Tensor) -> dict[str, Any]:
        """Read the attention payload header without copying q_data."""
        raw = self._payload_buffer(payload)
        magic = bytes(raw[:4])
        offset = 4
        if magic != self._config.MAGIC_QUANT:
            raise ValueError(f"invalid KVWeave payload magic: {magic!r}")
        qbit, rh, asym, precond, ndim, _, scaling_code, dtype_code = struct.unpack_from(
            ">BBBBBBBB", raw, offset
        )
        offset += 8
        shape = tuple(struct.unpack_from(">" + "i" * ndim, raw, offset))
        offset += 4 * ndim
        shape4d = (shape[1], 1, shape[0], shape[2]) if ndim == 3 else shape
        scales = []
        for _ in range(2):
            size = struct.unpack_from(">I", raw, offset)[0]
            offset += 4
            scales.append(bytes(raw[offset : offset + size]))
            offset += size
        method = {
            value: key for key, value in self._config.SCALING_TO_CODE.items()
        }.get(scaling_code, self.scaling_method)
        return {
            "shape4d": shape4d,
            "qbit": qbit,
            "rh": bool(rh),
            "asym": bool(asym),
            "precond": bool(precond),
            "scaling": method,
            "dtype": self._config.CODE_TO_DTYPE.get(dtype_code, torch.float16),
            "k_scales": scales[0],
            "v_scales": scales[1],
            "q_offset": offset,
            "tokens": shape4d[2],
            "head_num": self._head_num(shape4d[3]),
            "head_dim": self._head_dim(shape4d[3]),
        }

    def _attention_precond(
        self, method: str, rh: bool, precond: bool, tokens: int, hidden: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the (signs, perm) P/D pair for the attention path, if any.

        The transform length is the quantization chunk size of one layer plane
        (same rule as the fused-K/V path). Serialize and deserialize must derive
        it identically; the header's precond flag is only set when this applies.
        """
        if not (rh and precond):
            return None, None
        size = self._fused_rh_transform_size(
            tokens, hidden, self._head_dim(hidden), method
        )
        return self._config.rh_precond_tensors(size)

    # The codec never needs the engine's head geometry. A split-K/V plane is
    # quantized as one unit spanning the whole hidden axis (per_token: one scale
    # per token; per_tensor: one scale), and a fused chunk as two halves
    # (per_channel: one scale per hidden column; per_tensor: one scale). Neither
    # grouping depends on how ``hidden`` splits into heads.

    def _head_num(self, hidden: int) -> int:
        return 1

    def _head_dim(self, hidden: int) -> int:
        return hidden

    def _fused_head_dim(self, hidden: int) -> int:
        """Width of one K or V plane in a fused ``[K|V]`` content axis."""
        return hidden // self._fused_head_num(hidden)

    def _fused_head_num(self, hidden: int) -> int:
        """Native units of a fused chunk: its two halves (one when ``hidden`` is odd).

        Only per_channel and per_tensor are used on fused chunks, and their
        scales do not depend on the unit width; two units keep the chunk on the
        direct (no host head split) kernels.
        """
        return 2 if hidden % 2 == 0 else 1

    def _fused_scale_count(
        self, tokens: int, hidden: int, layers: int, method: str
    ) -> int:
        base = (
            tokens
            if method == "per_token"
            else self._fused_head_dim(hidden)
            if method == "per_channel"
            else 1
        )
        if method != "per_tensor":
            layers *= self._fused_head_num(hidden)
        return max(1, base * layers if layers > 1 else base)

    def _scale_count(self, tokens: int, hidden: int, layers: int, method: str) -> int:
        base = (
            tokens
            if method == "per_token"
            else self._head_dim(hidden)
            if method == "per_channel"
            else 1
        )
        return max(1, base * layers if layers > 1 else base)

    def encode_chunk(
        self,
        raw_chunk: torch.Tensor,
        attention_plane_layout: AttentionPlaneLayout | None = None,
    ) -> bytes:
        """Encode one gathered raw attention chunk into its quantized payload.

        Chunks are quantized via ``self.serialize_fused_tensor`` when
        ``attention_plane_layout`` is ``AttentionPlaneLayout.FUSED_KV`` (K/V
        packed into one tensor, no leading K/V axis), or
        ``self.serialize_tensor`` otherwise.
        """
        if attention_plane_layout == AttentionPlaneLayout.FUSED_KV:
            return cast(bytes, self.serialize_fused_tensor(raw_chunk))
        return cast(bytes, self.serialize_tensor(raw_chunk))

    def encode_chunk_into(
        self,
        raw_chunk: torch.Tensor,
        destination: torch.Tensor,
        attention_plane_layout: AttentionPlaneLayout | None = None,
    ) -> int:
        """Encode a chunk and copy its payload directly into ``destination``.

        Fused-K/V chunks (direct CPU kernel or XPU) and split chunks on the CPU
        are written straight into ``destination`` with no intermediate
        ``bytes`` object; split chunks on the XPU are built as ``bytes`` and
        copied in. Returns the number of payload bytes written.
        """
        if destination.dtype != torch.uint8 or not destination.is_contiguous():
            raise ValueError("destination must be a contiguous torch.uint8 tensor")
        if attention_plane_layout == AttentionPlaneLayout.FUSED_KV:
            written = self.serialize_fused_tensor(raw_chunk, out=destination.view(-1))
            if isinstance(written, int):
                return written
            return self._copy_payload_into(written, destination)
        if self.device == "cpu":
            # The CPU module writes split chunks straight into destination.
            return int(self.serialize_tensor(raw_chunk, out=destination.view(-1)))
        payload = self.encode_chunk(raw_chunk, attention_plane_layout)
        return self._copy_payload_into(payload, destination)

    @staticmethod
    def _copy_payload_into(payload: _BytesLike, destination: torch.Tensor) -> int:
        if len(payload) > destination.numel():
            raise ValueError(
                f"encoded payload ({len(payload)} bytes) exceeds destination "
                f"capacity ({destination.numel()} bytes)"
            )
        destination.view(-1)[: len(payload)].copy_(
            torch.frombuffer(memoryview(payload), dtype=torch.uint8)
        )
        return len(payload)

    def decode_chunk(
        self,
        raw_shape: torch.Size,
        raw_dtype: torch.dtype,
        chunk: torch.Tensor,
        attention_plane_layout: AttentionPlaneLayout | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode one retrieved quantized byte chunk back to its raw shape.

        Mirrors :meth:`encode_chunk`. When ``out`` is supplied, the decoded raw
        chunk is written there; otherwise a fresh tensor is allocated.
        """
        destination = (
            out if out is not None else torch.empty(raw_shape, dtype=raw_dtype)
        )
        if destination.shape != raw_shape or destination.dtype != raw_dtype:
            raise ValueError("out must match raw_shape and raw_dtype")
        if attention_plane_layout == AttentionPlaneLayout.FUSED_KV:
            self.deserialize_fused_tensor(chunk, destination)
        else:
            self.deserialize_tensor(chunk, destination)
        return destination


KVWeaveCodec = _KVWeaveCodec
