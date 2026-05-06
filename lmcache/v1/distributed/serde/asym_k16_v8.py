# SPDX-License-Identifier: Apache-2.0
"""
Asymmetric K16/V8 multi-output serde.

Concrete :class:`MultiSerializer` / :class:`MultiDeserializer`
pair that bridges :class:`AsymK16V8Codec` (in
``lmcache/v1/kv_codec``) into the tuple-shaped serde contract
from ``multi.py``.

**Storage-only-dequant mode.** Group of size 2 on both endpoints:

* serialize input ``src = (K, V)`` -- both at the model's native
  dtype (fp16 / bf16). The codec quantizes V to FP8 internally
  and writes a single byte buffer containing K bytes, V FP8
  bytes, and the codec header with V's scale.
* deserialize output ``dst = (K_out, V_out)`` -- both at the
  model's native dtype, with V dequantized from the stored FP8.

Compresses the bytes that ship to L2 storage; the consumer-visible
view on read is unchanged.
"""

# Future
from __future__ import annotations

# Standard
import math

# Third Party
import torch

# First Party
from lmcache.v1.distributed.serde.multi import (
    LayoutDescGroup,
    MemoryObjGroup,
    MultiDeserializer,
    MultiSerializer,
    validate_group_size,
)
from lmcache.v1.kv_codec import AsymK16V8Codec, ScaleScope
from lmcache.v1.memory_management import MemoryObj

_GROUP_SIZE_STORAGE_ONLY = 2  # (K, V) on both sides for this mode.


def _v_fp8_max_for_dtype(fp8_dtype: torch.dtype) -> float:
    return float(torch.finfo(fp8_dtype).max)


class AsymK16V8MultiSerializer(MultiSerializer):
    """Encode ``(K, V)`` into a single asymmetric K16/V8 byte blob.

    Slot semantics:

    * ``slot 0 = K`` (required, fp16 or bf16, native model dtype).
    * ``slot 1 = V`` (required, fp16 or bf16, native model dtype).

    ``None`` is not admitted in either slot for this mode -- both K
    and V must be supplied.

    The codec produces a self-describing blob: header (dtype tags,
    scale scope, scale tensor, optional shape metadata) followed by
    K bytes (native dtype), V bytes (FP8 e4m3), and scale bytes.
    The exact byte layout is owned by ``serialize_header`` /
    ``deserialize_header`` in ``encoded_kv``; this serializer is
    just glue so the codec is reachable through the
    :class:`MultiSerializer` interface.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_scope: ScaleScope = ScaleScope.PER_TENSOR,
        scale_dtype: torch.dtype = torch.float32,
    ) -> None:
        self._codec = AsymK16V8Codec(
            fp8_dtype=fp8_dtype,
            scale_scope=scale_scope,
            scale_dtype=scale_dtype,
        )
        self._fp8_dtype = fp8_dtype

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE_STORAGE_ONLY

    def serialize(self, src: MemoryObjGroup, dst: MemoryObj) -> int:
        validate_group_size(src, _GROUP_SIZE_STORAGE_ONLY, role="src")
        k_obj, v_obj = src
        if k_obj is None or v_obj is None:
            raise ValueError(
                "AsymK16V8MultiSerializer (storage-only-dequant): both K "
                "and V must be provided"
            )
        if k_obj.tensor is None or v_obj.tensor is None:
            raise ValueError(
                "AsymK16V8MultiSerializer: src MemoryObjs must have tensors set"
            )
        if dst.tensor is None:
            raise ValueError("AsymK16V8MultiSerializer: dst.tensor is None")

        enc = self._codec.encode(k_obj.tensor, v_obj.tensor)
        blob = self._codec.to_bytes(enc)
        n = len(blob)
        if dst.tensor.numel() < n:
            raise ValueError(
                f"AsymK16V8MultiSerializer: dst capacity {dst.tensor.numel()} "
                f"below required {n}"
            )
        dst_view = dst.tensor.view(torch.uint8)
        dst_view[:n].copy_(torch.frombuffer(blob, dtype=torch.uint8))
        return n

    def estimate_serialized_size(
        self,
        layout_descs: LayoutDescGroup,
    ) -> int:
        validate_group_size(layout_descs, _GROUP_SIZE_STORAGE_ONLY, role="layout")
        k_layout, v_layout = layout_descs
        if k_layout is None or v_layout is None:
            raise ValueError(
                "AsymK16V8MultiSerializer.estimate_serialized_size "
                "(storage-only-dequant): both K and V layouts required"
            )

        # Bytes accounting:
        #   K: sum(numel(shape)) * itemsize(K_dtype)
        #   V: sum(numel(shape)) * 1          (FP8 = 1 byte / elem)
        #   scales: scope-dependent; only PER_TENSOR and EXTERNAL are
        #           bounded by layout alone. PER_LAYER_HEAD and
        #           PER_PAGE_HEAD need head/page metadata that the
        #           layout descriptor does not carry, so they are
        #           rejected here -- callers with those scopes must
        #           extend the API rather than risk a silent undersize.
        #   header: <= 1 KB allowance for dtype tags + hashes + shape
        scope = self._codec.scale_scope
        scale_dtype = self._codec.scale_dtype
        if scope == ScaleScope.PER_TENSOR:
            scales_bytes = scale_dtype.itemsize
        elif scope == ScaleScope.EXTERNAL:
            scales_bytes = 0
        else:
            raise ValueError(
                f"AsymK16V8MultiSerializer.estimate_serialized_size: "
                f"scale_scope {scope} requires per-head/per-page metadata "
                f"that MemoryLayoutDesc does not carry; only PER_TENSOR "
                f"and EXTERNAL are supported here"
            )
        k_bytes = sum(
            math.prod(s) * d.itemsize
            for s, d in zip(k_layout.shapes, k_layout.dtypes, strict=True)
        )
        v_bytes = sum(math.prod(s) for s in v_layout.shapes)
        header_allowance = 1024
        return header_allowance + k_bytes + v_bytes + scales_bytes


class AsymK16V8MultiDeserializer(MultiDeserializer):
    """Decode an asymmetric K16/V8 byte blob into ``(K_out, V_out)``.

    Slot semantics (storage-only-dequant mode):

    * ``slot 0 = K_out`` — caller-provided MemoryObj at the model's
      native dtype, populated bit-exact from the blob.
    * ``slot 1 = V_out`` — caller-provided MemoryObj at the model's
      native dtype, populated by dequantizing the stored FP8 V.

    A ``None`` slot is treated as a deliberate skip (the corresponding
    output is left untouched), matching the contract documented in
    ``multi.py``. This provides the V-only-read shape: pass
    ``(K_out, None)`` to load K only; pass ``(None, V_out)`` to load
    V only.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_scope: ScaleScope = ScaleScope.PER_TENSOR,
        scale_dtype: torch.dtype = torch.float32,
    ) -> None:
        self._codec = AsymK16V8Codec(
            fp8_dtype=fp8_dtype,
            scale_scope=scale_scope,
            scale_dtype=scale_dtype,
        )

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE_STORAGE_ONLY

    def deserialize(self, src: MemoryObj, dst: MemoryObjGroup) -> None:
        validate_group_size(dst, _GROUP_SIZE_STORAGE_ONLY, role="dst")
        if src.tensor is None:
            raise ValueError("AsymK16V8MultiDeserializer: src.tensor is None")

        # Pull blob bytes out of the source uint8 buffer.
        src_view = src.tensor.view(torch.uint8).contiguous()
        blob = src_view.numpy().tobytes()

        # Decode through the codec; ``decode`` returns 1-D flat tensors
        # for K and V, plus the scale tensor.  We reshape into the
        # caller-provided dst shapes.
        k_obj, v_obj = dst
        if k_obj is None and v_obj is None:
            # Both slots skipped — nothing to do.
            return

        # Pick the V output dtype from whichever dst slot is set.  If
        # only K is requested we still decode V to the same dtype so
        # the codec contract (no silent fp16 fallback) stays explicit;
        # the V tensor is then discarded.
        if v_obj is not None and v_obj.tensor is not None:
            target_v_dtype = v_obj.tensor.dtype
        elif k_obj is not None and k_obj.tensor is not None:
            target_v_dtype = k_obj.tensor.dtype
        else:
            raise ValueError(
                "AsymK16V8MultiDeserializer: at least one non-None dst "
                "slot must have a tensor set"
            )

        enc = self._codec.from_bytes(blob)
        # Push the K dtype conversion into the codec so the serde
        # doesn't need a post-decode cast.
        if k_obj is not None and k_obj.tensor is not None:
            out_k_dtype = k_obj.tensor.dtype
        else:
            out_k_dtype = None
        k_flat, v_dq_flat, _scales = self._codec.decode(
            enc, out_k_dtype=out_k_dtype, out_v_dtype=target_v_dtype
        )

        if k_obj is not None:
            if k_obj.tensor is None:
                raise ValueError(
                    "AsymK16V8MultiDeserializer: non-None K dst slot "
                    "must have a tensor set"
                )
            target_shape = k_obj.tensor.shape
            if k_flat.numel() != k_obj.tensor.numel():
                raise ValueError(
                    f"AsymK16V8MultiDeserializer: decoded K has "
                    f"{k_flat.numel()} elements, dst K shape "
                    f"{tuple(target_shape)} expects "
                    f"{k_obj.tensor.numel()}"
                )
            k_obj.tensor.copy_(k_flat.reshape(target_shape))

        if v_obj is not None:
            if v_obj.tensor is None:
                raise ValueError(
                    "AsymK16V8MultiDeserializer: non-None V dst slot "
                    "must have a tensor set"
                )
            target_shape = v_obj.tensor.shape
            if v_dq_flat.numel() != v_obj.tensor.numel():
                raise ValueError(
                    f"AsymK16V8MultiDeserializer: decoded V has "
                    f"{v_dq_flat.numel()} elements, dst V shape "
                    f"{tuple(target_shape)} expects "
                    f"{v_obj.tensor.numel()}"
                )
            v_obj.tensor.copy_(v_dq_flat.reshape(target_shape))
