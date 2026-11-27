# SPDX-License-Identifier: Apache-2.0
"""GHA L2 serde: quantize L1 chunks while they are written to L2.

The L1 pool keeps full-precision KV. When the store controller moves a chunk
to an L2 adapter wrapped by :class:`SerdeL2AdapterWrapper`, this serde encodes
it with the GHA codec (4/8-bit, on CPU or XPU); the load path decodes it
back into the L1 destination chunk. Because only L2 is lossy, the double-lossy
guard of L1 quantization does not apply.

What a chunk *is* depends on the object group it belongs to, so every key is
routed through a per-group :class:`GHAGroupSpec`:

* ``attention`` + ``split_kv``: ``[2, L, T, H]``, K and V as separate planes.
* ``attention`` + ``fused_kv``: ``[L, T, H]``, K and V packed on the last axis.
* ``attention`` + ``auto`` (the default for unconfigured groups): split or
  fused, read from the chunk's shape. MLA / indexer-cache groups share the
  fused shape and must be declared explicitly.
* anything else (MLA, unclassified, or not worth quantizing): the bytes are
  copied through unchanged.

Group facts come from the serde's JSON kwargs (``"groups"`` /
``"default_group"``).

The scaling granularity never depends on the engine's head geometry, so no
head counts are configured: with ``per_tensor`` false (the default) split
chunks are quantized per token and fused chunks per channel; with
``per_tensor`` true both use a single scale per plane.

Example::

    {"type": "fs", "base_path": "/cache",
     "serde": {"type": "gha", "qbit": 4, "device": "xpu"}}
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
import threading

# Third Party
import torch

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import (
    Deserializer,
    SerdeProcessor,
    Serializer,
)
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.distributed.serde.gha.gha_config import (
    AttentionPlaneLayout,
    GHACodecConfig,
)
from lmcache.v1.distributed.serde.gha.gha_serde import GHACodec
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

_SUPPORTED_DTYPES = frozenset(GHACodecConfig.DTYPE_TO_CODE) | {torch.uint8}

# Length prefix of each part of a multi-part chunk (int32).
_FRAME_BYTES = 4

# Every key the serde config accepts (``type`` is consumed by the factory).
_CONFIG_KEYS = frozenset(
    {
        "qbit",
        "per_tensor",
        "rh",
        "asym",
        "precond",
        "device",
        "num_threads",
        "kv_fp8_dtype",
        "groups",
        "default_group",
        "max_workers",
    }
)


# ============================================================================
# Per-group metadata
# ============================================================================


@dataclass(frozen=True)
class GHAGroupSpec:
    """What the codec needs to know about one object group.

    Attributes:
        cache_category: ``"attention"`` or ``"unknown"``
            (``"unknown"`` groups are copied through unquantized).
        attention_plane_layout: K/V plane layout of an attention group.
    """

    cache_category: str = "unknown"
    attention_plane_layout: AttentionPlaneLayout = AttentionPlaneLayout.UNKNOWN

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GHAGroupSpec":
        """Build a spec from a JSON-config dict.

        Keys: ``cache_category`` and ``attention_plane_layout``.

        Raises:
            ValueError: On any other key.
        """
        unknown = set(raw) - {"cache_category", "attention_plane_layout"}
        if unknown:
            raise ValueError(
                f"gha serde: unsupported group keys {sorted(unknown)}; a group "
                "takes only cache_category and attention_plane_layout"
            )
        return cls(
            cache_category=str(raw.get("cache_category", "unknown")),
            attention_plane_layout=AttentionPlaneLayout(
                str(raw.get("attention_plane_layout", "unknown"))
            ),
        )


# ============================================================================
# Per-(group, shape) plan
# ============================================================================


@dataclass(frozen=True)
class _Plan:
    """How one chunk of a given group/shape/dtype is (de)serialized."""

    mode: str  # "raw" | "split" | "fused"
    size: int  # exact upper bound of the serialized bytes
    codec: GHACodec | None = None
    spec: GHAGroupSpec | None = None


def _raw_nbytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n * torch.empty((), dtype=dtype).element_size()


def _transform_length_ok(length: int) -> bool:
    return length > 0 and not length & (length - 1)


def _plane_layout(mode: str, spec: GHAGroupSpec) -> AttentionPlaneLayout | None:
    return spec.attention_plane_layout if mode in ("split", "fused") else None


def encode_chunk_into_with_plan(
    plan: "_Plan", raw: torch.Tensor, destination: torch.Tensor
) -> int:
    """Encode ``raw`` per ``plan`` straight into ``destination``.

    Args:
        plan: A quantizing plan (``split`` or ``fused``).
        raw: The CPU chunk tensor.
        destination: Flat contiguous ``uint8`` buffer of at least ``plan.size``.

    Returns:
        Bytes written into ``destination``.
    """
    assert plan.codec is not None and plan.spec is not None
    return plan.codec.encode_chunk_into(
        raw,
        destination,
        attention_plane_layout=_plane_layout(plan.mode, plan.spec),
    )


class _PlanResolver:
    """Resolves and caches the :class:`_Plan` of every (group, shape, dtype)."""

    def __init__(self, kwargs: Mapping[str, Any]) -> None:
        self._codec_kwargs = self._build_codec_kwargs(kwargs)
        # Static groups from the JSON config: ``object_group_id -> spec`` or
        # ``-> [spec per part]``.
        self._static: dict[int, tuple[GHAGroupSpec, ...]] = {
            int(group_id): tuple(
                GHAGroupSpec.from_dict(part)
                for part in (raw if isinstance(raw, list) else [raw])
            )
            for group_id, raw in dict(kwargs.get("groups", {})).items()
        }
        # Spec of any group the config does not describe: attention whose
        # split / fused layout is read from each chunk's shape.
        default_raw = dict(kwargs.get("default_group", {}))
        default_raw.setdefault("cache_category", "attention")
        default_raw.setdefault("attention_plane_layout", "auto")
        self._default_spec = GHAGroupSpec.from_dict(default_raw)
        self._per_tensor = bool(kwargs.get("per_tensor", False))
        self._codecs: dict[tuple[str, bool], GHACodec] = {}
        self._plans: dict[tuple, _Plan] = {}
        self._lock = threading.Lock()
        self._warned: set[tuple] = set()

    @staticmethod
    def _build_codec_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(kwargs) - _CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"gha serde: unknown config keys {sorted(unknown)}; "
                f"supported keys are {sorted(_CONFIG_KEYS)}"
            )
        qbit = int(kwargs.get("qbit", 4))
        if qbit not in (4, 8):
            raise ValueError(f"gha serde: qbit={qbit} is not one of [4, 8]")
        fp8 = kwargs.get("kv_fp8_dtype")
        if isinstance(fp8, str):
            fp8 = getattr(torch, fp8, None)
            if fp8 not in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise ValueError(
                    f"gha serde: kv_fp8_dtype={kwargs['kv_fp8_dtype']!r} "
                    "is not float8_e4m3fn / float8_e5m2"
                )
        return dict(
            qbit=qbit,
            rh=bool(kwargs.get("rh", True)),
            asym=bool(kwargs.get("asym", True)),
            precond=bool(kwargs.get("precond", True)),
            num_threads=int(kwargs.get("num_threads", 4)),
            device=str(kwargs.get("device", "cpu")),
            kv_fp8_dtype=fp8,
        )

    # ----- codec cache -----

    def _scaling_method(self, mode: str) -> str:
        """Scaling of a ``split`` / ``fused`` chunk: per_tensor when configured,
        else per_token (split) or per_channel (fused) -- the granularities that
        do not depend on the head geometry."""
        if self._per_tensor:
            return "per_tensor"
        return "per_token" if mode == "split" else "per_channel"

    def _codec(self, mode: str, rh: bool) -> GHACodec:
        ident = (mode, rh)
        with self._lock:
            codec = self._codecs.get(ident)
            if codec is None:
                kwargs = dict(self._codec_kwargs)
                kwargs.update(scaling_method=self._scaling_method(mode), rh=rh)
                codec = GHACodec(kwargs)
                self._codecs[ident] = codec
            return codec

    # ----- resolution -----

    def spec_for(self, key: ObjectKey | None, part: int) -> GHAGroupSpec:
        """The spec of part ``part`` of ``key``'s object group.

        Falls back to the serde-level default when the group is unknown.
        """
        if key is None:
            return self._default_spec
        static = self._static.get(key.object_group_id)
        if static is not None and part < len(static):
            return static[part]
        return self._default_spec

    def plan(
        self,
        key: ObjectKey | None,
        part: int,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> _Plan:
        """Return the (cached) plan for one part of ``shape``/``dtype``."""
        spec = self.spec_for(key, part)
        ident = (spec, tuple(int(d) for d in shape), dtype)
        with self._lock:
            plan = self._plans.get(ident)
        if plan is None:
            plan = self._build_plan(spec, ident[1], dtype)
            with self._lock:
                self._plans[ident] = plan
        return plan

    def _raw(self, shape: tuple[int, ...], dtype: torch.dtype, why: str) -> _Plan:
        ident = (why, shape, dtype)
        if ident not in self._warned:
            self._warned.add(ident)
            logger.info(
                "gha serde: copying shape=%s dtype=%s through unquantized (%s)",
                shape,
                dtype,
                why,
            )
        return _Plan(mode="raw", size=_raw_nbytes(shape, dtype))

    def _build_plan(
        self, spec: GHAGroupSpec, shape: tuple[int, ...], dtype: torch.dtype
    ) -> _Plan:
        if dtype not in _SUPPORTED_DTYPES:
            return self._raw(shape, dtype, f"unsupported dtype {dtype}")
        if spec.cache_category == "attention":
            return self._build_attention_plan(spec, shape, dtype)
        return self._raw(shape, dtype, f"category {spec.cache_category}")

    def _build_attention_plan(
        self, spec: GHAGroupSpec, shape: tuple[int, ...], dtype: torch.dtype
    ) -> _Plan:
        layout = spec.attention_plane_layout
        if layout == AttentionPlaneLayout.AUTO:
            if len(shape) == 4 and shape[0] == 2:
                layout = AttentionPlaneLayout.SPLIT_KV
            elif len(shape) == 3:
                layout = AttentionPlaneLayout.FUSED_KV
            else:
                return self._raw(
                    shape, dtype, "shape is neither split [2,L,T,H] nor fused [L,T,H]"
                )
            # The plan's spec drives the codec's plane layout, so it must carry
            # the resolved layout, not AUTO.
            spec = replace(spec, attention_plane_layout=layout)
        if layout == AttentionPlaneLayout.SPLIT_KV:
            if len(shape) != 4 or shape[0] != 2:
                return self._raw(shape, dtype, "shape is not split-K/V [2,L,T,H]")
            mode = "split"
        elif layout == AttentionPlaneLayout.FUSED_KV:
            if len(shape) != 3:
                return self._raw(shape, dtype, "shape is not fused-K/V [L,T,H]")
            mode = "fused"
        else:
            return self._raw(shape, dtype, f"attention layout {layout.value}")
        codec = self._codec(mode, self._codec_kwargs["rh"])
        if dtype == torch.uint8 and not (
            mode == "fused" and codec.supports_fp8_attention()
        ):
            return self._raw(
                shape, dtype, "uint8 (fp8) KV needs fused layout + kv_fp8_dtype"
            )
        desc = MemoryLayoutDesc(shapes=[torch.Size(shape)], dtypes=[dtype])
        try:
            if mode == "split":
                size = codec.estimate_serialized_size(desc)
            else:
                size = codec.estimate_fused_serialized_size(desc)
        except ValueError as exc:
            return self._raw(shape, dtype, f"size estimate rejected shape: {exc}")
        if size >= _raw_nbytes(shape, dtype):
            return self._raw(shape, dtype, "quantized chunk is not smaller")
        codec = self._probe_rh(codec, spec, mode, shape, dtype)
        return _Plan(mode=mode, size=size, codec=codec, spec=spec)

    def _probe_rh(
        self,
        codec: GHACodec,
        spec: GHAGroupSpec,
        mode: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> GHACodec:
        """Return a codec whose RH setting this chunk shape can actually run.

        The native RH kernel needs a power-of-two transform length, which
        depends on shape and scaling method. Encode a zero chunk once; when the
        kernel rejects it, fall back to the same codec without RH (the payload
        size does not depend on RH, and the header records the choice).
        """
        if not codec.rh or codec.scaling_method == "per_tensor":
            return codec
        probe = torch.zeros(shape, dtype=dtype)
        try:
            codec.encode_chunk(
                probe,
                attention_plane_layout=_plane_layout(mode, spec),
            )
            return codec
        except ValueError as exc:
            if "power of 2" not in str(exc):
                raise
        logger.info(
            "gha serde: shape %s cannot run RH (transform length is not a "
            "power of 2); using no-RH codec for it",
            shape,
        )
        return self._codec(mode, False)


# ============================================================================
# Serializer / Deserializer
# ============================================================================


def _flat_u8(obj: MemoryObj) -> torch.Tensor:
    tensor = obj.tensor
    if tensor is None:
        raise ValueError("gha serde requires MemoryObjs that hold a tensor")
    return tensor.view(torch.uint8).view(-1)


def _part_tensor(obj: MemoryObj, part: int) -> torch.Tensor:
    tensor = obj.get_tensor(part)
    if tensor is None:
        raise ValueError("gha serde requires MemoryObjs that hold a tensor")
    return tensor


class GHASerializer(Serializer):
    """Quantize an L1 chunk into the L2 byte buffer (see module docstring).

    A chunk with several parts (one tensor per kernel group) is stored as
    ``[u32 length | payload]`` per part; a single-part chunk is the bare
    payload.
    """

    def __init__(self, resolver: _PlanResolver) -> None:
        self._resolver = resolver

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Size of the chunk under the serde's default group spec."""
        return self.estimate_serialized_size_for_key(layout_desc, None)

    def estimate_serialized_size_for_key(
        self, layout_desc: MemoryLayoutDesc, key: ObjectKey | None
    ) -> int:
        """Exact size of the serialized chunk of ``key``'s object group."""
        total = 0
        parts = list(zip(layout_desc.shapes, layout_desc.dtypes, strict=True))
        for part, (shape, dtype) in enumerate(parts):
            total += self._resolver.plan(key, part, shape, dtype).size
        return total + (_FRAME_BYTES * len(parts) if len(parts) > 1 else 0)

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Encode every part of ``src`` into ``dst``; return the bytes written."""
        out = _flat_u8(dst)
        num_parts = len(src.get_shapes())
        offset = 0
        for part in range(num_parts):
            raw = _part_tensor(src, part)
            plan = self._resolver.plan(key, part, raw.shape, raw.dtype)
            if num_parts > 1:
                offset += _FRAME_BYTES
            written = self._serialize_part(plan, raw, out[offset:])
            if num_parts > 1:
                out[offset - _FRAME_BYTES : offset].copy_(
                    torch.tensor([written], dtype=torch.int32).view(torch.uint8)
                )
            offset += written
        return offset

    @staticmethod
    def _serialize_part(plan: _Plan, raw: torch.Tensor, out: torch.Tensor) -> int:
        if plan.mode == "raw":
            n = raw.numel() * raw.element_size()
            out[:n].copy_(raw.contiguous().view(torch.uint8).view(-1))
            return n
        raw_cpu = raw if raw.device.type == "cpu" else raw.cpu()
        written = encode_chunk_into_with_plan(plan, raw_cpu, out[: plan.size])
        if written > plan.size:
            raise RuntimeError(
                f"gha payload ({written} B) exceeds its estimate ({plan.size} B)"
            )
        return int(written)


class GHADeserializer(Deserializer):
    """Decode an L2 byte buffer back into the L1 chunk."""

    def __init__(self, resolver: _PlanResolver) -> None:
        self._resolver = resolver

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Decode every part of the stored buffer into ``dst``'s tensors."""
        payload = _flat_u8(src)
        num_parts = len(dst.get_shapes())
        offset = 0
        for part in range(num_parts):
            out = _part_tensor(dst, part)
            plan = self._resolver.plan(key, part, out.shape, out.dtype)
            if num_parts > 1:
                length = int(payload[offset : offset + _FRAME_BYTES].view(torch.int32))
                offset += _FRAME_BYTES
            else:
                length = payload.numel()
            self._deserialize_part(plan, payload[offset : offset + length], out)
            offset += length

    @staticmethod
    def _deserialize_part(
        plan: _Plan, payload: torch.Tensor, out: torch.Tensor
    ) -> None:
        if plan.mode == "raw":
            n = out.numel() * out.element_size()
            out.view(torch.uint8).view(-1).copy_(payload[:n])
            return
        assert plan.codec is not None and plan.spec is not None
        spec = plan.spec
        plane = spec.attention_plane_layout if plan.mode in ("split", "fused") else None
        plan.codec.decode_chunk(
            out.shape,
            out.dtype,
            payload,
            attention_plane_layout=plane,
            out=out,
        )


def _create_gha_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    resolver = _PlanResolver(kwargs)
    return AsyncSerdeProcessor(
        GHASerializer(resolver),
        GHADeserializer(resolver),
        max_workers=int(kwargs.get("max_workers", 1)),  # type: ignore[call-overload]
    )


register_serde_factory("gha", _create_gha_serde)
