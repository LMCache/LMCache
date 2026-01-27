# SPDX-License-Identifier: Apache-2.0
"""KVCache format descriptor for engine-agnostic KV layout definitions.

Layout is expressed as a single routing key (KVCacheLayout) to avoid a Cartesian
product of partially-valid combinations. Family/canonical/separation are derived
from that routing key and documented below.

Separation meaning:
- packed: K and V live in a single tensor/buffer (e.g., a 2LTD-style [2, L, ...]
  layout) and are copied together.
- separated: K and V live in separate tensors/buffers and are copied separately.

Canonical meaning:
- KV_2LTD: dense MHA layout with K/V packed along a leading dimension.
- KV_MLA_FMT: MLA latent layout.

Examples:
- vLLM dense MHA (paged attention): kv_shape=(L, 2, block, H, D) maps to
  layout=MHA_DENSE_PACKED, block_size=block, hidden_dim=H*D.
- SGLang dense MHA: same mapping as vLLM for integration mode.
- MLA models (use_mla=True): layout=MLA_LATENT_PACKED with the same block_size.
"""

# Future
from __future__ import annotations

# Standard
from typing import Literal, Optional, Protocol, Sequence, TypedDict
import re

# Third Party
import msgspec

# Public schema/version identifier for the format descriptor.
KV_FORMAT_SCHEMA_VERSION = 1

Family = Literal["MHA_DENSE", "MLA_LATENT"]
Canonical = Literal["KV_2LTD", "KV_MLA_FMT"]
Separation = Literal["packed", "separated"]
KVCacheLayout = Literal[
    "MHA_DENSE_PACKED",
    "MHA_DENSE_SEPARATED",
    "MLA_LATENT_PACKED",
    "MLA_LATENT_SEPARATED",
]
Addressing = Literal["gpu_block_ids"]


class L0LayoutSpec(msgspec.Struct, frozen=True):
    addressing: Addressing
    block_size: int
    pointer_order: Optional[str] = None


class KVLayerGroupSpec(msgspec.Struct, frozen=True):
    start_layer: int
    num_layers: int
    dtype: str
    hidden_dim: int


class KVCacheFormat(msgspec.Struct, frozen=True):
    layout: KVCacheLayout
    l0: L0LayoutSpec
    layer_groups: list[KVLayerGroupSpec]
    format_id: str
    schema_version: int = KV_FORMAT_SCHEMA_VERSION

    @property
    def family(self) -> Family:
        return _LAYOUT_DEFS[self.layout]["family"]

    @property
    def canonical(self) -> Canonical:
        return _LAYOUT_DEFS[self.layout]["canonical"]

    @property
    def separation(self) -> Optional[Separation]:
        return _LAYOUT_DEFS[self.layout]["separation"]

    @property
    def use_mla(self) -> bool:
        return _LAYOUT_DEFS[self.layout]["use_mla"]


_FORMAT_ID_RE = re.compile(
    r"^[A-Za-z0-9._:-]+(/[A-Za-z0-9._:-]+)*$"
)  # simple, versioned-friendly


class _LayoutDef(TypedDict):
    family: Family
    canonical: Canonical
    separation: Optional[Separation]
    use_mla: bool


_LAYOUT_DEFS: dict[KVCacheLayout, _LayoutDef] = {
    "MHA_DENSE_PACKED": {
        "family": "MHA_DENSE",
        "canonical": "KV_2LTD",
        "separation": "packed",
        "use_mla": False,
    },
    "MHA_DENSE_SEPARATED": {
        "family": "MHA_DENSE",
        "canonical": "KV_2LTD",
        "separation": "separated",
        "use_mla": False,
    },
    "MLA_LATENT_PACKED": {
        "family": "MLA_LATENT",
        "canonical": "KV_MLA_FMT",
        "separation": "packed",
        "use_mla": True,
    },
    "MLA_LATENT_SEPARATED": {
        "family": "MLA_LATENT",
        "canonical": "KV_MLA_FMT",
        "separation": "separated",
        "use_mla": True,
    },
}


class _KVMetadata(Protocol):
    kv_format: Optional[KVCacheFormat]
    kv_shape: tuple
    use_mla: bool
    kv_dtype: object
    chunk_size: int


def _validate_layer_groups(layer_groups: Sequence[KVLayerGroupSpec]) -> None:
    if not layer_groups:
        raise ValueError("layer_groups must be non-empty")

    # Ensure groups are sorted and non-overlapping.
    sorted_groups = sorted(layer_groups, key=lambda g: g.start_layer)
    prev_end = 0
    for group in sorted_groups:
        if group.start_layer < 0:
            raise ValueError("start_layer must be non-negative")
        if group.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if group.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not group.dtype:
            raise ValueError("dtype must be non-empty")
        if group.start_layer < prev_end:
            raise ValueError("layer_groups overlap or are unsorted")
        prev_end = group.start_layer + group.num_layers


def validate_kvcache_format(fmt: KVCacheFormat) -> KVCacheFormat:
    """Validate a KVCacheFormat, raising ValueError on issues."""
    if fmt.layout not in _LAYOUT_DEFS:
        raise ValueError(f"unsupported layout: {fmt.layout}")

    if fmt.l0.block_size <= 0:
        raise ValueError("block_size must be positive")
    if fmt.l0.addressing != "gpu_block_ids":
        raise ValueError("addressing must be 'gpu_block_ids' for v1")

    if not fmt.format_id or not _FORMAT_ID_RE.match(fmt.format_id):
        raise ValueError(
            "format_id must be non-empty, versioned string-like "
            "(e.g., 'mha_dense/packed/v1')"
        )

    if fmt.schema_version != KV_FORMAT_SCHEMA_VERSION:
        raise ValueError(
            "schema_version mismatch: "
            f"{fmt.schema_version} != {KV_FORMAT_SCHEMA_VERSION}"
        )

    _validate_layer_groups(fmt.layer_groups)
    return fmt


def serialize_kvcache_format(fmt: KVCacheFormat) -> bytes:
    """Serialize format to msgpack bytes."""
    return msgspec.msgpack.encode(validate_kvcache_format(fmt))


def deserialize_kvcache_format(data: bytes) -> KVCacheFormat:
    """Deserialize and validate format from msgpack bytes."""
    fmt = msgspec.msgpack.decode(data, type=KVCacheFormat)
    return validate_kvcache_format(fmt)


def _dtype_to_str(dtype: object) -> str:
    # Accept torch.dtype or string; avoid importing torch to keep core engine-agnostic.
    if dtype is None:
        raise ValueError("dtype cannot be None")
    if isinstance(dtype, str):
        if not dtype:
            raise ValueError("dtype string cannot be empty")
        return dtype
    return str(dtype)


def _layout_from_kv_shape(
    kv_shape: Optional[tuple], use_mla: bool, kv_dtype: object
) -> dict:
    if kv_shape is None:
        raise ValueError("kv_shape must be provided when kv_format is absent")
    num_layers = kv_shape[0]
    block_size = kv_shape[2]
    hidden_dim = kv_shape[3] * kv_shape[4]
    return {
        "block_size": block_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "use_mla": use_mla,
        "dtype": kv_dtype,
    }


def get_kv_layout_from_metadata(
    metadata: _KVMetadata, *, hidden_dim_context: str
) -> tuple[dict, Optional[KVCacheFormat]]:
    """Return a layout dict and optional validated format from metadata.

    This centralizes the validation of kv_format against legacy metadata fields
    to avoid duplicating the same logic in multiple call sites.
    """
    kv_format = getattr(metadata, "kv_format", None)
    kv_shape = getattr(metadata, "kv_shape", None)
    use_mla = metadata.use_mla
    kv_dtype = metadata.kv_dtype
    chunk_size = getattr(metadata, "chunk_size", None)

    if kv_format is None:
        return _layout_from_kv_shape(kv_shape, use_mla, kv_dtype), None

    fmt = validate_kvcache_format(kv_format)
    block_size = fmt.l0.block_size
    total_layers = max(g.start_layer + g.num_layers for g in fmt.layer_groups)
    hidden_dims = {g.hidden_dim for g in fmt.layer_groups}
    if len(hidden_dims) != 1:
        raise ValueError(
            "kv_format must use a uniform hidden_dim across layer_groups "
            f"{hidden_dim_context}"
        )
    hidden_dim = hidden_dims.pop()

    if kv_shape is not None:
        num_layers_shape = kv_shape[0]
        chunk_size_shape = kv_shape[2]
        hidden_dim_shape = kv_shape[3] * kv_shape[4]
        if num_layers_shape != total_layers:
            raise ValueError(
                "kv_format layers "
                f"({total_layers}) mismatch kv_shape[0] ({num_layers_shape})"
            )
        if chunk_size_shape != block_size:
            raise ValueError(
                "kv_format block_size "
                f"({block_size}) mismatch kv_shape chunk_size ({chunk_size_shape})"
            )
        if hidden_dim_shape != hidden_dim:
            raise ValueError(
                "kv_format hidden_dim "
                f"({hidden_dim}) mismatch kv_shape inferred hidden_dim "
                f"({hidden_dim_shape})"
            )

    if chunk_size is not None and chunk_size != block_size:
        raise ValueError(
            "kv_format block_size "
            f"({block_size}) mismatch metadata.chunk_size ({chunk_size})"
        )

    dtype_str = _dtype_to_str(kv_dtype)
    for group in fmt.layer_groups:
        if group.dtype != dtype_str:
            raise ValueError(
                f"kv_format dtype {group.dtype} mismatch metadata.kv_dtype {dtype_str}"
            )

    expected_use_mla = fmt.use_mla
    if use_mla != expected_use_mla:
        raise ValueError(
            "kv_format family "
            f"({fmt.family}) inconsistent with metadata.use_mla ({use_mla})"
        )

    layout = {
        "block_size": block_size,
        "hidden_dim": hidden_dim,
        "num_layers": total_layers,
        "use_mla": expected_use_mla,
        "dtype": kv_dtype,
    }
    return layout, fmt


def build_dense_format_single_group(
    num_layers: int,
    dtype: object,
    hidden_dim: int,
    block_size: int,
    *,
    use_mla: bool = False,
    separation: Separation = "packed",
    format_version: str = "v1",
) -> KVCacheFormat:
    """Convenience builder for a single-group dense or MLA layout."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")

    if separation not in ("packed", "separated"):
        raise ValueError("separation must be 'packed' or 'separated'")

    if use_mla:
        layout: KVCacheLayout = (
            "MLA_LATENT_PACKED" if separation == "packed" else "MLA_LATENT_SEPARATED"
        )
    else:
        layout = "MHA_DENSE_PACKED" if separation == "packed" else "MHA_DENSE_SEPARATED"

    format_id = (
        f"mla_latent/{separation}/{format_version}"
        if use_mla
        else f"mha_dense/{separation}/{format_version}"
    )

    layer_group = KVLayerGroupSpec(
        start_layer=0,
        num_layers=num_layers,
        dtype=_dtype_to_str(dtype),
        hidden_dim=hidden_dim,
    )
    l0 = L0LayoutSpec(addressing="gpu_block_ids", block_size=block_size)
    fmt = KVCacheFormat(
        layout=layout,
        l0=l0,
        layer_groups=[layer_group],
        format_id=format_id,
    )
    return validate_kvcache_format(fmt)
