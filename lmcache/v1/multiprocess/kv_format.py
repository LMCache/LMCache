# SPDX-License-Identifier: Apache-2.0
"""
KVCache Format Descriptors for Engine-Agnostic LMCache Multi-Process Mode.

This module defines dataclasses that describe the KV cache format used by
different inference engines (SGLang, vLLM, etc.) so that the LMCache server
can handle them uniformly without engine-specific branching.

Format Families:
- MHA_DENSE: Standard Multi-Head Attention with dense KV cache
- MLA_LATENT: Multi-head Latent Attention (e.g., DeepSeek)
- SPARSE: Sparse attention patterns (future support)
"""

# Standard
from dataclasses import dataclass
from typing import List, Literal, Optional

# Third Party
import msgspec


@dataclass(frozen=True)
class SparseSpec:
    """
    Specification for sparse attention patterns (future use).

    As per design doc section 5.3, this supports NSA, double-sparsity, and other
    sparse attention schemes.

    Attributes:
        kind: The sparse attention kind (NSA, DOUBLE_SPARSITY, OTHER)
        aux_metadata_schema: Versioned schema string for aux metadata (e.g., "nsa/v1")
    """

    kind: Literal["NSA", "DOUBLE_SPARSITY", "OTHER"]
    aux_metadata_schema: str  # versioned schema string, e.g., "nsa/v1"


@dataclass(frozen=True)
class L0LayoutSpec:
    """
    Describes the L0 (GPU-side) layout of the KV cache.

    Attributes:
        separation: Whether K and V are packed together or separated
            - "packed": K and V interleaved or in same tensor
            - "separated": K and V in separate tensor arrays
        addressing: How blocks are addressed
            - "gpu_block_ids": Block IDs map to GPU memory blocks
        block_size: Number of tokens per block
        pointer_order: How pointers are ordered (e.g., "K_layers_then_V_layers")
        sparse_spec: Optional sparse attention specification
    """

    separation: Literal["packed", "separated"]
    addressing: Literal["gpu_block_ids"]
    block_size: int
    pointer_order: Optional[str] = None
    sparse_spec: Optional[SparseSpec] = None


@dataclass(frozen=True)
class KVLayerGroupSpec:
    """
    Describes a group of layers with the same KV cache shape.

    This allows handling models with heterogeneous layer configurations
    (e.g., different hidden dimensions across layer groups).

    Attributes:
        start_layer: Starting layer index (inclusive)
        num_layers: Number of layers in this group
        dtype: Data type as string (e.g., "float16", "bfloat16")
        hidden_dim: Hidden dimension size (num_heads * head_size for MHA)
        num_heads: Number of attention heads (optional, for MHA)
        head_size: Size per attention head (optional, for MHA)
    """

    start_layer: int
    num_layers: int
    dtype: str
    hidden_dim: int
    num_heads: Optional[int] = None
    head_size: Optional[int] = None


@dataclass(frozen=True)
class KVCacheFormat:
    """
    Complete KV cache format descriptor.

    This dataclass captures all information needed for the LMCache server
    to handle KV cache transfers without engine-specific code.

    Attributes:
        family: The format family (MHA_DENSE, MLA_LATENT, SPARSE)
        canonical: The canonical memory format name
        l0: L0 layout specification
        layer_groups: List of layer group specifications
        format_id: Unique identifier established at REGISTER time
        total_layers: Total number of layers (computed)
    """

    family: Literal["MHA_DENSE", "MLA_LATENT", "SPARSE"]
    canonical: Literal["KV_2LTD", "KV_MLA_FMT", "KV_SPARSE_FMT"]
    l0: L0LayoutSpec
    layer_groups: List[KVLayerGroupSpec]
    format_id: str

    @property
    def total_layers(self) -> int:
        """Returns the total number of layers across all groups."""
        return sum(group.num_layers for group in self.layer_groups)

    @property
    def block_size(self) -> int:
        """Returns the block size from L0 layout."""
        return self.l0.block_size

    @property
    def is_mla(self) -> bool:
        """Returns True if this is an MLA format."""
        return self.family == "MLA_LATENT"

    @property
    def hidden_dim(self) -> int:
        """Returns the hidden dimension from the first layer group."""
        if self.layer_groups:
            return self.layer_groups[0].hidden_dim
        raise ValueError("No layer groups defined")

    @property
    def dtype_str(self) -> str:
        """Returns the dtype string from the first layer group."""
        if self.layer_groups:
            return self.layer_groups[0].dtype
        raise ValueError("No layer groups defined")


# ==================== Serialization Support ====================


def kv_format_to_dict(fmt: KVCacheFormat) -> dict:
    """Convert KVCacheFormat to a dictionary for serialization."""
    return {
        "family": fmt.family,
        "canonical": fmt.canonical,
        "l0": {
            "separation": fmt.l0.separation,
            "addressing": fmt.l0.addressing,
            "block_size": fmt.l0.block_size,
            "pointer_order": fmt.l0.pointer_order,
            "sparse_spec": None
            if fmt.l0.sparse_spec is None
            else {
                "kind": fmt.l0.sparse_spec.kind,
                "aux_metadata_schema": fmt.l0.sparse_spec.aux_metadata_schema,
            },
        },
        "layer_groups": [
            {
                "start_layer": g.start_layer,
                "num_layers": g.num_layers,
                "dtype": g.dtype,
                "hidden_dim": g.hidden_dim,
                "num_heads": g.num_heads,
                "head_size": g.head_size,
            }
            for g in fmt.layer_groups
        ],
        "format_id": fmt.format_id,
    }


def dict_to_kv_format(d: dict) -> KVCacheFormat:
    """Convert a dictionary back to KVCacheFormat."""
    l0_dict = d["l0"]
    sparse_spec = None
    if l0_dict.get("sparse_spec"):
        sparse_spec = SparseSpec(**l0_dict["sparse_spec"])

    l0 = L0LayoutSpec(
        separation=l0_dict["separation"],
        addressing=l0_dict["addressing"],
        block_size=l0_dict["block_size"],
        pointer_order=l0_dict.get("pointer_order"),
        sparse_spec=sparse_spec,
    )

    layer_groups = [
        KVLayerGroupSpec(
            start_layer=g["start_layer"],
            num_layers=g["num_layers"],
            dtype=g["dtype"],
            hidden_dim=g["hidden_dim"],
            num_heads=g.get("num_heads"),
            head_size=g.get("head_size"),
        )
        for g in d["layer_groups"]
    ]

    return KVCacheFormat(
        family=d["family"],
        canonical=d["canonical"],
        l0=l0,
        layer_groups=layer_groups,
        format_id=d["format_id"],
    )


class KVCacheFormatEncoder:
    """Custom encoder for KVCacheFormat serialization via msgspec."""

    @staticmethod
    def Serialize(obj: KVCacheFormat) -> bytes:
        """Serialize KVCacheFormat to bytes."""
        return msgspec.msgpack.encode(kv_format_to_dict(obj))

    @staticmethod
    def Deserialize(data: bytes) -> KVCacheFormat:
        """Deserialize bytes to KVCacheFormat."""
        d = msgspec.msgpack.decode(data)
        return dict_to_kv_format(d)


# ==================== Helper Functions ====================


def create_mha_dense_format(
    num_layers: int,
    hidden_dim: int,
    dtype: str,
    block_size: int,
    num_heads: Optional[int] = None,
    head_size: Optional[int] = None,
    format_id: Optional[str] = None,
) -> KVCacheFormat:
    """
    Create a KVCacheFormat for standard MHA dense attention.

    Args:
        num_layers: Number of transformer layers
        hidden_dim: Hidden dimension (num_heads * head_size)
        dtype: Data type string (e.g., "float16")
        block_size: Tokens per block
        num_heads: Number of attention heads (optional)
        head_size: Size per head (optional)
        format_id: Custom format ID (auto-generated if None)

    Returns:
        KVCacheFormat for MHA dense attention
    """
    l0 = L0LayoutSpec(
        separation="separated",
        addressing="gpu_block_ids",
        block_size=block_size,
        pointer_order="K_layers_then_V_layers",
    )

    layer_group = KVLayerGroupSpec(
        start_layer=0,
        num_layers=num_layers,
        dtype=dtype,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_size=head_size,
    )

    if format_id is None:
        format_id = f"MHA_DENSE/KV_2LTD/{dtype}/v1"

    return KVCacheFormat(
        family="MHA_DENSE",
        canonical="KV_2LTD",
        l0=l0,
        layer_groups=[layer_group],
        format_id=format_id,
    )


def create_mla_latent_format(
    num_layers: int,
    hidden_dim: int,
    dtype: str,
    block_size: int,
    format_id: Optional[str] = None,
) -> KVCacheFormat:
    """
    Create a KVCacheFormat for MLA (Multi-head Latent Attention).

    Args:
        num_layers: Number of transformer layers
        hidden_dim: Latent hidden dimension
        dtype: Data type string (e.g., "float16")
        block_size: Tokens per block
        format_id: Custom format ID (auto-generated if None)

    Returns:
        KVCacheFormat for MLA latent attention
    """
    l0 = L0LayoutSpec(
        separation="separated",
        addressing="gpu_block_ids",
        block_size=block_size,
        pointer_order="K_layers_then_V_layers",
    )

    layer_group = KVLayerGroupSpec(
        start_layer=0,
        num_layers=num_layers,
        dtype=dtype,
        hidden_dim=hidden_dim,
    )

    if format_id is None:
        format_id = f"MLA_LATENT/KV_MLA_FMT/{dtype}/v1"

    return KVCacheFormat(
        family="MLA_LATENT",
        canonical="KV_MLA_FMT",
        l0=l0,
        layer_groups=[layer_group],
        format_id=format_id,
    )
