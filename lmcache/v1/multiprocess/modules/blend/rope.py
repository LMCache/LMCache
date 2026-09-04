# SPDX-License-Identifier: Apache-2.0
"""Blend re-RoPE state and geometry (pure: no server context or streams)."""

# Standard
from dataclasses import dataclass, field
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format import get_spec_class


# torch dtype -> at::ScalarType (rope dispatch); missing -> Python fallback.
_TORCH_TO_AT_SCALAR = {
    torch.float16: 5,  # at::ScalarType::Half
    torch.float32: 6,  # at::ScalarType::Float
    torch.bfloat16: 15,  # at::ScalarType::BFloat16
}


@dataclass
class _CBRopeState:
    """Per-instance RoPE state IPC-shared from vLLM; dangles on reallocate.

    Models with per-layer-type RoPE (distinct local/global theta)
    register one cache per distinct rope and a per-layer index into
    ``cos_sin_caches``.
    """

    head_size: int
    is_neox_style: bool  # NeoX = contiguous halves; else GPT-J.
    cos_sin_caches: list[torch.Tensor]
    group_to_cache: list[int]  # engine group idx -> cache idx; empty = cache 0
    # Per-group rotation window ``(offset_elems, width_elems)``; ``None``
    # skips re-RoPE for the group, empty list = legacy inferred geometry.
    # Required for MLA: inference would rotate the latent's content dims.
    group_rot: "list[tuple[int, int] | None]" = field(default_factory=list)

    def rot_for_group(
        self, engine_group_idx: int, dtype: "torch.dtype | None" = None
    ) -> "tuple[int, int] | None":
        """The rotation window for one kernel group.

        Args:
            engine_group_idx: The kernel group's engine group index.
            dtype: The kernel group's buffer dtype, when known. One engine
                group can hold several *kernel* groups (GLM: the bf16 latent
                and the uint8 fp8 index cache both sit in engine group 0), so
                under a DECLARED map a non-float kernel group is skipped —
                the rope kernel cannot rotate quantized rows, and the
                declared window describes the family's float plane. Legacy
                registrations keep today's behavior (no dtype-based skip).

        Returns:
            ``(offset_elems, width_elems)``, or ``None`` when the group's
            re-RoPE is skipped (declared ``[]``, or non-float under a
            declared map). Legacy registrations (empty ``group_rot``) get
            ``(0, head_size)``.

        Note:
            Answers only "what window would rotate"; whether a cos/sin
            cache exists at all (NoPE) is checked by the consumers.

        Raises:
            RuntimeError: If ``engine_group_idx`` is outside a non-empty map.
        """
        if not self.group_rot:
            return (0, self.head_size)
        if dtype is not None and not dtype.is_floating_point:
            return None
        if engine_group_idx >= len(self.group_rot):
            raise RuntimeError(
                f"CB re-RoPE: engine group {engine_group_idx} has no rope "
                f"geometry (map covers {len(self.group_rot)} groups)."
            )
        return self.group_rot[engine_group_idx]

    def cache_for_group(self, engine_group_idx: int) -> "torch.Tensor | None":
        """The cos/sin cache for one engine group.

        Engine groups partition layers by attention type, and rope follows
        attention type (sliding=local theta, full=global theta),
        so each engine group has exactly one cache. NoPE models register
        zero caches; every group then returns ``None`` and re-RoPE is
        skipped.

        Args:
            engine_group_idx: The kernel group's engine group index.

        Returns:
            The group's cos/sin cache tensor, or ``None`` for a NoPE model.

        Raises:
            RuntimeError: If ``engine_group_idx`` is outside the map.
        """
        if not self.cos_sin_caches:
            return None
        if not self.group_to_cache:
            return self.cos_sin_caches[0]
        if engine_group_idx >= len(self.group_to_cache):
            raise RuntimeError(
                f"CB re-RoPE: engine group {engine_group_idx} has no rope "
                f"cache mapping (map covers {len(self.group_to_cache)} groups)."
            )
        return self.cos_sin_caches[self.group_to_cache[engine_group_idx]]


def _cb_group_rope_geometry(
    group: Any,
    kv_size: int,
    hidden_dim: int,
    head_size: int,
    group_idx: int,
    rot: "tuple[int, int] | None" = None,
) -> "tuple[bool, int, int, int]":
    """Per-group re-RoPE geometry rules, shared by the batched rope path and
    the retrieve-plan builder so they cannot drift.

    Fused blocks-first K/V packs K+V into a doubled head dim (kv_size==1);
    detect it so only the K half is re-RoPE'd in place. kv_size==1 without
    fused packing is the M3 key-only index side cache; kv_size==2 is main
    K/V. In every case only the K plane is rotated.

    ``rot`` is the declared ``(offset_elems, width_elems)`` rotation window;
    offset > 0 means MLA (rope dims trail the row) — the row is one "head"
    and only ``[offset, offset + width)`` rotates. MLA groups must arrive
    with ``rot`` set: undeclared, a 576-wide latent passes the inference
    checks below as 9 x 64 heads and its content dims get rotated.

    Returns:
        ``(fused_packed, per_head, n_heads, rot_offset)`` — per-head width is
        ``2 * head_size`` for fused-packed layouts; ``rot_offset`` is the
        element offset of the rotation window within each per-head row (0
        for every non-MLA layout).

    Raises:
        RuntimeError: On a compressed (compress_ratio != 1) layout, a
            kv_size other than 2 (K/V) or 1 (key-only index), a
            head_size/hidden_dim mismatch, or a declared window that does
            not fit the row.
    """
    if group.tokens_per_block != group.slots_per_block:
        raise RuntimeError(
            f"CB: group {group_idx} is compressed "
            f"(tokens_per_block={group.tokens_per_block}, "
            f"slots_per_block={group.slots_per_block}); "
            f"compressed layouts unsupported."
        )
    if rot is not None and rot[0] > 0:
        rot_offset, rot_width = rot
        if kv_size != 1:
            raise RuntimeError(
                f"CB: group {group_idx} declares an MLA rope window "
                f"{rot} but has kv_size={kv_size}; MLA latents are a "
                "single plane (kv_size 1)."
            )
        if rot_offset + rot_width != hidden_dim:
            raise RuntimeError(
                f"CB: group {group_idx} rope window {rot} does not end "
                f"the row (hidden_dim={hidden_dim}); MLA latents are "
                "[content | rope]."
            )
        return False, hidden_dim, 1, rot_offset
    if kv_size not in (1, 2):
        raise RuntimeError(
            f"CB: group {group_idx} has kv_size={kv_size}; only K/V "
            "(2), fused-packed K/V, key-only (1), and declared-MLA "
            "layouts are supported."
        )
    # Use the spec fact, not a format list: a missed fused format is treated
    # as split K/V and the V half of every head gets re-RoPE'd.
    engine_kv_format = getattr(group, "engine_kv_format", None)
    fused_format = (
        engine_kv_format is not None
        and get_spec_class(engine_kv_format).is_fused_packed
    )
    declared_hs = getattr(getattr(group, "shape_desc", None), "hs", None)
    fused_packed = fused_format and (
        declared_hs is None or declared_hs == 2 * head_size
    )
    per_head = head_size * (2 if fused_packed else 1)
    n_heads = hidden_dim // per_head
    if n_heads * per_head != hidden_dim:
        raise RuntimeError(
            f"CB rope: group {group_idx} hidden_dim ({hidden_dim}) not a "
            f"multiple of per-head width ({per_head}; fused={fused_packed})."
        )
    return fused_packed, per_head, n_heads, 0
