# SPDX-License-Identifier: Apache-2.0
"""Blend Python fallback scatter: per-wave rope + per-token paged writes."""

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.v1.multiprocess.custom_types import CBMatchResult
from lmcache.v1.multiprocess.modules.blend.rope import (
    _cb_group_rope_geometry,
    _CBRopeState,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native


def _group_slot_mappings(
    resolved_groups: "list[tuple[torch.Tensor, int]]", pos: torch.Tensor
) -> "list[torch.Tensor]":
    """Per-group paged slot ids for the logical positions ``pos``
    (``block_ids[pos // bs] * bs + pos % bs``). The div/mod pair is shared
    across groups with the same block size — dispatch count is the cost that
    matters under the shared GIL."""
    div_mod: "dict[int, tuple[torch.Tensor, torch.Tensor]]" = {}
    mappings: "list[torch.Tensor]" = []
    for group_block_ids, group_bs in resolved_groups:
        pair = div_mod.get(group_bs)
        if pair is None:
            pair = (pos // group_bs, pos % group_bs)
            div_mod[group_bs] = pair
        mappings.append(group_block_ids[pair[0]] * group_bs + pair[1])
    return mappings


class ScatterFallbackMixin:
    """The Python wave-loop retrieve path (methods of
    :class:`~lmcache.v1.multiprocess.modules.blend.module.BlendModule`,
    moved verbatim); the native flat plan is the production path."""

    def _apply_cb_rope_batched(
        self,
        gpu_context: BaseCacheContext,
        rope_state: _CBRopeState,
        batch_len: int,
        slots_to_rope: list[tuple[int, int, int]],
        staged_kernel: list[int],
    ) -> None:
        """Re-RoPE the given tmp-pool slots in place (K-only, per kernel group).

        Args:
            gpu_context (GPUCacheContext): The instance's GPU cache context.
            rope_state (_CBRopeState): Cached cos/sin + head layout.
            batch_len (int): Number of tmp slots staged for this batch.
            slots_to_rope (list[tuple[int, int, int]]): ``(slot_idx, old_st,
                cur_st)`` per shifted slot — re-RoPE K from stored position
                ``old_st`` to new position ``cur_st``.
            staged_kernel (list[int]): Kernel-group indices blend staged (see
                :meth:`_cb_staged_groups`); recurrent groups are absent and
                stay untouched.

        Raises:
            RuntimeError: On a compressed (compress_ratio != 1) layout, a
                kv_size other than 2 (K/V) or 1 (key-only index), or a
                head_size/hidden_dim mismatch.
        """
        if not slots_to_rope:
            return
        if not rope_state.cos_sin_caches:
            return  # NoPE model: stored K is position-independent.
        for group_idx in staged_kernel:
            group = gpu_context.kv_layer_groups_manager.kernel_groups[group_idx]
            all_slots = [
                gpu_context.get_temp_kernel_group_buffer(slot_idx, group_idx)
                for slot_idx in range(batch_len)
            ]
            rot = rope_state.rot_for_group(group.engine_group_idx, all_slots[0].dtype)
            if rot is None:
                # Skipped group (declared [], or quantized under a declared
                # map): scattered as-is, positions left stale.
                continue
            num_layers, slots, hidden_dim = all_slots[0].shape[1:]
            fused_packed, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
                group,
                int(all_slots[0].shape[0]),
                int(hidden_dim),
                rope_state.head_size,
                group_idx,
                rot,
            )
            # Per-group rope cache: dual-RoPE models rotate each kernel
            # group with its own theta's cos/sin; NoPE returned above.
            group_cos_sin = rope_state.cache_for_group(group.engine_group_idx)
            assert group_cos_sin is not None
            if rot_offset > 0 and int(group_cos_sin.shape[1]) != rot[1]:
                raise RuntimeError(
                    f"CB re-RoPE: group {group_idx} declares rope width "
                    f"{rot[1]} but the cos/sin cache has rot_dim "
                    f"{int(group_cos_sin.shape[1])}."
                )
            # slot ramp tiled across layers is invariant per (num_layers,
            # slots) — cache it; each shifted slot then just adds its offset.
            device = all_slots[0].device
            sp_key = (str(device), num_layers, slots)
            sp_cache = getattr(self, "_cb_sp_rep_cache", None)
            if sp_cache is None:
                sp_cache = {}
                self._cb_sp_rep_cache = sp_cache
            slot_positions_rep = sp_cache.get(sp_key)
            if slot_positions_rep is None:
                slot_positions_rep = torch.arange(
                    slots, device=device, dtype=torch.long
                ).repeat(num_layers)
                sp_cache[sp_key] = slot_positions_rep
            for slot_idx, old_st, cur_st in slots_to_rope:
                # reshape returns an in-place view (tmp slots are contiguous).
                k_view = all_slots[slot_idx][0].reshape(
                    num_layers * slots, n_heads, per_head
                )
                if rot_offset > 0:
                    # MLA latent: rotate only the trailing rope window. The
                    # slice advances data_ptr to the window start; the kernel
                    # addresses rows via the explicit head_stride (the full
                    # row width), so the non-contiguous view is safe. It then
                    # rotates rot_dim (= window width) dims from that base —
                    # the content dims [0, rot_offset) are never touched.
                    device_ops.rotary_embedding_k_fused_strided(
                        old_st + slot_positions_rep,
                        cur_st + slot_positions_rep,
                        k_view[..., rot_offset:],
                        per_head - rot_offset,  # window width
                        per_head,  # head_stride: the full latent row
                        group_cos_sin,
                        rope_state.is_neox_style,
                    )
                elif fused_packed:
                    # Strided kernel rotates only the K half of each slot.
                    device_ops.rotary_embedding_k_fused_strided(
                        old_st + slot_positions_rep,
                        cur_st + slot_positions_rep,
                        k_view,
                        rope_state.head_size,
                        per_head,  # head_stride: hop over the packed V half
                        group_cos_sin,
                        rope_state.is_neox_style,
                    )
                else:
                    device_ops.rotary_embedding_k_fused(
                        old_st + slot_positions_rep,
                        cur_st + slot_positions_rep,
                        k_view,
                        rope_state.head_size,
                        group_cos_sin,
                        rope_state.is_neox_style,
                    )

    def _scatter_batch_to_paged(
        self,
        gpu_context: BaseCacheContext,
        resolved_groups: "list[tuple[torch.Tensor, int]]",
        batch: "list[tuple[CBMatchResult, Any]]",
        head_size: int,
        staged_kernel: list[int],
    ) -> None:
        """Scatter one tmp-slot batch into the paged KV, one launch per
        (kernel group, tmp slot), straight from each slot's contiguous buffer
        (no ``torch.cat``). A partially filled slot is narrowed first: the
        kernel scatters ``size(2)`` tokens, so a full-capacity buffer would
        mis-align every later slot against ``slot_mapping``.

        Args:
            gpu_context (GPUCacheContext): The instance's GPU cache context.
            resolved_groups: Per STAGED kernel group ``(block_ids,
                block_size)``, positionally aligned with ``staged_kernel``.
            batch: ``(match, memory_obj)`` pairs; slot ``i`` holds ``batch[i]``.
            head_size: RoPE head size forwarded to the kernel.
            staged_kernel: Kernel-group indices blend staged (see
                :meth:`_cb_staged_groups`); recurrent groups are absent and
                never scattered.
        """
        kgm = gpu_context.kv_layer_groups_manager
        tok_counts = [int(r.cur_ed - r.cur_st) for (r, _) in batch]
        pos = torch.cat(
            [
                torch.arange(
                    r.cur_st,
                    r.cur_ed,
                    device=gpu_context.device,
                    dtype=torch.long,
                )
                for (r, _) in batch
            ]
        )
        slot_mappings = _group_slot_mappings(resolved_groups, pos)
        for pos_idx, group_idx in enumerate(staged_kernel):
            _, group_bs = resolved_groups[pos_idx]
            slot_mapping = slot_mappings[pos_idx]
            # Per-group block count: under HMA the sliding group has fewer
            # blocks than the full group, so gpu_context.num_blocks (group
            # 0's) would truncate the other groups' bounds check.
            page_buffer_size = kgm.kernel_groups[group_idx].shape_desc.nb * group_bs
            tok_off = 0
            for slot_idx, n_tok in enumerate(tok_counts):
                key_value = gpu_context.get_temp_kernel_group_buffer(
                    slot_idx, group_idx
                )
                if n_tok < key_value.shape[2]:
                    # Partial chunk: narrow to the real token count (the
                    # kernel scatters size(2) tokens). Slicing dim 2 breaks
                    # contiguity, so this one slot pays a small copy.
                    key_value = key_value[:, :, :n_tok].contiguous()
                device_ops.multi_layer_kv_transfer(
                    key_value,
                    gpu_context.get_kernel_group_kv_pointers(group_idx),
                    slot_mapping[tok_off : tok_off + n_tok],
                    gpu_context.device,
                    page_buffer_size,
                    lmcache_native.TransferDirection.H2D,
                    gpu_context.get_engine_kv_format(group_idx),
                    block_size=group_bs,
                    head_size=head_size,
                )
                tok_off += n_tok
