# SPDX-License-Identifier: Apache-2.0
"""SGLang KV pool adapter for the unified LMCache connector."""

# Future
from __future__ import annotations

# Standard
from typing import Any, Optional

# Third Party
from sglang.srt.configs.model_config import AttentionArch, is_deepseek_v4
from sglang.srt.mem_cache.unified_cache.components import ComponentType
import torch

# First Party
from lmcache.integration.sglang.lmcache_mp_metadata import (
    SGLangKVComponentGroup,
)


class SGLangUnifiedKVAdapter:
    """Map SGLang unified KV components to LMCache groups and block IDs."""

    def __init__(
        self,
        *,
        token_to_kv_pool_allocator: Any,
        req_to_token_pool: Any,
        tree_components: tuple[ComponentType, ...],
        mamba_component: Any,
        page_size: int,
        sliding_window_size: Optional[int],
    ) -> None:
        # check tree components
        if len(tree_components) <= 0:
            raise ValueError("tree_components must not be empty")
        if tree_components[0] is not ComponentType.FULL:
            raise ValueError("tree_components must start with FULL")
        if ComponentType.C128 in tree_components:
            raise NotImplementedError("C128 is not supported by LMCache")

        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_components = tree_components
        self.mamba_component = mamba_component
        self.page_size = page_size
        self.sliding_window_size = sliding_window_size

    def is_mla_enabled(self, model_config: Any) -> bool:
        """Return whether KV is TP-replicated for the MLA optimization."""
        return bool(
            model_config.attention_arch == AttentionArch.MLA
            or is_deepseek_v4(model_config.hf_config)
        )

    def reset_mamba_checkpoint_metadata(self, indices: torch.Tensor) -> None:
        """Reset ReplaySSM cursors that are not persisted by LMCache."""
        mamba_pool = self.req_to_token_pool.mamba_pool
        for name in (
            "replayssm_write_pos",
            "replayssm_cache_base",
            "replayssm_is_flush",
        ):
            value = getattr(mamba_pool, name, None)
            if value is not None:
                value[indices] = 0

    def resolve_registered_groups(self) -> list[SGLangKVComponentGroup]:
        """Map Unified tree components to LMCache engine KV groups."""
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        # Third Party
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        if isinstance(kv_pool, DeepSeekV4TokenToKVPool):
            return self._resolve_dsv4_registered_groups(kv_pool)

        groups: list[SGLangKVComponentGroup] = []
        for component_type in self.tree_components:
            if component_type is ComponentType.FULL:
                component_pool = getattr(kv_pool, "full_kv_pool", kv_pool)
                tensors = self._resolve_pool_tensors(component_pool)
                tensor_rows_per_block = (self.page_size,) * len(tensors)
                dsa_tensors = self._resolve_dsa_indexer_tensors(component_pool)
                if dsa_tensors:
                    tensors = (*tensors, *dsa_tensors)
                    tensor_rows_per_block += (1,) * len(dsa_tensors)
                groups.append(
                    SGLangKVComponentGroup(
                        name=component_type.name.lower(),
                        kv_tensors=tensors,
                        sliding_window_size=-1,
                        tokens_per_block=self.page_size,
                        slots_per_block=self.page_size,
                        tensor_rows_per_block=tensor_rows_per_block,
                        recurrent_state=False,
                    )
                )
            elif component_type is ComponentType.SWA:
                component_pool = getattr(kv_pool, "swa_kv_pool", None)
                if component_pool is None:
                    raise NotImplementedError(
                        f"SWA component requires an SWA KV sub-pool, got "
                        f"{type(kv_pool).__name__}"
                    )
                tensors = self._resolve_pool_tensors(component_pool)
                groups.append(
                    SGLangKVComponentGroup(
                        name=component_type.name.lower(),
                        kv_tensors=tensors,
                        sliding_window_size=self.aligned_swa_window_size(),
                        tokens_per_block=self.page_size,
                        slots_per_block=self.page_size,
                        tensor_rows_per_block=(self.page_size,) * len(tensors),
                        recurrent_state=False,
                    )
                )
            elif component_type is ComponentType.MAMBA:
                mamba_pool = self.req_to_token_pool.mamba_pool
                checkpoint_grid = self.mamba_component.mamba_checkpoint_grid
                groups.append(
                    SGLangKVComponentGroup(
                        name=component_type.name.lower(),
                        kv_tensors=tuple(
                            tensor.view(tensor.shape[0], 1, -1)
                            for tensor in self._resolve_mamba_pool_tensors(mamba_pool)
                        ),
                        sliding_window_size=checkpoint_grid,
                        tokens_per_block=checkpoint_grid,
                        slots_per_block=1,
                        recurrent_state=True,
                    )
                )
            else:
                raise AssertionError(f"Unexpected LMCache component {component_type}")
        return groups

    def aligned_swa_window_size(self) -> int:
        assert self.sliding_window_size is not None
        return (
            (self.sliding_window_size + self.page_size - 1)
            // self.page_size
            * self.page_size
        )

    def device_indices_by_group(
        self,
        full_indices: torch.Tensor,
        *,
        mamba_value: Optional[torch.Tensor] = None,
        mamba_transfer_tokens: Optional[int] = None,
    ) -> list[torch.Tensor]:
        """Translate tree FULL ids into each component's physical address space."""
        allocator = self.token_to_kv_pool_allocator
        result: list[torch.Tensor] = []
        for component_type in self.tree_components:
            if component_type is ComponentType.FULL:
                result.append(allocator.translate_kv_indices_for_transfer(full_indices))
            elif component_type is ComponentType.SWA:
                result.append(allocator.translate_loc_from_full_to_swa(full_indices))
            elif component_type is ComponentType.MAMBA:
                checkpoint_grid = self.mamba_component.mamba_checkpoint_grid
                logical_tokens = (
                    len(full_indices)
                    if mamba_transfer_tokens is None
                    else mamba_transfer_tokens
                )
                if logical_tokens % checkpoint_grid:
                    raise ValueError(
                        "LMCache Mamba transfer range must align to checkpoint "
                        f"grid {checkpoint_grid}, got {logical_tokens} tokens"
                    )
                checkpoint_slots = torch.zeros(
                    logical_tokens // checkpoint_grid,
                    dtype=torch.int64,
                    device=full_indices.device,
                )
                if mamba_value is not None and checkpoint_slots.numel():
                    physical = self.req_to_token_pool.translate_mamba_indices(
                        mamba_value.view(-1)
                    )
                    if physical.numel() != 1:
                        raise ValueError(
                            "LMCache Mamba transfer expects exactly one state slot"
                        )
                    checkpoint_slots[-1] = physical[0]
                result.append(checkpoint_slots)
            else:
                raise AssertionError(f"Unexpected LMCache component {component_type}")
        return result

    def _resolve_pool_tensors(self, kv_pool) -> tuple[torch.Tensor, ...]:
        kv_buffer = getattr(kv_pool, "kv_buffer", None)
        if kv_buffer is not None:
            if not kv_buffer:
                raise NotImplementedError("LMCache cannot register an empty MLA pool")
            tensors = tuple(
                tensor
                for tensor in kv_buffer
                if tensor.numel() > 0 or not getattr(kv_pool, "use_dsa", False)
            )
            if not tensors:
                raise NotImplementedError("DSA pool has no locally owned KV buffers")
            return tensors

        k_buffer = getattr(kv_pool, "k_buffer", None)
        v_buffer = getattr(kv_pool, "v_buffer", None)
        if not k_buffer or not v_buffer or len(k_buffer) != len(v_buffer):
            raise NotImplementedError(
                f"Unsupported SGLang KV pool for LMCache MP: {type(kv_pool).__name__}"
            )
        if getattr(kv_pool, "kv_cache_layout", "nhd") != "nhd":
            raise NotImplementedError(
                "LMCacheUnifiedRadixCache currently supports NHD MHA pools only"
            )
        return tuple([*k_buffer, *v_buffer])

    def _resolve_dsa_indexer_tensors(self, kv_pool) -> tuple[torch.Tensor, ...]:
        """Return page-native DSA indexers that share FULL block IDs."""
        if not getattr(kv_pool, "use_dsa", False):
            return ()
        buffers = getattr(kv_pool, "index_k_with_scale_buffer", None)
        if buffers is None:
            raise NotImplementedError(
                f"DSA pool {type(kv_pool).__name__} has no indexer buffers"
            )
        tensors = tuple(tensor for tensor in buffers if tensor.numel() > 0)
        if not tensors:
            raise NotImplementedError("DSA pool has no locally owned indexer buffers")
        if any(tensor.dim() != 2 for tensor in tensors):
            shapes = [tuple(tensor.shape) for tensor in tensors]
            raise NotImplementedError(
                f"LMCache MP expects page-native 2-D DSA indexer buffers, got {shapes}"
            )
        return tensors

    def _validate_page_native_tensors(
        self, pool_name: str, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Validate tensors whose leading row is one complete SGLang page."""
        if not tensors:
            return tensors
        shapes = [tuple(tensor.shape) for tensor in tensors]
        if any(tensor.dim() != 2 for tensor in tensors):
            raise NotImplementedError(
                f"LMCache MP expects 2-D page-native {pool_name} buffers, got {shapes}"
            )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise NotImplementedError(
                f"LMCache MP requires contiguous page-native {pool_name} buffers"
            )
        block_counts = {tensor.shape[0] for tensor in tensors}
        if len(block_counts) != 1:
            raise ValueError(
                f"DeepSeek V4 {pool_name} buffers expose different page counts: "
                f"{sorted(block_counts)}"
            )
        return tensors

    def _resolve_dsv4_full_page_tensors(self, kv_pool) -> tuple[torch.Tensor, ...]:
        """Resolve DS V4 C4/C128 sidecars that share FULL block IDs."""
        if getattr(kv_pool, "_unified_kv", False):
            raise NotImplementedError(
                "LMCache DeepSeek V4 does not yet support ROCm unified_kv_triton; "
                "its request-scoped SWA ring has no content-stable block-id space"
            )

        c4_pool = getattr(kv_pool, "c4_kv_pool", None)
        if c4_pool is not None and hasattr(
            c4_pool, "full_to_hisparse_device_index_mapping"
        ):
            raise NotImplementedError(
                "LMCache DeepSeek V4 does not yet support HiSparse C4 remapping"
            )

        tensors: list[torch.Tensor] = []
        c4_buffers = getattr(c4_pool, "kv_buffer", None)
        if c4_buffers:
            tensors.extend(tensor for tensor in c4_buffers if tensor.numel() > 0)

        indexer_pool = getattr(kv_pool, "c4_indexer_kv_pool", None)
        if indexer_pool is not None:
            if getattr(indexer_pool, "uses_aiter_fp4_layout", False):
                payload_buffers = getattr(indexer_pool, "index_k_payload_buffer", None)
                scale_buffers = getattr(indexer_pool, "index_k_scale_buffer", None)
                if not payload_buffers or not scale_buffers:
                    raise NotImplementedError(
                        "DeepSeek V4 AITER FP4 indexer is missing payload or scale "
                        "buffers"
                    )
                for buffers in (payload_buffers, scale_buffers):
                    tensors.extend(
                        tensor.view(torch.uint8).reshape(tensor.shape[0], -1)
                        for tensor in buffers
                        if tensor.numel() > 0
                    )
            else:
                buffers = getattr(indexer_pool, "index_k_with_scale_buffer", None)
                if buffers:
                    tensors.extend(tensor for tensor in buffers if tensor.numel() > 0)

        c128_pool = getattr(kv_pool, "c128_kv_pool", None)
        c128_buffers = getattr(c128_pool, "kv_buffer", None)
        if c128_buffers:
            tensors.extend(tensor for tensor in c128_buffers if tensor.numel() > 0)

        resolved = self._validate_page_native_tensors("FULL sidecar", tuple(tensors))
        if not resolved:
            raise NotImplementedError(
                "DeepSeek V4 pool has no locally owned C4/C128/indexer buffers"
            )
        return resolved

    def _resolve_dsv4_swa_page_tensors(self, kv_pool) -> tuple[torch.Tensor, ...]:
        """Resolve DS V4 SWA KV and C4 state with shared SWA block IDs."""
        swa_pool = getattr(kv_pool, "swa_kv_pool", None)
        if swa_pool is None:
            raise NotImplementedError(
                "LMCache DeepSeek V4 requires the standard page-addressed SWA pool"
            )

        swa_tensors = tuple(
            tensor
            for tensor in getattr(swa_pool, "kv_buffer", ())
            if tensor.numel() > 0
        )
        resolved_swa = self._validate_page_native_tensors("SWA", swa_tensors)
        if not resolved_swa:
            raise NotImplementedError(
                "DeepSeek V4 pool has no locally owned SWA buffers"
            )
        swa_page_count = resolved_swa[0].shape[0]
        tensors: list[torch.Tensor] = list(resolved_swa)
        for state_pools in (
            getattr(kv_pool, "compress_state_pools", ()),
            getattr(kv_pool, "indexer_compress_state_pools", ()),
        ):
            for state_pool in state_pools:
                if state_pool is None or state_pool.ratio != 4:
                    continue
                state = state_pool.kv_score_buffer.kv_score
                if not state.is_contiguous():
                    raise NotImplementedError(
                        "LMCache DeepSeek V4 requires contiguous C4 state buffers"
                    )
                ring_size = int(state_pool.ring_size)
                if ring_size <= 0:
                    raise ValueError(
                        f"DeepSeek V4 C4 state has invalid ring size {ring_size}"
                    )
                usable_rows = state.shape[0] // ring_size * ring_size
                state_bytes = state.view(torch.uint8).reshape(state.shape[0], -1)
                state_pages = state_bytes[:usable_rows].reshape(
                    usable_rows // ring_size, -1
                )
                if state_pages.shape[0] < swa_page_count:
                    raise ValueError(
                        "DeepSeek V4 C4 state exposes fewer pages than its SWA "
                        f"pool: {state_pages.shape[0]} < {swa_page_count}"
                    )
                tensors.append(state_pages[:swa_page_count])

        resolved = self._validate_page_native_tensors("SWA/state", tuple(tensors))
        if not resolved:
            raise NotImplementedError(
                "DeepSeek V4 pool has no locally owned SWA/state buffers"
            )
        return resolved

    def _resolve_dsv4_registered_groups(self, kv_pool) -> list[SGLangKVComponentGroup]:
        if tuple(self.tree_components) != (ComponentType.FULL, ComponentType.SWA):
            names = [component.name for component in self.tree_components]
            raise NotImplementedError(
                f"LMCache DeepSeek V4 expects FULL/SWA tree components, got {names}"
            )

        full_tensors = self._resolve_dsv4_full_page_tensors(kv_pool)
        swa_tensors = self._resolve_dsv4_swa_page_tensors(kv_pool)
        return [
            SGLangKVComponentGroup(
                name="full",
                kv_tensors=full_tensors,
                sliding_window_size=-1,
                tokens_per_block=self.page_size,
                slots_per_block=self.page_size,
                tensor_rows_per_block=(1,) * len(full_tensors),
            ),
            SGLangKVComponentGroup(
                name="swa",
                kv_tensors=swa_tensors,
                sliding_window_size=self.aligned_swa_window_size(),
                tokens_per_block=self.page_size,
                slots_per_block=self.page_size,
                tensor_rows_per_block=(1,) * len(swa_tensors),
            ),
        ]

    def _resolve_mamba_pool_tensors(self, mamba_pool) -> tuple[torch.Tensor, ...]:
        """Expose Mamba state as zero-copy LMCache tensors."""
        unified_buffer = getattr(mamba_pool, "_unified_buffer", None)
        sub_pool_name = getattr(mamba_pool, "_sub_pool_name", None)
        if unified_buffer is not None and sub_pool_name is not None:
            if unified_buffer.anchor_bytes(sub_pool_name) != 0:
                raise NotImplementedError(
                    "LMCache requires the unified Mamba pool to start at its "
                    "backing buffer base"
                )
            entry_bytes = unified_buffer.mamba_spec(sub_pool_name).entry_bytes()
            num_slots = mamba_pool._max_size + 1
            raw = unified_buffer._raw[: num_slots * entry_bytes]
            return (raw.view(num_slots, 1, entry_bytes),)

        tensors: list[torch.Tensor] = []
        for (
            field,
            state_tensor,
            slice_axis,
        ) in mamba_pool._iter_transfer_state_tensors():
            if slice_axis != 0:
                raise NotImplementedError(
                    f"LMCache MP does not support {field} state with slot "
                    f"slice_axis={slice_axis}"
                )
            for layer_idx in range(mamba_pool.num_mamba_layers):
                layer_tensor = state_tensor[layer_idx]
                if not layer_tensor.is_contiguous():
                    raise NotImplementedError(
                        f"LMCache MP requires contiguous {field} state for "
                        f"Mamba layer {layer_idx}"
                    )
                tensors.append(layer_tensor)
        return tuple(tensors)
