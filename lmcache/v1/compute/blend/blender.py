# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Sequence, Union, Any
import os
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata, LMCFlashAttnMetadata
from lmcache.v1.compute.blend.metadata import BLEND_MODES, LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        # layerwise_model 内部已兼容 Qwen2.5-VL 的层解析
        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)

        # 使用 layerwise_model 暴露的层与切片信息，避免硬编码 vllm_model.model.layers
        self.layers = self.layerwise_model.layers
        self.start_layer = getattr(self.layerwise_model, "start_layer", 0)
        self.end_layer = getattr(self.layerwise_model, "end_layer", len(self.layers))
        self.num_layers = self.end_layer - self.start_layer

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        blend_mode = getattr(config, "blend_mode", "") or ""
        if not blend_mode:
            blend_mode = "codecsight" if config.is_codecsight else "topk"
        if blend_mode not in BLEND_MODES:
            logger.warning(
                "Unknown blend_mode '%s', falling back to 'direct_reuse'. "
                "Valid modes: %s", blend_mode, BLEND_MODES,
            )
            blend_mode = "direct_reuse"
        self.blend_mode = blend_mode
        self.gop = max(int(config.GOP), 1)
        self.vlcache_recompute_ratio = float(
            getattr(config, "vlcache_recompute_ratio", 0.05)
        )
        self.vlcache_mode = str(
            getattr(config, "vlcache_mode", "per_frame")
        )
        logger.info("Blender blend_mode=%s, GOP=%d, vlcache_ratio=%.3f, vlcache_mode=%s",
                     self.blend_mode, self.gop, self.vlcache_recompute_ratio,
                     self.vlcache_mode)

        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
            blend_mode=self.blend_mode,
            GOP=config.GOP,
            vlcache_recompute_ratio=self.vlcache_recompute_ratio,
        )
        self.skip_ffn = False
        self.skip_ffn_only_codecsight = True
        # When True, blend_mode direct_reuse only runs layerwise GPU retrieve
        # (same KV load as enable_blending=False) and skips the redundant
        # layerwise_model.compute_layer pass that recomputes attention/FFN.
        self.direct_reuse_retrieve_only = True
        self._single_zero_idx: dict[torch.device, torch.Tensor] = {}
        # P2 ablation instrumentation (behind flags, default off -> no behavior change):
        #  LMCACHE_TIME_SELECTION=1 -> log "SELECT_TIME mode=.. ms=.." per selection
        #  LMCACHE_EQUAL_K=1        -> top-K refreshes the SAME #tokens as I-frame (equal-K)
        self._time_sel = os.environ.get("LMCACHE_TIME_SELECTION") == "1"
        self._equal_k = os.environ.get("LMCACHE_EQUAL_K") == "1"
        if config.extra_config is not None:
            self.skip_ffn = bool(config.extra_config.get("skip_ffn", False))
            self.skip_ffn_only_codecsight = bool(
                config.extra_config.get("skip_ffn_only_codecsight", True)
            )
            if "direct_reuse_retrieve_only" in config.extra_config:
                self.direct_reuse_retrieve_only = bool(
                    config.extra_config.get("direct_reuse_retrieve_only", True)
                )
        if self.skip_ffn:
            logger.warning(
                "FFN skip is enabled (only_codecsight=%s). This may reduce output quality.",
                self.skip_ffn_only_codecsight,
            )

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )
        # Batch-safety: each blend() call installs its own per-request metadata
        # here so concurrent requests in one prefill step do not clobber each
        # other. Defaults to the shared instance for any non-blend path.
        self._active_metadata = self.metadata
        self._rotary_by_layer = [
            self._get_rotary_emb(layer.self_attn) for layer in self.layers
        ]

        self.is_mrope = getattr(self.layerwise_model, "is_mrope", False)
        self.mrope_section = getattr(self.layerwise_model, "mrope_section", None)
        self._mrope_model_config = None
        if self.is_mrope:
            try:
                vllm_cfg = self.layerwise_model.vllm_model.config
                self._mrope_model_config = {
                    "image_token_id": getattr(vllm_cfg, "image_token_id", 151655),
                    "video_token_id": getattr(vllm_cfg, "video_token_id", 151656),
                    "vision_start_token_id": getattr(vllm_cfg, "vision_start_token_id", 151652),
                    "spatial_merge_size": getattr(
                        getattr(vllm_cfg, "vision_config", None),
                        "spatial_merge_size", 2),
                }
                logger.info("mRoPE blender initialized with config: %s", self._mrope_model_config)
            except Exception as e:
                logger.warning("Could not extract mRoPE config from model: %s", e)
                self.is_mrope = False

    def _get_rotary_emb(self, attn_layer):
        if hasattr(attn_layer, "rotary_emb"):
            return attn_layer.rotary_emb
        if hasattr(attn_layer, "rotary_emb_func"):
            return attn_layer.rotary_emb_func
        raise AttributeError("Attention layer does not expose rotary embedding module.")

    def _compute_mrope_positions(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Compute correct M-RoPE 3D positions for Qwen3-VL models.

        Falls back to 1D arange if the required metadata is unavailable.
        Returns a tensor of shape [3, num_tokens] for M-RoPE or [num_tokens] for 1D.
        """
        input_ids = self._active_metadata.input_ids
        image_grid_thw = self._active_metadata.image_grid_thw
        cfg = self._mrope_model_config

        if input_ids is None or cfg is None:
            logger.warning("M-RoPE metadata missing; falling back to 1D positions.")
            return torch.arange(num_tokens, device=device, dtype=torch.int64)

        input_ids_for_pos = input_ids[:num_tokens]

        image_token_id = cfg["image_token_id"]
        video_token_id = cfg["video_token_id"]
        vision_start_token_id = cfg["vision_start_token_id"]
        spatial_merge_size = cfg["spatial_merge_size"]

        if image_grid_thw is None:
            image_grid_thw = []

        # Flatten nested grid lists: each image has [[t, h, w]]
        flat_grid = []
        for entry in image_grid_thw:
            if isinstance(entry, (list, tuple)):
                if len(entry) > 0 and isinstance(entry[0], (list, tuple)):
                    flat_grid.extend(entry)
                else:
                    flat_grid.append(entry)
            else:
                flat_grid.append(entry)

        input_tokens_tensor = torch.tensor(input_ids_for_pos)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id
        ).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum())
        video_nums = int((vision_tokens == video_token_id).sum())

        # For Qwen3-VL: video frames are sent as individual images, so
        # video_grid_thw should be empty (each frame is an image entry).
        video_grid_thw_expanded: list = []

        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        image_index, video_index = 0, 0

        for _ in range(image_nums + video_nums):
            ed_image = len(input_ids_for_pos) + 1
            ed_video = len(input_ids_for_pos) + 1
            if remain_images > 0:
                try:
                    ed_image = input_ids_for_pos.index(image_token_id, st)
                except ValueError:
                    pass
            if remain_videos > 0:
                try:
                    ed_video = input_ids_for_pos.index(video_token_id, st)
                except ValueError:
                    pass

            sentinel = len(input_ids_for_pos) + 1
            if ed_image >= sentinel and ed_video >= sentinel:
                break

            if ed_image < ed_video:
                if image_index < len(flat_grid):
                    t, h, w = flat_grid[image_index]
                else:
                    logger.warning(
                        "image_grid_thw index %d out of range (len=%d); "
                        "falling back to 1D positions.",
                        image_index, len(flat_grid),
                    )
                    return torch.arange(num_tokens, device=device, dtype=torch.int64)
                image_index += 1
                remain_images -= 1
                ed = ed_image
            elif ed_video < sentinel:
                if video_index < len(video_grid_thw_expanded):
                    t, h, w = video_grid_thw_expanded[video_index]
                else:
                    return torch.arange(num_tokens, device=device, dtype=torch.int64)
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            else:
                break

            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            text_len = ed - st

            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if llm_pos_ids_list
                else 0
            )
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_ids_for_pos):
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if llm_pos_ids_list
                else 0
            )
            text_len = len(input_ids_for_pos) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        if not llm_pos_ids_list:
            return torch.arange(num_tokens, device=device, dtype=torch.int64)

        positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        return positions.to(device=device, dtype=torch.int64)

    # ------------------------------------------------------------------
    # Helpers shared by codecsight / vlcache selection paths
    # ------------------------------------------------------------------

    def _compute_hit_indices(self, effective_len: int, device: torch.device):
        """Return indices of cache-*hit* tokens (excluding gaps)."""
        gap_positions = getattr(
            self.gpu_connector, "current_gap_positions", None
        )
        if gap_positions is None or gap_positions.numel() == 0:
            return torch.arange(effective_len, device=device, dtype=torch.long)
        hit_mask = torch.ones(effective_len, device=device, dtype=torch.bool)
        if gap_positions.device != device or gap_positions.dtype != torch.long:
            gap_positions = gap_positions.to(device=device, dtype=torch.long)
        valid_gap = gap_positions[
            (gap_positions >= 0) & (gap_positions < effective_len)
        ]
        hit_mask[valid_gap] = False
        return torch.where(hit_mask)[0]

    def _codecsight_select(
        self,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """CodecSight I-frame selection using GOP / mm_positions."""
        gop = self.gop
        tokens_per_frame = int(self._active_metadata.tokens_per_frame or 0)
        mm_positions: Optional[Sequence[Any]] = self._active_metadata.mm_positions

        if mm_positions:
            selected_chunks: list[torch.Tensor] = []
            first_key_start: Optional[int] = None
            for frame_id, placeholder in enumerate(mm_positions):
                start = int(getattr(placeholder, "offset", 0))
                length = int(getattr(placeholder, "length", 0))
                if length <= 0 or start >= effective_len:
                    continue
                end = min(start + length, effective_len)
                if gop > 1 and (frame_id % gop) != 0:
                    continue
                if first_key_start is None:
                    first_key_start = start
                hit_in_range = hit_indices[
                    (hit_indices >= start) & (hit_indices < end)
                ]
                if hit_in_range.numel() > 0:
                    selected_chunks.append(hit_in_range)

            if selected_chunks:
                selected = torch.cat(selected_chunks, dim=0)
            else:
                selected = hit_indices.new_empty((0,))
            if selected.numel() == 0 and first_key_start is not None:
                if first_key_start < effective_len:
                    selected = torch.tensor(
                        [first_key_start], device=device, dtype=torch.long
                    )
        elif tokens_per_frame > 0:
            if gop > 1:
                frame_ids = hit_indices // tokens_per_frame
                selected = hit_indices[(frame_ids % gop) == 0]
            else:
                selected = hit_indices
            if selected.numel() == 0 and hit_indices.numel() > 0:
                selected = hit_indices[:1]
            elif selected.numel() == 0:
                selected = hit_indices
        elif gop > 1:
            selected = hit_indices[(hit_indices % gop) == 0]
            if selected.numel() == 0 and hit_indices.numel() > 0:
                selected = hit_indices[:1]
        else:
            selected = hit_indices
        return selected

    def _random_select(
        self,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Random-K control (P2 ablation): select the SAME NUMBER of tokens as
        the I-frame (codecsight) strategy would, but at random positions among
        the cache-hit tokens. Matched budget K isolates the I-frame *signal*
        from the *amount* of recomputation. Deterministic via a fixed seed so
        the ablation is reproducible.
        """
        k = int(self._codecsight_select(hit_indices, effective_len, device).numel())
        n = int(hit_indices.numel())
        if k <= 0 or n == 0:
            return hit_indices[:1] if n > 0 else hit_indices
        if k >= n:
            return hit_indices
        g = torch.Generator(device="cpu")
        g.manual_seed(1234 + effective_len)  # stable across layers/runs
        perm = torch.randperm(n, generator=g).to(device)
        sel = hit_indices[perm[:k]]
        sel, _ = torch.sort(sel)
        return sel

    def _vlcache_select(
        self,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """VLCache baseline: select image-token prefix for recomputation.

        Two sub-modes controlled by ``self.vlcache_mode``:
          - ``per_frame``: floor(r * T_i) from each frame (paper-faithful)
          - ``prefix``:    first r% of all image tokens concatenated
        Falls back to prefix-of-all-overlap when mm_positions is absent.
        """
        mm_positions: Optional[Sequence[Any]] = self._active_metadata.mm_positions
        r = self.vlcache_recompute_ratio

        if not mm_positions:
            logger.debug(
                "vlcache: mm_positions unavailable, falling back to "
                "prefix-of-all (effective_len=%d, r=%.4f)", effective_len, r,
            )
            prefix_len = max(1, int(effective_len * r))
            return torch.arange(
                min(prefix_len, effective_len), device=device, dtype=torch.long
            )

        logger.debug(
            "vlcache mode=%s: %d frames in mm_positions, effective_len=%d, r=%.4f",
            self.vlcache_mode, len(mm_positions), effective_len, r,
        )

        if self.vlcache_mode == "prefix":
            selected = self._vlcache_video_prefix(
                mm_positions, r, hit_indices, effective_len, device,
            )
        else:
            selected = self._vlcache_per_frame(
                mm_positions, r, hit_indices, effective_len, device,
            )

        logger.debug(
            "vlcache mode=%s selected %d tokens for recompute out of %d hit tokens",
            self.vlcache_mode, selected.numel(), hit_indices.numel(),
        )
        return selected

    def _vlcache_per_frame(
        self,
        mm_positions: Sequence[Any],
        r: float,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Paper-faithful: floor(r * T_i) prefix tokens from each frame."""
        selected: list[torch.Tensor] = []
        for placeholder in mm_positions:
            start = int(getattr(placeholder, "offset", 0))
            length = int(getattr(placeholder, "length", 0))
            if length <= 0 or start >= effective_len:
                continue
            end = min(start + length, effective_len)
            n = max(1, int((end - start) * r))
            frame_hit = hit_indices[
                (hit_indices >= start) & (hit_indices < start + n)
            ]
            if frame_hit.numel() > 0:
                selected.append(frame_hit)
        if selected:
            return torch.cat(selected)
        if hit_indices.numel() > 0:
            return hit_indices[:1]
        return hit_indices.new_empty((0,))

    def _vlcache_video_prefix(
        self,
        mm_positions: Sequence[Any],
        r: float,
        hit_indices: torch.Tensor,
        effective_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Whole-video prefix: first r% of all image tokens concatenated."""
        all_image: list[torch.Tensor] = []
        for placeholder in mm_positions:
            start = int(getattr(placeholder, "offset", 0))
            length = int(getattr(placeholder, "length", 0))
            if length <= 0 or start >= effective_len:
                continue
            end = min(start + length, effective_len)
            frame_hit = hit_indices[
                (hit_indices >= start) & (hit_indices < end)
            ]
            if frame_hit.numel() > 0:
                all_image.append(frame_hit)
        if not all_image:
            if hit_indices.numel() > 0:
                return hit_indices[:1]
            return hit_indices.new_empty((0,))
        all_image_t = torch.cat(all_image)
        budget = max(1, int(all_image_t.numel() * r))
        return all_image_t[:budget]

    # ------------------------------------------------------------------
    # Shared logic: apply index selection + rotary for selective modes
    # ------------------------------------------------------------------

    def _apply_selected_indices(
        self,
        selected_indices: torch.Tensor,
        effective_len: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        old_k: torch.Tensor,
        old_v: torch.Tensor,
        residual: torch.Tensor,
        attn_output: torch.Tensor,
        attn_metadata: LMCAttnMetadata,
        rotary,
        layer_id: int,
        mode_label: str,
    ):
        """Build imp_indices on first layer, then apply per-layer.

        Slices q/k/v/residual to the selected subset and updates
        attn_metadata (query side only -- cu_seqlens_k stays at full
        sequence length so the selected queries attend to the entire
        cached KV context, matching the official lmcache behaviour).
        """
        num_tokens = q.shape[0]

        if self._active_metadata.imp_indices is None:
            self._active_metadata.imp_indices = selected_indices
            if self._active_metadata.positions.ndim == 2:
                self._active_metadata.positions = self._active_metadata.positions[:, selected_indices]
            else:
                self._active_metadata.positions = self._active_metadata.positions[selected_indices]
            self._active_metadata.selection_effective_len = effective_len
            self._active_metadata.is_full_selection = (
                selected_indices.numel() == effective_len
            )
            logger.debug(
                "%s selected %d/%d tokens for recompute.",
                mode_label,
                int(selected_indices.numel()),
                num_tokens,
            )

        imp_indices = self._active_metadata.imp_indices
        assert imp_indices is not None
        sel_eff_len = int(self._active_metadata.selection_effective_len or 0)
        if sel_eff_len > 0 and effective_len >= sel_eff_len:
            layer_imp = imp_indices
            layer_positions = self._active_metadata.positions
        else:
            valid_mask = imp_indices < effective_len
            layer_imp = imp_indices[valid_mask]
            if self._active_metadata.positions.ndim == 2:
                layer_positions = self._active_metadata.positions[:, valid_mask]
            else:
                layer_positions = self._active_metadata.positions[valid_mask]

        if layer_imp.numel() == 0 and effective_len > 0:
            if q.device not in self._single_zero_idx:
                self._single_zero_idx[q.device] = torch.tensor(
                    [0], device=q.device, dtype=torch.long
                )
            layer_imp = self._single_zero_idx[q.device]
            if self._active_metadata.positions.ndim == 2:
                layer_positions = self._single_zero_idx[q.device].unsqueeze(0).expand(3, -1)
            else:
                layer_positions = self._single_zero_idx[q.device]
            self._active_metadata.imp_indices = layer_imp
            self._active_metadata.positions = layer_positions

        full_range = (
            self._active_metadata.is_full_selection
            and effective_len == num_tokens
            and sel_eff_len == num_tokens
        )
        if not full_range:
            attn_metadata.update_from_top_indices(layer_imp)
            k = k.index_select(0, layer_imp)
            v = v.index_select(0, layer_imp)
            q = q.index_select(0, layer_imp)
            residual = residual.index_select(0, layer_imp)
            attn_output = attn_output[: len(layer_imp)]

        q, k = rotary(layer_positions, q, k)

        old_k[layer_imp] = k
        old_v[layer_imp] = v
        return q, old_k, old_v, residual, attn_output, attn_metadata

    # ------------------------------------------------------------------
    # Main entry: process_qkv
    # ------------------------------------------------------------------

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ):
        """Dispatch to the appropriate blend strategy based on ``blend_mode``.

        Supported modes: direct_reuse, topk, codecsight, vlcache.
        """
        logger.debug("Blender is processing KV for layer %d", layer_id)
        try:
            old_k, old_v = self.gpu_connector.get_kv(layer_id)
        except ValueError:
            logger.warning(
                "KV cache for layer %s is not loaded into GPU buffer, skip blending.",
                layer_id,
            )
            return q, k, v, residual, attn_output, attn_metadata

        if attn_output is None:
            attn_output = torch.empty(
                q.shape, dtype=q.dtype, device=q.device,
            )

        # Initialize positions once per blend request.
        if self._active_metadata.positions is None:
            if self.is_mrope and self._mrope_model_config is not None:
                self._active_metadata.positions = self._compute_mrope_positions(
                    q.shape[0], q.device,
                )
                logger.info(
                    "Computed M-RoPE positions: shape=%s",
                    list(self._active_metadata.positions.shape),
                )
            else:
                self._active_metadata.positions = torch.arange(
                    q.shape[0], device=q.device, dtype=torch.int64
                )

        rotary = self._rotary_by_layer[layer_id]

        # ==============================================================
        # direct_reuse: no recomputation, just return cached KV
        # ==============================================================
        if self.blend_mode == "direct_reuse":
            q, _ = rotary(self._active_metadata.positions, q, k)
            return q, old_k, old_v, residual, attn_output, attn_metadata

        # ==============================================================
        # topk: L2-diff based selection (original check_layers path)
        #
        # Matches the official lmcache approach: slice q/k/v to the
        # selected subset but keep cu_seqlens_k at full sequence length
        # so each selected query attends to the entire cached KV.
        # ==============================================================
        if self.blend_mode == "topk":
            q, k = rotary(self._active_metadata.positions, q, k)
            write_indices = self._active_metadata.imp_indices

            if layer_id in self.common_metadata.check_layers:
                if self._time_sel:
                    torch.cuda.synchronize(); _t0 = time.perf_counter()
                diff_k = torch.sum(
                    (k.to(torch.float32) - old_k.to(torch.float32)) ** 2,
                    dim=[1],
                )
                total_len = diff_k.shape[0]
                assert self.common_metadata.recomp_ratios is not None
                if self._equal_k:
                    # equal-K: match the I-frame (codecsight) refresh budget
                    _hit = self._compute_hit_indices(total_len, k.device)
                    topk_num = int(
                        self._codecsight_select(_hit, total_len, k.device).numel())
                else:
                    topk_num = int(
                        total_len * self.common_metadata.recomp_ratios[0]
                    )
                logger.info(
                    "TOPK check layer=%d: total=%d, topk_num=%d, "
                    "diff_k min=%.6f max=%.6f mean=%.6f median=%.6f, "
                    "k_norm=%.4f old_k_norm=%.4f, "
                    "nonzero_diff=%d/%d",
                    layer_id, total_len, topk_num,
                    diff_k.min().item(), diff_k.max().item(),
                    diff_k.mean().item(), diff_k.median().item(),
                    k.norm().item(), old_k.norm().item(),
                    (diff_k > 1e-6).sum().item(), total_len,
                )
                top_indices = torch.topk(diff_k, k=topk_num).indices
                top_indices, _ = torch.sort(top_indices)
                if self._time_sel:
                    torch.cuda.synchronize()
                    logger.info("SELECT_TIME mode=topk layer=%d ms=%.4f k=%d",
                                layer_id, (time.perf_counter() - _t0) * 1000, topk_num)

                k, v = k[top_indices], v[top_indices]
                q = q[top_indices]
                residual = residual[top_indices]

                self._active_metadata.imp_indices = top_indices
                if self._active_metadata.positions.ndim == 2:
                    self._active_metadata.positions = self._active_metadata.positions[:, top_indices]
                else:
                    self._active_metadata.positions = self._active_metadata.positions[top_indices]
                attn_output = attn_output[:topk_num]
                attn_metadata.update_from_top_indices(top_indices)
                write_indices = top_indices

            if write_indices is not None:
                old_k[write_indices] = k
                old_v[write_indices] = v
                return q, old_k, old_v, residual, attn_output, attn_metadata
            return q, k, v, residual, attn_output, attn_metadata

        # ==============================================================
        # codecsight / vlcache: index-based selective recomputation
        # ==============================================================
        _MIN_BLEND_TOKENS = 128
        first_layer = self._active_metadata.imp_indices is None

        if first_layer:
            effective_len = min(q.shape[0], old_k.shape[0])
            if effective_len < _MIN_BLEND_TOKENS:
                logger.info(
                    "Cached prefix too short (%d < %d tokens) for %s, "
                    "falling back to direct_reuse",
                    effective_len, _MIN_BLEND_TOKENS, self.blend_mode,
                )
                q, _ = rotary(self._active_metadata.positions, q, k)
                return q, old_k, old_v, residual, attn_output, attn_metadata
            hit_indices = self._compute_hit_indices(effective_len, q.device)
            if self._time_sel:
                torch.cuda.synchronize(); _t0 = time.perf_counter()
            if self.blend_mode == "codecsight":
                selected = self._codecsight_select(
                    hit_indices, effective_len, q.device,
                )
            elif self.blend_mode == "random":
                selected = self._random_select(
                    hit_indices, effective_len, q.device,
                )
            else:
                selected = self._vlcache_select(
                    hit_indices, effective_len, q.device,
                )
            if self._time_sel:
                torch.cuda.synchronize()
                logger.info("SELECT_TIME mode=%s layer=%d ms=%.4f k=%d",
                            self.blend_mode, layer_id,
                            (time.perf_counter() - _t0) * 1000, int(selected.numel()))
            return self._apply_selected_indices(
                selected, effective_len, q, k, v, old_k, old_v,
                residual, attn_output, attn_metadata, rotary,
                layer_id, self.blend_mode,
            )

        # Subsequent layers: q/k/v/residual are already reduced to
        # the selected subset by compute_layer. Apply rotary and write
        # into the full KV cache at the stored indices.
        imp_indices = self._active_metadata.imp_indices
        q, k = rotary(self._active_metadata.positions, q, k)
        old_k[imp_indices] = k
        old_v[imp_indices] = v
        return q, old_k, old_v, residual, attn_output, attn_metadata

    # ------------------------------------------------------------------
    # Tier-2: batched selective recompute (LMCACHE_BATCHED_BLEND=1)
    # ------------------------------------------------------------------
    def process_qkv_batched(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        req_meta: list,
        kvcaches,
        cu_q: torch.Tensor,
    ):
        """Batched counterpart to process_qkv for the packed anchor tensor.

        RoPE is per-token (position-indexed), so it applies once on the packed
        q/k with concatenated positions -- identical to per-segment. Then, per
        request: scatter the freshly recomputed (RoPE'd) anchor K/V into the
        paged cache, gather the request's full S-token context (now including
        the fresh anchors), and concat across requests so attention sees one
        contiguous [ΣS] key tensor with cu_seqlens_k marking the boundaries.

        Returns (q_packed, packed_old_k, packed_old_v, attn_metadata).
        """
        rotary = self._rotary_by_layer[layer_id]

        # packed positions across all requests (1D [ΣA] or mRoPE [3, ΣA])
        pos0 = req_meta[0]["positions"]
        if pos0.ndim == 2:
            packed_pos = torch.cat([m["positions"] for m in req_meta], dim=1)
        else:
            packed_pos = torch.cat([m["positions"] for m in req_meta], dim=0)
        q, k = rotary(packed_pos, q, k)

        attn_core = self.layerwise_model.vllm_attn_layers[layer_id]
        nkv = int(getattr(attn_core, "num_kv_heads", None)
                  or getattr(attn_core, "num_key_value_heads"))
        hd = int(getattr(attn_core, "head_size", None)
                 or getattr(attn_core, "head_dim"))

        k = k.view(-1, nkv, hd)
        v = v.view(-1, nkv, hd)

        kv = kvcaches[layer_id]
        # flash-attn paged layout: [2, num_blocks, block_size, nkv, hd] (two-major)
        two_major = kv.shape[0] == 2
        if two_major:
            k_all = kv[0].view(-1, nkv, hd)
            v_all = kv[1].view(-1, nkv, hd)
        else:  # flash-infer: [num_blocks, 2, block_size, nkv, hd]
            k_all = kv[:, 0].reshape(-1, nkv, hd)
            v_all = kv[:, 1].reshape(-1, nkv, hd)

        old_k_segs, old_v_segs, S_list = [], [], []
        for i, m in enumerate(req_meta):
            a0, a1 = int(cu_q[i]), int(cu_q[i + 1])
            slot_full = m["slot_full"]
            anchor_local = m["anchor_local"]
            anchor_slots = slot_full[anchor_local]
            # scatter fresh anchor K/V into the paged cache (refresh the cache)
            k_all[anchor_slots] = k[a0:a1]
            v_all[anchor_slots] = v[a0:a1]
            # gather full context (now includes the fresh anchors)
            old_k_segs.append(k_all[slot_full])
            old_v_segs.append(v_all[slot_full])
            S_list.append(slot_full.shape[0])

        packed_old_k = torch.cat(old_k_segs, dim=0)
        packed_old_v = torch.cat(old_v_segs, dim=0)

        dev = q.device
        cu_k = torch.tensor(
            [0] + list(torch.tensor(S_list).cumsum(0)),
            dtype=torch.int32, device=dev,
        )
        attn_metadata = LMCFlashAttnMetadata(
            query_start_loc=cu_q.to(torch.int32),
            seq_lens=torch.tensor(S_list, device=dev),
            cu_seqlens_k=cu_k,
            max_query_len=int((cu_q[1:] - cu_q[:-1]).max()),
            max_seq_len=max(S_list),
        )
        return q, packed_old_k, packed_old_v, attn_metadata

    def blend_batched(self, requests: list, kvcaches, fetch_gens=None,
                      defer=False):
        """Orchestrate batched selective recompute for N requests in one forward.

        `requests`: list of per-request dicts, each with:
          - 'anchor_embeds': [A, hidden] embeddings of the anchor tokens
          - 'positions':      anchor RoPE positions, [A] or [3, A]
          - 'slot_full':       paged-cache flat slots for all S cached tokens, [S]
          - 'anchor_local':    anchor indices within 0..S, [A]

        Eager (defer=False): Phase-1 (load + RoPE-correct each request's full
        cached KV into the paged cache) must already have run -- this drives only
        the packed recompute to completion and returns None.

        Deferred (defer=True, Tier-2 Level-2): `fetch_gens` are the per-request
        retrieve_layer generators, each already primed ONCE (warmup -> gap
        positions set + layer-0 loaded, not yet sent). Returns a generator that
        `wait_for_layer_load` steps once per decoder layer: each step sends the
        current layer's KV for every request then recomputes that layer's packed
        anchors. Cadence mirrors ``blend_layer``: one post-warmup yield, then one
        yield per layer, then a final fetch-drain yield (num_layers + 2 yields,
        matching the 2x prime + num_layers wait_for_layer_load calls).
        """
        logger.info("blend_batched: packing %d request(s), total anchors=%d%s",
                    len(requests),
                    sum(int(r["anchor_local"].numel()) for r in requests),
                    " [deferred/overlap]" if defer else "")
        packed_embeds = torch.cat([r["anchor_embeds"] for r in requests], dim=0)
        req_meta = [
            {"positions": r["positions"],
             "slot_full": r["slot_full"],
             "anchor_local": r["anchor_local"]}
            for r in requests
        ]
        gen = self.layerwise_model.compute_layer_batched(
            packed_embeds, req_meta, kvcaches,
        )
        if not defer:
            for _ in range(self.num_layers):
                next(gen)
            return None

        num_layers = self.num_layers
        fetch_gens = list(fetch_gens or [])

        # PHASE FIX (2026-07-27, LMCACHE_DEFER_PHASE_FIX=0 restores the old, WRONG
        # phasing for A/B). The old driver was OFF BY ONE against the connector's
        # 3-stage pipeline and that is why the deferred path produced different
        # output from eager (7/18 keys, 0.45-1.81 nats, one answer flip; jobs
        # 15001729 / 15002005 / 15002109 / 15002110 / 15002139 -- the last of those
        # ran the deferred CODE with EAGER TIMING and still failed, proving the bug
        # is here and not in the interleaving).
        #
        # Why off by one: gpu_connector.batched_to_gpu stores layer i-2 at iteration
        # i (`single_layer_kv_transfer(buffer_mapping[i-2] -> kvcaches[i-2])`), and
        # retrieve_layer yields BEFORE _send_layer. So the fetch gen's (L+2)th next
        # performs send(L), after which the PAGED cache holds only layers 0..L-1.
        # The old loop then recomputed layer L against a layer that had not been
        # written yet. Eager never hit this because it drains every layer before
        # blend_batched runs.
        #
        # Fix = re-phase, not extra work: advance the fetch gens ONE extra time
        # before the loop and drop the trailing drain. Totals are unchanged --
        # fetch gens still get 1 (caller warmup) + 1 + num_layers = num_layers + 2
        # nexts, and the driver still yields num_layers + 2 times, so
        # wait_for_layer_load's cadence is untouched. Now at loop step L the gens
        # have had L+3 nexts -> send(L+1) done -> paged holds 0..L -> the recompute
        # of layer L reads a layer that is actually there.
        _phase_fix = os.environ.get("LMCACHE_DEFER_PHASE_FIX", "1") == "1"

        def _driver():
            # Post-warmup handshake (mirrors blend_layer's first yield). Fetches
            # are already warmed up by the caller; nothing to send yet.
            yield
            if _phase_fix:
                # Extra prime: push the store stage one layer ahead of the recompute.
                for fg in fetch_gens:
                    next(fg)
            for _ in range(num_layers):
                for fg in fetch_gens:
                    next(fg)
                next(gen)
                yield
            if not _phase_fix:
                # Old trailing drain (each gen's (num_layers+2)th next).
                for fg in fetch_gens:
                    next(fg)
            yield

        d = _driver()
        next(d)  # prime 1: advance past the post-warmup handshake
        next(d)  # prime 2: run layer 0 (stay one layer ahead of prefill)
        return d

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to orchestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwise retrieve + blending.
        """
        # TODO(Jiayi): store is currently not included in this function
        # Capture this request's metadata; re-installed before each per-layer
        # step below so interleaved concurrent generators (batched prefill in
        # the deferred/async path) never read each other's selection state.
        md = self._active_metadata
        check_layers = self.common_metadata.check_layers
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        deepstack_input_embeds = kwargs.pop("deepstack_input_embeds", None)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        # warmup retriever
        warmup_retrieved = next(layerwise_retriever)
        has_retrieved_tokens = False
        if warmup_retrieved is not None:
            if torch.is_tensor(warmup_retrieved):
                has_retrieved_tokens = int(warmup_retrieved.item()) > 0
            else:
                has_retrieved_tokens = int(warmup_retrieved) > 0
        yield

        if not has_retrieved_tokens:
            logger.debug("No retrievable tokens in layerwise retrieve; skip blending compute.")
            for _ in range(self.num_layers):
                next(layerwise_retriever)
                yield
            next(layerwise_retriever)
            md.clean()
            yield
            return

        if self.blend_mode == "direct_reuse" and self.direct_reuse_retrieve_only:
            logger.info(
                "direct_reuse: retrieve-only path (skip layerwise compute_layer); "
                "KV load matches non-blending retrieve_layer."
            )
            for _ in range(self.num_layers):
                next(layerwise_retriever)
                yield
            next(layerwise_retriever)
            md.clean()
            yield
            return

        layerwise_model_executor = self.layerwise_model.compute_layer(
            check_layers, tokens,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
        )
        for _ in range(self.num_layers):
            self._active_metadata = md
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)

        md.clean()
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        defer: bool = False,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.

        If ``defer`` is True, prime the layerwise generator (filling the
        internal load pipeline, mirroring retrieve_layer's 2x prime) and
        RETURN it instead of draining it. The caller then steps it once per
        decoder layer from ``wait_for_layer_load`` so the per-layer KV load
        overlaps the previous layer's prefill compute. Returns None in the
        eager path.
        """

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()
        logger.info("enter blend (defer=%s)", defer)
        # Per-request metadata: isolate this request's blend state so concurrent
        # requests batched in the same prefill step cannot collide on shared
        # state. blend_layer re-installs this before each layer step.
        md = LMCBlendMetadata(imp_indices=None, attn_mask=None, positions=None)
        self._active_metadata = md
        tokens_per_frame = kwargs.get("tokens_per_frame")
        if tokens_per_frame is not None:
            self._active_metadata.tokens_per_frame = int(tokens_per_frame)
        mm_positions = kwargs.get("mm_positions")
        if mm_positions is not None:
            self._active_metadata.mm_positions = mm_positions
        image_grid_thw = kwargs.get("image_grid_thw")
        if image_grid_thw is not None:
            self._active_metadata.image_grid_thw = image_grid_thw
        if isinstance(tokens, torch.Tensor):
            self._active_metadata.input_ids = tokens.tolist()
        else:
            self._active_metadata.input_ids = list(tokens)

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        if defer:
            # Prime twice to fill the 3-stage load pipeline (same count as the
            # non-blend retrieve_layer deferral). The remaining num_layers
            # steps are driven by wait_for_layer_load during the forward.
            next(layerwise_blender)
            next(layerwise_blender)
            return layerwise_blender

        # +2 is for the handshake/closing process with the retriever at both the beginning and end.
        for _ in range(self.num_layers + 2):
            next(layerwise_blender)
        return None
