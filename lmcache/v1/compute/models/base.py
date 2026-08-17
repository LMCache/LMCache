# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import os

# Third Party
from torch import nn
import torch

# First Party
from lmcache.v1.compute.attention.utils import infer_attn_backend_from_vllm
from lmcache.v1.compute.positional_encoding import get_fused_rope

# TODO(Jiayi): A few things need to be tested/supported:
# TP, PP, Multimodal

from lmcache.logging import init_logger

logger = init_logger(__name__)

def _get_attr_rec(obj, dotted: str) -> Optional[object]:
    cur = obj
    for name in dotted.split("."):
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def _resolve_decoder_layers(vllm_model: nn.Module) -> Tuple[nn.Module, nn.ModuleList]:
    """
    Return (decoder_root, layers) where `layers` is the ModuleList of transformer blocks.
    Compatible with LLaMA/Qwen3 and Qwen2.5-VL structures.
    """
    # Common paths first
    candidates = [
        "model.layers",
        "language_model.model.layers",   # Qwen2.5-VL
        "transformer.layers",
        "backbone.model.layers",
        "base_model.model.layers",
        "model.decoder.layers",
    ]
    for path in candidates:
        layers = _get_attr_rec(vllm_model, path)
        if isinstance(layers, nn.ModuleList):
            # decoder_root = path without final .layers
            parent_path = ".".join(path.split(".")[:-1]) or ""
            decoder_root = _get_attr_rec(vllm_model, parent_path) if parent_path else vllm_model
            return decoder_root, layers

    # Fallback: breadth-first search named_children to find module with layers attribute
    for name, mod in vllm_model.named_modules():
        if hasattr(mod, "layers") and isinstance(mod.layers, nn.ModuleList):
            return mod, mod.layers

    raise AttributeError(
        "Cannot locate decoder layers. Tried common paths for LLaMA/Qwen families."
    )


def _pick(attr_owner, *names, default=None):
    for n in names:
        if hasattr(attr_owner, n):
            return getattr(attr_owner, n)
    return default


class LMCBaseModel(nn.Module, ABC):
    def __init__(
        self,
        vllm_model,
        blender,
        enable_sparse: bool = False,
    ):
        super().__init__()
        # Allow subclasses to patch/monkeypatch external deps before heavy init.
        self._maybe_patch_mm_inputs()
        self.vllm_model = vllm_model

        # --- decoder layers extraction ---
        self.decoder_root, self.layers = _resolve_decoder_layers(vllm_model)
        self.num_layers = len(self.layers)

        # vLLM sometimes has wrappers that attach start_layer/end_layer to decoder_root
        self.start_layer = getattr(self.decoder_root, "start_layer", 0)
        self.end_layer = getattr(self.decoder_root, "end_layer", self.num_layers)

        # --- attention backend parsing ---
        self.vllm_attn_layers = []
        self.lmc_attn_layers = []
        for i in range(self.num_layers):
            block = self.layers[i]
            self_attn = block.self_attn
            # vLLM common pattern is self_attn.attn; if not present, use self_attn itself
            vllm_attn_mod = getattr(self_attn, "attn", self_attn)
            self.vllm_attn_layers.append(vllm_attn_mod)
            self.lmc_attn_layers.append(
                infer_attn_backend_from_vllm(vllm_attn_mod, enable_sparse)
            )

        # --- Rotary Embedding (field name compatibility) ---
        # Take rotary_emb from the 0th layer
        rotary = self.layers[0].self_attn.rotary_emb
        head_dim = _pick(rotary, "head_size", "head_dim")
        max_position_embeddings = _pick(rotary, "max_position_embeddings", "max_seq_len_cached", default=8192)
        # Qwen common base field may also be called rope_theta
        base = _pick(rotary, "base", "rope_theta", default=10000.0)
        is_neox_style = getattr(rotary, "is_neox_style", True)
        dtype = getattr(rotary, "dtype", torch.get_default_dtype())
        rope_scaling = getattr(rotary, "rope_scaling", None)

        self.is_mrope = hasattr(rotary, "mrope_section") and rotary.mrope_section is not None
        self.mrope_section = getattr(rotary, "mrope_section", None)

        if self.is_mrope:
            logger.info(
                "Detected mRoPE model (mrope_section=%s). "
                "Disabling 1D fused RoPE; will use mRoPE-aware position handling.",
                self.mrope_section,
            )
            self.fused_rotary_emb = None
        else:
            self.fused_rotary_emb = get_fused_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=max_position_embeddings,
                base=base,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                dtype=dtype,
            )

        # NOTE(Jiayi): better not to pass the blender in init
        # if we want to make this LMCModel more general.
        self.blender = blender

    @abstractmethod
    def _process_qkv(self, q, k, v, layer):
        """Process QKV tensors. Model-specific implementation."""
        pass

    def _project_qkv(self, layer, hidden_states: torch.Tensor):
        """
        Args:
            layer:  The transformer block/module that owns self_attn.
            hidden_states: [*, hidden_size] input to the QKV projection.
        """
        out = layer.self_attn.qkv_proj(hidden_states)
        # Compatible with (tensor, None) or just tensor
        if isinstance(out, tuple):
            qkv = out[0]
        else:
            qkv = out

        # Prefer explicit size
        q_size = getattr(layer.self_attn, "q_size", None)
        kv_size = getattr(layer.self_attn, "kv_size", None)

        num_heads = _pick(self.vllm_attn_layers[layer.layer_id if hasattr(layer, "layer_id") else 0],
                          "num_heads", default=None)
        num_kv_heads = _pick(self.vllm_attn_layers[layer.layer_id if hasattr(layer, "layer_id") else 0],
                             "num_kv_heads", "num_key_value_heads", default=None)
        head_size = _pick(self.vllm_attn_layers[layer.layer_id if hasattr(layer, "layer_id") else 0],
                          "head_size", "head_dim", default=None)

        if q_size is None or kv_size is None:
            # Fallback: infer from heads
            assert num_heads is not None and num_kv_heads is not None and head_size is not None, \
                "Cannot infer Q/K/V split sizes; missing heads/head_size info."
            q_size = num_heads * head_size
            kv_size = num_kv_heads * head_size

        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        return q, k, v

    def _maybe_patch_mm_inputs(self):
        """
        Hook for subclasses that need to patch vLLM multimodal helpers
        before initialization (no-op by default).
        """
        return

    def rope_cache_to_device(self, device: torch.device):
        self.fused_rotary_emb.rope_cache_to_device(device)

    @torch.compile
    def compute_layer(
        self,
        check_layers,
        input_ids: torch.Tensor,
        page_stream=None,
        inputs_embeds: "Optional[torch.Tensor]" = None,
        deepstack_input_embeds: "Optional[torch.Tensor]" = None,
        **kwargs,
    ):
        timing = True
        input_ids = input_ids.cuda()

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.vllm_model.get_input_embeddings(input_ids)

        residual = None
        attn_output = None
        logger.info(f"input token ids shape is {input_ids.shape}")
        # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
        attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
            input_ids=input_ids,
        )

        stream = page_stream if page_stream is not None else torch.cuda.current_stream()

        for layer_idx, layer in enumerate(self.layers[self.start_layer:self.end_layer]):
            pre_attn_events = None
            attn_events = None
            post_attn_events = None
            ffn_events = None
            total_events = None
            # Pre-LN/residual before Self Attention
            with torch.cuda.stream(stream):
                if timing is not None:
                    total_start = torch.cuda.Event(enable_timing=True)
                    total_end = torch.cuda.Event(enable_timing=True)
                    total_start.record(stream)
                if timing is not None:
                    pre_attn_start = torch.cuda.Event(enable_timing=True)
                    pre_attn_end = torch.cuda.Event(enable_timing=True)
                    pre_attn_start.record(stream)
                if residual is None:
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = layer.input_layernorm(hidden_states, residual)

                # QKV projection (compatible with multiple implementations)
                q, k, v = self._project_qkv(layer, hidden_states)

                # Model-specific QKV processing (e.g., q_norm/k_norm in Qwen3)
                q, k, v = self._process_qkv(q, k, v, layer)

                # Pass to blender
                q, k, v, residual, attn_output, attn_metadata = self.blender.process_qkv(
                    q, k, v, residual, self.start_layer + layer_idx, attn_output, attn_metadata
                )
                if timing is not None:
                    pre_attn_end.record(stream)
                    pre_attn_events = (pre_attn_start, pre_attn_end)

                # Take attention dimensions for this layer (extracted from vllm_attn_layers, stored in __init__)
                attn_core = self.vllm_attn_layers[self.start_layer + layer_idx]
                num_heads = _pick(attn_core, "num_heads")
                num_kv_heads = _pick(attn_core, "num_kv_heads", "num_key_value_heads")
                head_size = _pick(attn_core, "head_size", "head_dim")

                # Reshape and forward
                q = q.view(-1, num_heads, head_size)
                k = k.view(-1, num_kv_heads, head_size)
                v = v.view(-1, num_kv_heads, head_size)
                attn_output = (attn_output if attn_output is not None else torch.zeros_like(q)).view(
                    -1, num_heads, head_size
                )

                if timing is not None:
                    attn_start = torch.cuda.Event(enable_timing=True)
                    attn_end = torch.cuda.Event(enable_timing=True)
                    attn_start.record(stream)
                attn_output = self.lmc_attn_layers[self.start_layer + layer_idx].forward_contiguous(
                    q, k, v, attn_output, attn_metadata
                )
                if timing is not None:
                    attn_end.record(stream)
                    attn_events = (attn_start, attn_end)

                attn_output = attn_output.view(-1, num_heads * head_size)
                # K/V reshape back to flat for downstream (even if not directly used, keep consistent with original implementation)
                _ = k.view(-1, num_kv_heads * head_size)
                _ = v.view(-1, num_kv_heads * head_size)

                # # if layer_idx > check_layer, skip the following operations
                # if layer_idx > check_layers[-1]: 
                #     yield
                #     continue 

                # Output projection
                if timing is not None:
                    post_attn_start = torch.cuda.Event(enable_timing=True)
                    post_attn_end = torch.cuda.Event(enable_timing=True)
                    post_attn_start.record(stream)
                hidden_states, _ = layer.self_attn.o_proj(attn_output)

                # FFN
                hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
                if timing is not None:
                    post_attn_end.record(stream)
                    post_attn_events = (post_attn_start, post_attn_end)

                if timing is not None:
                    ffn_start = torch.cuda.Event(enable_timing=True)
                    ffn_end = torch.cuda.Event(enable_timing=True)
                    ffn_start.record(stream)
                skip_ffn = bool(getattr(self.blender, "skip_ffn", False))
                if skip_ffn and bool(getattr(self.blender, "skip_ffn_only_codecsight", True)):
                    skip_ffn = getattr(self.blender, "blend_mode", "") in ("codecsight",)
                if skip_ffn:
                    hidden_states = torch.zeros_like(hidden_states)
                else:
                    hidden_states = layer.mlp(hidden_states)

                # Deepstack injection for Qwen3-VL: add multi-scale features
                # at early decoder layers.
                if deepstack_input_embeds is not None:
                    global_layer_idx = self.start_layer + layer_idx
                    if deepstack_input_embeds.ndim == 3 and global_layer_idx < deepstack_input_embeds.shape[0]:
                        ds_slice = deepstack_input_embeds[global_layer_idx]
                        if residual is not None:
                            imp_indices = getattr(self.blender._active_metadata, "imp_indices", None)
                            if imp_indices is not None and ds_slice.shape[0] != residual.shape[0]:
                                ds_slice = ds_slice.index_select(0, imp_indices)
                            if ds_slice.shape[0] == residual.shape[0]:
                                residual = residual + ds_slice
                if timing is not None:
                    ffn_end.record(stream)
                    ffn_events = (ffn_start, ffn_end)
                    total_end.record(stream)
                    total_events = (total_start, total_end)

            if timing is not None:
                # Synchronize once per layer so all CUDA events are complete.
                stream.synchronize()
                layer_no = self.start_layer + layer_idx
                pre_ms = pre_attn_events[0].elapsed_time(pre_attn_events[1])
                attn_ms = attn_events[0].elapsed_time(attn_events[1])
                post_ms = post_attn_events[0].elapsed_time(post_attn_events[1])
                ffn_ms = ffn_events[0].elapsed_time(ffn_events[1])
                total_ms = total_events[0].elapsed_time(total_events[1])
                logger.info(
                    "layer=%d pre_attn_ms=%.3f attention_ms=%.3f post_attn_ms=%.3f ffn_ms=%.3f total_ms=%.3f",
                    layer_no,
                    pre_ms,
                    attn_ms,
                    post_ms,
                    ffn_ms,
                    total_ms,
                )

            yield

    # ------------------------------------------------------------------
    # Tier-2: batched selective recompute (LMCACHE_BATCHED_BLEND=1)
    # ------------------------------------------------------------------
    def compute_layer_batched(
        self,
        packed_embeds: torch.Tensor,
        req_meta: list,
        kvcaches,
    ):
        """Packed selective-recompute forward for N requests in ONE pass.

        See codecsight-bench/TIER2_BATCHED_BLEND.md. The token-wise ops (LN, QKV,
        o_proj, FFN) run on the concatenated [ΣA, hidden] anchor tensor; attention
        is a single varlen call whose cu_seqlens isolate each request (validated
        bitwise-equal to N serial calls in tests/tier2_batched_attn_parity.py).

        req_meta: list of per-request dicts, each with:
          - 'positions':    anchor RoPE positions, [A] (1D) or [3, A] (mRoPE)
          - 'slot_full':     paged-cache flat slots for all S cached tokens, [S]
          - 'anchor_local':  anchor indices within 0..S, [A]
        Phase-1 must have already loaded + RoPE-corrected each request's full
        cached KV into the paged cache (the non-anchor context attended here).

        Generator: yields once per layer (mirrors compute_layer's cadence so the
        caller can step Phase-1 retrievers in lockstep if pipelining the load).

        Optional ``LMCACHE_RECOMPUTE_SUBTIMING=1``: CUDA-event spans per sub-op,
        summed across all layers, logged once as ``[recompute-sub]`` after a
        single device sync at the end of the last layer (same pattern as
        ``[blend-timing]`` — does not sync per layer).
        """
        hidden_states = packed_embeds.cuda()
        residual = None
        A_list = [int(m["positions"].shape[-1]) for m in req_meta]
        cu_q = torch.tensor(
            [0] + list(torch.tensor(A_list).cumsum(0)),
            dtype=torch.int32, device=hidden_states.device,
        )

        # Sub-phase timing (default OFF). Keys match NVTX range names.
        _sub = (
            os.environ.get("LMCACHE_RECOMPUTE_SUBTIMING", "0") == "1"
            and torch.cuda.is_available()
        )
        _phases = ("ln1", "qkv_proj", "rope_gather_scatter", "attention",
                   "post_attn", "ffn")
        _evs = {k: [] for k in _phases} if _sub else None

        def _span_begin():
            if not _sub:
                return None
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            return s

        def _span_end(name, start):
            if not _sub or start is None:
                return
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            _evs[name].append((start, e))

        layers = self.layers[self.start_layer:self.end_layer]
        n_layers = len(layers)
        for layer_idx, layer in enumerate(layers):
            global_layer = self.start_layer + layer_idx
            # NVTX only by default -- no CUDA events / no torch.cuda.synchronize()
            # here, so (unlike LMCACHE_LAYER_LOAD_TIMING on the fetch side) this
            # cannot reintroduce a per-layer device sync. push/pop are
            # non-blocking markers nsys reads off the existing timeline.
            torch.cuda.nvtx.range_push(f"recompute_layer_{global_layer}")

            torch.cuda.nvtx.range_push("ln1")
            _s = _span_begin()
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            _span_end("ln1", _s)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("qkv_proj")
            _s = _span_begin()
            q, k, v = self._project_qkv(layer, hidden_states)
            q, k, v = self._process_qkv(q, k, v, layer)
            _span_end("qkv_proj", _s)
            torch.cuda.nvtx.range_pop()

            # Blender owns RoPE + per-request KV gather/scatter/concat.
            torch.cuda.nvtx.range_push("rope_gather_scatter")
            _s = _span_begin()
            q, old_k, old_v, attn_metadata = self.blender.process_qkv_batched(
                q, k, v, global_layer, req_meta, kvcaches, cu_q,
            )
            _span_end("rope_gather_scatter", _s)
            torch.cuda.nvtx.range_pop()

            attn_core = self.vllm_attn_layers[global_layer]
            num_heads = _pick(attn_core, "num_heads")
            num_kv_heads = _pick(attn_core, "num_kv_heads", "num_key_value_heads")
            head_size = _pick(attn_core, "head_size", "head_dim")

            q = q.view(-1, num_heads, head_size)
            old_k = old_k.view(-1, num_kv_heads, head_size)
            old_v = old_v.view(-1, num_kv_heads, head_size)
            attn_output = torch.zeros_like(q)

            torch.cuda.nvtx.range_push("attention")
            _s = _span_begin()
            attn_output = self.lmc_attn_layers[global_layer].forward_contiguous(
                q, old_k, old_v, attn_output, attn_metadata
            )
            _span_end("attention", _s)
            torch.cuda.nvtx.range_pop()
            attn_output = attn_output.view(-1, num_heads * head_size)

            torch.cuda.nvtx.range_push("post_attn")
            _s = _span_begin()
            hidden_states, _ = layer.self_attn.o_proj(attn_output)
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            _span_end("post_attn", _s)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("ffn")
            _s = _span_begin()
            skip_ffn = bool(getattr(self.blender, "skip_ffn", False))
            if skip_ffn and bool(getattr(self.blender, "skip_ffn_only_codecsight", True)):
                skip_ffn = getattr(self.blender, "blend_mode", "") in ("codecsight",)
            if skip_ffn:
                hidden_states = torch.zeros_like(hidden_states)
            else:
                hidden_states = layer.mlp(hidden_states)
            _span_end("ffn", _s)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()  # recompute_layer_{global_layer}

            # One sync after the last layer's events are recorded, then log.
            if _sub and layer_idx == n_layers - 1:
                torch.cuda.synchronize()
                totals = {
                    k: sum(a.elapsed_time(b) for a, b in pairs)
                    for k, pairs in _evs.items()
                }
                total = sum(totals.values()) or 1.0
                n_anch = sum(A_list)
                n_ctx = sum(int(m["slot_full"].shape[0]) for m in req_meta)
                logger.info(
                    "[recompute-sub] N=%d anchors=%d ctx_tokens=%d layers=%d | "
                    "ln1=%.2fms qkv=%.2fms rope_gather_scatter=%.2fms "
                    "attn=%.2fms post_attn=%.2fms ffn=%.2fms | sum=%.2fms "
                    "(rgs=%.0f%% attn=%.0f%% ffn=%.0f%% qkv+post=%.0f%%)",
                    len(req_meta), n_anch, n_ctx, n_layers,
                    totals["ln1"], totals["qkv_proj"],
                    totals["rope_gather_scatter"], totals["attention"],
                    totals["post_attn"], totals["ffn"], total,
                    100.0 * totals["rope_gather_scatter"] / total,
                    100.0 * totals["attention"] / total,
                    100.0 * totals["ffn"] / total,
                    100.0 * (totals["qkv_proj"] + totals["post_attn"]) / total,
                )

                # Finer split INSIDE rope_gather_scatter -- the only non-GEMM
                # phase above, and so the only one that can be REMOVED rather
                # than merely made cheaper. Drained here, after the sync that
                # already happened, so it costs no extra barrier.
                rgs = getattr(self.blender, "take_rgs_spans", lambda: {})()
                if rgs:
                    rgs_tot = {
                        k: sum(a.elapsed_time(b) for a, b in pairs)
                        for k, pairs in rgs.items()
                    }
                    rgs_sum = sum(rgs_tot.values()) or 1.0
                    logger.info(
                        "[rgs-parts] active=1 N=%d anchors=%d ctx_tokens=%d "
                        "layers=%d | rope=%.2fms scatter=%.2fms gather=%.2fms "
                        "cat=%.2fms | sum=%.2fms (gather+cat=%.0f%% of rgs) "
                        "parent_rgs=%.2fms",
                        len(req_meta), n_anch, n_ctx, n_layers,
                        rgs_tot.get("rope", 0.0), rgs_tot.get("scatter", 0.0),
                        rgs_tot.get("gather", 0.0), rgs_tot.get("cat", 0.0),
                        rgs_sum,
                        100.0 * (rgs_tot.get("gather", 0.0)
                                 + rgs_tot.get("cat", 0.0)) / rgs_sum,
                        totals["rope_gather_scatter"],
                    )
            yield
