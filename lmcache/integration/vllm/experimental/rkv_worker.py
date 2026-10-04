# SPDX-License-Identifier: Apache-2.0
"""Worker-local R-KV query capture and KV compaction."""

from __future__ import annotations

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.integration.vllm.experimental.physical_kv_view import (
    apply_physical_kv_view,
)

class RKVWorker:
    """Capture recent queries and compact private FlashAttention KV in place."""

    def __init__(
        self,
        budget: int,
        *,
        buffer: int = 128,
        window_size: int = 8,
        kernel_size: int = 7,
        mix_lambda: float = 0.1,
        retain_ratio: float = 0.1,
        score_chunk_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if window_size <= 0:
            raise ValueError("R-KV window_size must be positive")
        if budget <= window_size:
            raise ValueError("R-KV budget must exceed window_size")
        if buffer < window_size:
            raise ValueError("R-KV buffer must be at least window_size")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("R-KV kernel_size must be a positive odd integer")
        if not 0.0 <= mix_lambda <= 1.0:
            raise ValueError("R-KV mix_lambda must be in [0, 1]")
        if not 0.0 < retain_ratio <= 1.0:
            raise ValueError("R-KV retain_ratio must be in (0, 1]")
        if score_chunk_bytes <= 0:
            raise ValueError("R-KV score_chunk_bytes must be positive")

        try:
            from rkv import R1KV
        except ImportError as exc:
            raise ImportError(
                "R-KV is enabled but the optional 'rkv' package is not installed"
            ) from exc

        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.score_chunk_bytes = score_chunk_bytes
        self._policy = R1KV(
            budget=budget,
            window_size=window_size,
            kernel_size=kernel_size,
            mix_lambda=mix_lambda,
            retain_ratio=retain_ratio,
            buffer=buffer,
        )
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._block_size = 0

        self._query_rings: dict[str, torch.Tensor] = {}
        self._query_slots: dict[str, int] = {}
        self._free_query_slots: list[int] = []
        self._query_counts: dict[str, int] = {}
        self._next_query_slot = 0
        self._query_ring_width = 0
        self._query_last_indices: torch.Tensor | None = None
        self._query_write_indices: torch.Tensor | None = None

        self._request_ids: list[str] = []
        self._seq_lens: list[int] | None = None
        self._block_table: torch.Tensor | None = None
        self._is_genuine_decode: list[bool] | None = None
        self._num_decoded_tokens: list[int] | None = None
        self._num_new_tokens: list[int] | None = None
        self._compacted_requests: set[str] = set()
        self._query_hooks_installed = False

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not kv_caches:
            raise ValueError("R-KV requires KV caches")

        block_size = 0
        for layer_name, kv_cache in kv_caches.items():
            if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
                raise ValueError(
                    "R-KV requires FlashAttention KV shaped "
                    "[num_blocks, 2, block_size, kv_heads, head_dim]"
                )
            if kv_cache.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("R-KV requires FP16 or BF16 KV cache")
            if kv_cache.device.type != "cuda":
                raise ValueError("R-KV requires CUDA-resident KV cache")
            if not kv_cache.is_contiguous():
                raise ValueError("R-KV MVP requires contiguous NHD KV layout")
            if block_size and kv_cache.shape[2] != block_size:
                raise ValueError("R-KV requires one shared KV block size")
            block_size = kv_cache.shape[2]

        self._kv_caches = dict(kv_caches)
        self._layer_names = list(kv_caches)
        self._block_size = block_size

    def reset(self) -> None:
        self._query_rings.clear()
        self._query_slots.clear()
        self._free_query_slots.clear()
        self._query_counts.clear()
        self._next_query_slot = 0
        self._query_ring_width = 0
        self._compacted_requests.clear()
        self._clear_step()

    def drop_requests(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            slot = self._query_slots.pop(request_id, None)
            if slot is not None:
                self._free_query_slots.append(slot)
            self._query_counts.pop(request_id, None)
            self._compacted_requests.discard(request_id)

    def prepare_forward(
        self,
        forward_context: Any,
        request_states: list[Any],
    ) -> None:
        if not request_states:
            self._clear_step()
            return

        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict) or not attn_metadata:
            raise ValueError("R-KV MVP requires eager attention metadata")

        representative = next(iter(attn_metadata.values()))
        num_reqs = len(request_states)
        if representative.query_start_loc.shape[0] != num_reqs + 1:
            raise ValueError("R-KV request/attention row count mismatch")

        by_first_block: dict[int, Any] = {}
        for state in request_states:
            first_block = int(state.first_block_id)
            if first_block in by_first_block:
                raise ValueError("R-KV requires private, uniquely-owned KV blocks")
            by_first_block[first_block] = state

        # vLLM's model runner already owns the authoritative paged-KV block
        # table. The connector only needs a stable row identity to align
        # scheduler facts after persistent-batch condense/reordering.
        block_table = representative.block_table
        first_blocks = block_table[:num_reqs, 0].tolist()
        try:
            ordered_states = [by_first_block[block_id] for block_id in first_blocks]
        except KeyError as exc:
            raise ValueError("R-KV could not match worker rows to requests") from exc

        query_lens = [int(state.num_new_tokens) for state in ordered_states]
        if any(state.resident_kv_tokens is None for state in ordered_states):
            raise RuntimeError("R-KV scheduler metadata is missing resident KV length")
        physical_seq_lens = [
            int(state.resident_kv_tokens) for state in ordered_states
        ]
        if any(
            query_len <= 0 or physical_len < query_len
            for query_len, physical_len in zip(
                query_lens, physical_seq_lens, strict=True
            )
        ):
            raise ValueError("R-KV received invalid physical/query lengths")

        seq_lens = representative.seq_lens.clone()
        physical_seq_lens_gpu = torch.as_tensor(
            physical_seq_lens,
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )
        seq_lens[:num_reqs] = physical_seq_lens_gpu

        # Re-map this step's newly written tokens to the compacted physical
        # frontier in one vectorized paged-KV lookup.
        query_start_loc = representative.query_start_loc
        query_lens_gpu = (
            query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
        ).long()
        rows = torch.repeat_interleave(
            torch.arange(num_reqs, device=block_table.device),
            query_lens_gpu,
        )
        flat_positions = torch.arange(
            sum(query_lens),
            device=block_table.device,
            dtype=torch.long,
        )
        relative_positions = flat_positions - query_start_loc[:num_reqs].long().index_select(
            0, rows
        )
        physical_starts = physical_seq_lens_gpu.long() - query_lens_gpu
        positions = relative_positions + physical_starts.index_select(0, rows)
        slot_mapping = (
            block_table[rows, positions // self._block_size].long() * self._block_size
            + positions % self._block_size
        )

        apply_physical_kv_view(
            forward_context,
            seq_lens=seq_lens,
            max_seq_len=max(physical_seq_lens),
            block_table=block_table,
            slot_mapping={layer_name: slot_mapping for layer_name in attn_metadata},
        )
        self.install_query_hooks(forward_context.no_compile_layers)
        self.begin_step(
            [state.request_id for state in ordered_states],
            next(iter(forward_context.attn_metadata.values())),
            physical_seq_lens=physical_seq_lens,
            is_genuine_decode=[
                bool(state.is_genuine_decode) for state in ordered_states
            ],
            num_decoded_tokens=[
                int(state.num_decoded_tokens) for state in ordered_states
            ],
            num_new_tokens=[
                int(state.num_new_tokens) for state in ordered_states
            ],
        )

    def begin_step(
        self,
        request_ids: list[str],
        attn_metadata: Any,
        *,
        physical_seq_lens: list[int],
        is_genuine_decode: list[bool] | None = None,
        num_decoded_tokens: list[int] | None = None,
        num_new_tokens: list[int] | None = None,
    ) -> None:
        if not self._kv_caches:
            raise RuntimeError("R-KV KV caches are not registered")
        if getattr(attn_metadata, "use_cascade", False):
            raise ValueError("R-KV MVP does not support cascade attention")

        self._clear_step()
        self._request_ids = list(request_ids)

        query_start_loc = attn_metadata.query_start_loc
        if query_start_loc.shape[0] != len(self._request_ids) + 1:
            raise ValueError("R-KV request/query segmentation mismatch")
        if len(physical_seq_lens) != len(self._request_ids):
            raise ValueError("R-KV request/sequence-length mismatch")

        num_reqs = len(self._request_ids)
        self._is_genuine_decode = (
            [True] * num_reqs
            if is_genuine_decode is None
            else list(is_genuine_decode)
        )
        self._num_decoded_tokens = (
            [0] * num_reqs
            if num_decoded_tokens is None
            else list(num_decoded_tokens)
        )
        self._num_new_tokens = (
            [1] * num_reqs
            if num_new_tokens is None
            else list(num_new_tokens)
        )
        if (
            len(self._is_genuine_decode) != num_reqs
            or len(self._num_decoded_tokens) != num_reqs
            or len(self._num_new_tokens) != num_reqs
        ):
            raise ValueError("R-KV request/step fact count mismatch")

        self._prepare_query_write_plan(query_start_loc)
        self._seq_lens = list(physical_seq_lens)
        self._block_table = attn_metadata.block_table

    def install_query_hooks(self, no_compile_layers: dict[str, Any]) -> None:
        if self._query_hooks_installed:
            return

        missing = set(self._layer_names) - set(no_compile_layers)
        if missing:
            raise RuntimeError(f"R-KV attention layers are missing: {sorted(missing)}")

        for layer_name in self._layer_names:
            layer = no_compile_layers[layer_name]
            impl = getattr(layer, "impl", None)
            original_forward = getattr(impl, "forward", None)
            if original_forward is None:
                raise RuntimeError(
                    f"R-KV attention backend is missing for {layer_name}"
                )

            def forward_with_query_capture(
                attn_layer: Any,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                kv_cache: torch.Tensor,
                attn_metadata: Any,
                *args: Any,
                _forward: Any = original_forward,
                **kwargs: Any,
            ) -> Any:
                # CUDA uses the opaque unified-attention custom op. Under
                # PIECEWISE cudagraph this backend call remains outside the
                # graph and therefore executes on every model step. vLLM may
                # pad query rows for cudagraphs; only real scheduled rows belong
                # to the per-request observation window.
                num_actual_tokens = getattr(
                    attn_metadata, "num_actual_tokens", query.shape[0]
                )
                self.capture_query(
                    attn_layer.layer_name, query[:num_actual_tokens]
                )
                return _forward(
                    attn_layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    *args,
                    **kwargs,
                )

            # Assign on the implementation instance (not its class) so only
            # this engine's R-KV attention path is observed.
            impl.forward = forward_with_query_capture

        self._query_hooks_installed = True

    def _prepare_query_write_plan(self, query_start_loc: torch.Tensor) -> None:
        assert self._is_genuine_decode is not None

        active_rows: list[int] = []
        active_slots: list[int] = []
        cursors: list[int] = []
        for row, (request_id, is_decode) in enumerate(
            zip(self._request_ids, self._is_genuine_decode, strict=True)
        ):
            slot = self._query_slots.get(request_id)
            if slot is None:
                slot = (
                    self._free_query_slots.pop()
                    if self._free_query_slots
                    else self._next_query_slot
                )
                if slot == self._next_query_slot:
                    self._next_query_slot += 1
                self._query_slots[request_id] = slot
                self._query_counts[request_id] = 0

            if not is_decode:
                continue

            count = self._query_counts[request_id]
            active_rows.append(row)
            active_slots.append(slot)
            cursors.append(count % self.window_size)
            self._query_counts[request_id] = count + 1

        if not active_rows:
            return

        if self._next_query_slot > self._query_ring_width:
            width = max(self._next_query_slot, 16)
            if self._query_ring_width:
                width = max(width, self._query_ring_width * 2)
            self._query_ring_width = width

        device = query_start_loc.device
        rows = torch.as_tensor(active_rows, dtype=torch.long, device=device)
        self._query_last_indices = (
            query_start_loc.index_select(0, rows + 1) - 1
        ).clamp_min(0).long()
        self._query_write_indices = (
            torch.as_tensor(cursors, dtype=torch.long, device=device)
            * self._query_ring_width
            + torch.as_tensor(active_slots, dtype=torch.long, device=device)
        )

    def capture_query(self, layer_name: str, query: torch.Tensor) -> None:
        if layer_name not in self._kv_caches:
            return
        if self._query_write_indices is None or self._query_last_indices is None:
            return

        head_dim = self._kv_caches[layer_name].shape[-1]
        if query.ndim == 2 and query.shape[1] % head_dim == 0:
            query = query.view(query.shape[0], -1, head_dim)
        elif query.ndim != 3 or query.shape[2] != head_dim:
            raise ValueError(
                "R-KV expects query shaped [tokens, q_heads * head_dim] "
                "or [tokens, q_heads, head_dim]"
            )

        last_q = query.index_select(0, self._query_last_indices)
        ring = self._query_rings.get(layer_name)
        if ring is None or ring.shape[1] < self._query_ring_width:
            new_ring = last_q.new_zeros(
                (
                    self.window_size,
                    self._query_ring_width,
                    last_q.shape[1],
                    last_q.shape[2],
                )
            )
            if ring is not None:
                new_ring[:, : ring.shape[1]] = ring
            ring = new_ring
            self._query_rings[layer_name] = ring

        ring.view(
            self.window_size * self._query_ring_width,
            last_q.shape[1],
            last_q.shape[2],
        ).index_copy_(0, self._query_write_indices, last_q)

    def _slots_for_request(self, row: int, seq_len: int) -> torch.Tensor:
        assert self._block_table is not None
        device = self._kv_caches[self._layer_names[0]].device
        positions = torch.arange(seq_len, device=device)
        block_ids = self._block_table[row, positions // self._block_size].long()
        return block_ids * self._block_size + positions % self._block_size

    def _compact_group_batched(
        self,
        members: list[tuple[int, str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score contiguous K/Q and return shared source/destination slots."""
        num_reqs = len(members)
        num_layers = len(self._layer_names)
        seq_len = members[0][2].numel()
        first_cache = self._kv_caches[self._layer_names[0]][:, 0]
        kv_heads = first_cache.shape[2]
        head_dim = first_cache.shape[3]
        elt = first_cache.element_size()

        # Match the upstream serving implementation's quadratic-score cap.
        per_unit = max(
            1,
            2 * (2 * elt + 1 + 4) * kv_heads * seq_len * seq_len,
        )
        units_cap = max(1, self.score_chunk_bytes // per_unit)
        req_chunk = max(1, min(num_reqs, units_cap))
        layer_chunk = max(1, min(num_layers, units_cap // req_chunk))

        source_parts: list[torch.Tensor] = []
        destination_parts: list[torch.Tensor] = []

        for r0 in range(0, num_reqs, req_chunk):
            chunk_members = members[r0 : r0 + req_chunk]
            rc = len(chunk_members)
            slots_list = [member[2] for member in chunk_members]
            slots_cat = torch.cat(slots_list)
            blocks = slots_cat // self._block_size
            offsets = slots_cat % self._block_size

            shared_scores: torch.Tensor | None = None
            for l0 in range(0, num_layers, layer_chunk):
                layer_names = self._layer_names[l0 : l0 + layer_chunk]
                lc = len(layer_names)
                keys = (
                    torch.stack(
                        [
                            self._kv_caches[name][:, 0][blocks, offsets]
                            for name in layer_names
                        ]
                    )
                    .view(lc, rc, seq_len, kv_heads, head_dim)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(lc * rc, kv_heads, seq_len, head_dim)
                    .contiguous()
                )
                queries = torch.stack(
                    [
                        torch.stack(
                            [
                                self._query_rings[name][
                                    :, self._query_slots[request_id]
                                ]
                                for _, request_id, _ in chunk_members
                            ]
                        )
                        for name in layer_names
                    ]
                )
                q_heads = queries.shape[3]
                queries = (
                    queries.permute(0, 1, 3, 2, 4)
                    .reshape(lc * rc, q_heads, self.window_size, head_dim)
                    .contiguous()
                )

                layer_scores = self._policy.score_kv(keys, queries).mean(dim=1)
                layer_scores = layer_scores.view(
                    lc, rc, seq_len - self.window_size
                )
                for li in range(lc):
                    shared_scores = (
                        layer_scores[li]
                        if shared_scores is None
                        else shared_scores + layer_scores[li]
                    )

            assert shared_scores is not None
            past_idx = shared_scores.topk(
                self.budget - self.window_size,
                dim=-1,
            ).indices
            window_idx = torch.arange(
                seq_len - self.window_size,
                seq_len,
                device=past_idx.device,
            ).expand(rc, self.window_size)
            kept = torch.sort(
                torch.cat([past_idx, window_idx], dim=-1),
                dim=-1,
            ).values

            source_parts.append(
                torch.gather(
                    torch.stack(slots_list),
                    1,
                    kept,
                ).reshape(-1)
            )
            destination_parts.append(
                torch.cat([slots[: self.budget] for slots in slots_list])
            )

        return torch.cat(source_parts), torch.cat(destination_parts)

    def compact(self) -> dict[str, int]:
        if self._seq_lens is None or self._block_table is None:
            return {}

        assert self._is_genuine_decode is not None
        assert self._num_decoded_tokens is not None
        assert self._num_new_tokens is not None
        groups: dict[int, list[tuple[int, str, torch.Tensor]]] = {}
        for row, request_id in enumerate(self._request_ids):
            seq_len = self._seq_lens[row]
            query_window_tokens = min(
                self._query_counts.get(request_id, 0),
                self.window_size,
            )
            if (
                request_id in self._compacted_requests
                and query_window_tokens < self.window_size
            ):
                raise RuntimeError(
                    "R-KV lost its observation window after compaction"
                )

            if not self._policy.should_compact(
                resident_len=seq_len,
                num_decoded_tokens=self._num_decoded_tokens[row],
                num_new_tokens=self._num_new_tokens[row],
                is_genuine_decode=self._is_genuine_decode[row],
                query_window_tokens=query_window_tokens,
            ):
                continue

            groups.setdefault(seq_len, []).append(
                (row, request_id, self._slots_for_request(row, seq_len))
            )

        if not groups:
            return {}

        updates: dict[str, int] = {}
        source_parts: list[torch.Tensor] = []
        destination_parts: list[torch.Tensor] = []
        for members in groups.values():
            source_slots, destination_slots = self._compact_group_batched(members)
            source_parts.append(source_slots)
            destination_parts.append(destination_slots)
            for _, request_id, _ in members:
                updates[request_id] = self.budget
                self._compacted_requests.add(request_id)

        source_slots = torch.cat(source_parts)
        destination_slots = torch.cat(destination_parts)
        src_blocks = source_slots // self._block_size
        src_offsets = source_slots % self._block_size
        dst_blocks = destination_slots // self._block_size
        dst_offsets = destination_slots % self._block_size
        for name in self._layer_names:
            key_cache = self._kv_caches[name][:, 0]
            value_cache = self._kv_caches[name][:, 1]
            kept_keys = key_cache[src_blocks, src_offsets]
            kept_values = value_cache[src_blocks, src_offsets]
            key_cache[dst_blocks, dst_offsets] = kept_keys
            value_cache[dst_blocks, dst_offsets] = kept_values

        return updates

    def _clear_step(self) -> None:
        self._request_ids = []
        self._query_last_indices = None
        self._query_write_indices = None
        self._seq_lens = None
        self._block_table = None
        self._is_genuine_decode = None
        self._num_decoded_tokens = None
        self._num_new_tokens = None
