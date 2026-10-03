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
        retain_direction: str = "last",
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
        if retain_direction not in ("last", "first"):
            raise ValueError("R-KV retain_direction must be 'last' or 'first'")
        if score_chunk_bytes <= 0:
            raise ValueError("R-KV score_chunk_bytes must be positive")

        try:
            from rkv import R1KV
        except ImportError as exc:
            raise ImportError(
                "R-KV is enabled but the optional 'rkv' package is not installed"
            ) from exc

        self.budget = budget
        self.buffer = buffer
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.score_chunk_bytes = score_chunk_bytes
        self._policy = R1KV(
            budget=budget,
            window_size=window_size,
            kernel_size=kernel_size,
            mix_lambda=mix_lambda,
            retain_ratio=retain_ratio,
            retain_direction=retain_direction,
        )
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._block_size = 0
        self._recent_queries: dict[str, dict[str, torch.Tensor]] = {}

        self._request_ids: list[str] = []
        self._query_ranges: list[tuple[int, int]] | None = None
        self._seq_lens: list[int] | None = None
        self._block_table: torch.Tensor | None = None
        self._is_genuine_decode: list[bool] | None = None
        self._should_compress: list[bool] | None = None
        self._compacted_requests: set[str] = set()
        self._n_compactions = 0
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
        self._recent_queries.clear()
        self._compacted_requests.clear()
        self._clear_step()

    def drop_requests(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self._recent_queries.pop(request_id, None)
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
        query_start_loc = representative.query_start_loc.tolist()
        num_reqs = len(query_start_loc) - 1
        if len(request_states) != num_reqs:
            raise ValueError("R-KV request/attention row count mismatch")

        by_first_block: dict[int, Any] = {}
        for state in request_states:
            if not state.block_ids:
                raise ValueError(f"R-KV request {state.request_id} has no KV blocks")
            first_block = state.block_ids[0]
            if first_block in by_first_block:
                raise ValueError("R-KV requires private, uniquely-owned KV blocks")
            by_first_block[first_block] = state

        first_blocks = representative.block_table[:num_reqs, 0].tolist()
        try:
            ordered_states = [by_first_block[block_id] for block_id in first_blocks]
        except KeyError as exc:
            raise ValueError("R-KV could not match worker rows to requests") from exc

        query_lens = [
            end - start
            for start, end in zip(
                query_start_loc[:-1], query_start_loc[1:], strict=True
            )
        ]
        logical_seq_lens = representative.seq_lens[:num_reqs].tolist()
        physical_seq_lens = [
            logical_len
            if state.resident_kv_tokens is None
            else state.resident_kv_tokens
            for state, logical_len in zip(ordered_states, logical_seq_lens, strict=True)
        ]
        if any(
            query_len <= 0 or physical_len < query_len
            for query_len, physical_len in zip(
                query_lens, physical_seq_lens, strict=True
            )
        ):
            raise ValueError("R-KV received invalid physical/query lengths")

        block_table = torch.zeros_like(representative.block_table)
        for row, state in enumerate(ordered_states):
            if len(state.block_ids) > block_table.shape[1]:
                raise ValueError("R-KV block table exceeds worker capacity")
            block_table[row, : len(state.block_ids)] = torch.tensor(
                state.block_ids,
                dtype=block_table.dtype,
                device=block_table.device,
            )

        seq_lens = representative.seq_lens.clone()
        seq_lens[:num_reqs] = torch.tensor(
            physical_seq_lens,
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )

        slot_parts: list[torch.Tensor] = []
        for row, (physical_len, query_len) in enumerate(
            zip(physical_seq_lens, query_lens, strict=True)
        ):
            positions = torch.arange(
                physical_len - query_len,
                physical_len,
                device=block_table.device,
                dtype=torch.long,
            )
            row_blocks = block_table[row]
            slot_parts.append(
                row_blocks[positions // self._block_size].long() * self._block_size
                + positions % self._block_size
            )
        slot_mapping = torch.cat(slot_parts)

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
            is_genuine_decode=[
                bool(state.is_genuine_decode) for state in ordered_states
            ],
            should_compress=[
                bool(state.should_compress) for state in ordered_states
            ],
        )

    def begin_step(
        self,
        request_ids: list[str],
        attn_metadata: Any,
        *,
        is_genuine_decode: list[bool] | None = None,
        should_compress: list[bool] | None = None,
    ) -> None:
        if not self._kv_caches:
            raise RuntimeError("R-KV KV caches are not registered")
        if getattr(attn_metadata, "use_cascade", False):
            raise ValueError("R-KV MVP does not support cascade attention")

        self._clear_step()
        self._request_ids = list(request_ids)

        query_start_loc = attn_metadata.query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()
        if len(query_start_loc) != len(self._request_ids) + 1:
            raise ValueError("R-KV request/query segmentation mismatch")
        if len(seq_lens) != len(self._request_ids):
            raise ValueError("R-KV request/sequence-length mismatch")

        num_reqs = len(self._request_ids)
        self._is_genuine_decode = (
            [True] * num_reqs
            if is_genuine_decode is None
            else list(is_genuine_decode)
        )
        self._should_compress = (
            [True] * num_reqs
            if should_compress is None
            else list(should_compress)
        )
        if (
            len(self._is_genuine_decode) != num_reqs
            or len(self._should_compress) != num_reqs
        ):
            raise ValueError("R-KV request/step flag count mismatch")

        self._query_ranges = list(
            zip(query_start_loc[:-1], query_start_loc[1:], strict=True)
        )
        self._seq_lens = seq_lens
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
                # graph and therefore executes on every model step.
                self.capture_query(attn_layer.layer_name, query)
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

    def capture_query(self, layer_name: str, query: torch.Tensor) -> None:
        if layer_name not in self._kv_caches:
            return
        if self._query_ranges is None:
            return

        head_dim = self._kv_caches[layer_name].shape[-1]
        if query.ndim == 2 and query.shape[1] % head_dim == 0:
            query = query.view(query.shape[0], -1, head_dim)
        elif query.ndim != 3 or query.shape[2] != head_dim:
            raise ValueError(
                "R-KV expects query shaped [tokens, q_heads * head_dim] "
                "or [tokens, q_heads, head_dim]"
            )

        if self._query_ranges[-1][1] != query.shape[0]:
            raise ValueError("R-KV query row count mismatch")

        assert self._is_genuine_decode is not None
        for row_index, (request_id, (start, end)) in enumerate(
            zip(self._request_ids, self._query_ranges, strict=True)
        ):
            if not self._is_genuine_decode[row_index]:
                continue
            rows = query[start:end]
            if rows.shape[0] == 0:
                continue
            # Upstream R-KV records one frontier query per genuine decode step.
            rows = rows[-1:]

            by_layer = self._recent_queries.setdefault(request_id, {})
            previous = by_layer.get(layer_name)
            if rows.shape[0] >= self.window_size:
                by_layer[layer_name] = rows[-self.window_size :].detach().clone()
            elif previous is None:
                by_layer[layer_name] = rows.detach().clone()
            else:
                by_layer[layer_name] = torch.cat([previous, rows], dim=0)[-self.window_size :]

    def _slots_for_request(self, row: int, seq_len: int) -> torch.Tensor:
        assert self._block_table is not None
        device = self._kv_caches[self._layer_names[0]].device
        positions = torch.arange(seq_len, device=device)
        block_ids = self._block_table[row, positions // self._block_size].long()
        return block_ids * self._block_size + positions % self._block_size

    def _compact_group_batched(
        self,
        members: list[tuple[int, str, torch.Tensor]],
    ) -> None:
        """Run canonical R1KV on contiguous Q/K/V and write compacted K/V back."""
        num_reqs = len(members)
        num_layers = len(self._layer_names)
        seq_len = members[0][2].numel()
        first_cache = self._kv_caches[self._layer_names[0]][:, 0]
        kv_heads = first_cache.shape[2]
        head_dim = first_cache.shape[3]
        elt = first_cache.element_size()

        # Bound the same quadratic score working set as the upstream vLLM port.
        per_unit = max(
            1,
            2 * (2 * elt + 1 + 4) * kv_heads * seq_len * seq_len,
        )
        units_cap = max(1, self.score_chunk_bytes // per_unit)
        req_chunk = max(1, min(num_reqs, units_cap))
        layer_chunk = max(1, min(num_layers, units_cap // req_chunk))

        for r0 in range(0, num_reqs, req_chunk):
            chunk_members = members[r0 : r0 + req_chunk]
            rc = len(chunk_members)
            slots_cat = torch.cat([member[2] for member in chunk_members])
            blocks = slots_cat // self._block_size
            offsets = slots_cat % self._block_size

            destination_slots = torch.cat(
                [member[2][: self.budget] for member in chunk_members]
            )
            destination_blocks = destination_slots // self._block_size
            destination_offsets = destination_slots % self._block_size

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
                values = (
                    torch.stack(
                        [
                            self._kv_caches[name][:, 1][blocks, offsets]
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
                                self._recent_queries[request_id][name]
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

                compacted_keys, compacted_values = self._policy.update_kv(
                    keys,
                    queries,
                    values,
                )
                compacted_keys = (
                    compacted_keys.view(
                        lc, rc, kv_heads, self.budget, head_dim
                    )
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                )
                compacted_values = (
                    compacted_values.view(
                        lc, rc, kv_heads, self.budget, head_dim
                    )
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                )

                for li, name in enumerate(layer_names):
                    key_cache = self._kv_caches[name][:, 0]
                    value_cache = self._kv_caches[name][:, 1]
                    key_cache[destination_blocks, destination_offsets] = (
                        compacted_keys[li].reshape(
                            rc * self.budget, kv_heads, head_dim
                        )
                    )
                    value_cache[destination_blocks, destination_offsets] = (
                        compacted_values[li].reshape(
                            rc * self.budget, kv_heads, head_dim
                        )
                    )

    def compact(self) -> dict[str, int]:
        if self._seq_lens is None or self._block_table is None:
            return {}

        assert self._should_compress is not None
        groups: dict[int, list[tuple[int, str, torch.Tensor]]] = {}
        for row, request_id in enumerate(self._request_ids):
            if not self._should_compress[row]:
                continue
            seq_len = self._seq_lens[row]
            if seq_len < self.budget + self.buffer:
                continue
            queries = self._recent_queries.get(request_id, {})
            has_window = all(
                name in queries and queries[name].shape[0] == self.window_size
                for name in self._layer_names
            )
            if not has_window:
                if request_id in self._compacted_requests:
                    raise RuntimeError(
                        "R-KV lost its observation window after compaction"
                    )
                # Safe before the first compaction (e.g. post-preemption catch-up).
                continue
            groups.setdefault(seq_len, []).append(
                (row, request_id, self._slots_for_request(row, seq_len))
            )

        if not groups:
            return {}

        updates: dict[str, int] = {}
        for members in groups.values():
            self._compact_group_batched(members)
            for _, request_id, _ in members:
                updates[request_id] = self.budget
                self._compacted_requests.add(request_id)
                self._n_compactions += 1

        return updates

    def _clear_step(self) -> None:
        self._request_ids = []
        self._query_ranges = None
        self._seq_lens = None
        self._block_table = None
        self._is_genuine_decode = None
        self._should_compress = None
