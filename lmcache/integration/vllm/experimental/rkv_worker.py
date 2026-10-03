# SPDX-License-Identifier: Apache-2.0
"""Worker-local R-KV query capture and KV compaction."""

from __future__ import annotations

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.integration.vllm.experimental.kv_compaction import (
    apply_kv_compaction_moves,
)
from lmcache.integration.vllm.experimental.physical_kv_view import (
    apply_physical_kv_view,
)
from lmcache.integration.vllm.experimental.rkv import select_rkv_retained_indices

_RKV_WINDOW = 8


class RKVWorker:
    """Capture recent queries and compact private FlashAttention KV in place."""

    def __init__(self, budget: int) -> None:
        if budget <= _RKV_WINDOW:
            raise ValueError(f"R-KV budget must exceed {_RKV_WINDOW}")
        self.budget = budget
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._block_size = 0
        self._recent_queries: dict[str, dict[str, torch.Tensor]] = {}

        self._request_ids: list[str] = []
        self._query_ranges: list[tuple[int, int]] | None = None
        self._seq_lens: list[int] | None = None
        self._block_table: torch.Tensor | None = None
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
        self._clear_step()

    def drop_requests(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self._recent_queries.pop(request_id, None)

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
        )

    def begin_step(self, request_ids: list[str], attn_metadata: Any) -> None:
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

            def capture(_module: Any, args: tuple[Any, ...], name: str = layer_name):
                if not args:
                    raise RuntimeError("R-KV attention hook did not receive query")
                self.capture_query(name, args[0])

            layer.register_forward_pre_hook(capture)

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

        for request_id, (start, end) in zip(
            self._request_ids, self._query_ranges, strict=True
        ):
            rows = query[start:end]
            if rows.shape[0] == 0:
                continue

            by_layer = self._recent_queries.setdefault(request_id, {})
            previous = by_layer.get(layer_name)
            if rows.shape[0] >= _RKV_WINDOW:
                by_layer[layer_name] = rows[-_RKV_WINDOW:].detach().clone()
            elif previous is None:
                by_layer[layer_name] = rows.detach().clone()
            else:
                by_layer[layer_name] = torch.cat([previous, rows], dim=0)[-_RKV_WINDOW:]

    def compact(self) -> dict[str, int]:
        if self._seq_lens is None or self._block_table is None:
            return {}

        updates: dict[str, int] = {}
        for i, request_id in enumerate(self._request_ids):
            seq_len = self._seq_lens[i]
            if seq_len < self.budget + _RKV_WINDOW:
                continue

            queries = self._recent_queries.get(request_id, {})
            if any(
                layer_name not in queries or queries[layer_name].shape[0] != _RKV_WINDOW
                for layer_name in self._layer_names
            ):
                raise RuntimeError("R-KV does not have a complete query window")

            device = self._kv_caches[self._layer_names[0]].device
            positions = torch.arange(seq_len, device=device)
            block_ids = self._block_table[i, positions // self._block_size].long()
            slots = block_ids * self._block_size + positions % self._block_size

            layers = [
                (self._kv_caches[layer_name][:, 0], queries[layer_name])
                for layer_name in self._layer_names
            ]
            retained = select_rkv_retained_indices(layers, slots, self.budget)
            source_slots = slots[retained]
            destination_slots = slots[: self.budget]
            moving = source_slots != destination_slots
            copies = torch.stack(
                (source_slots[moving], destination_slots[moving]), dim=1
            )

            for layer_name in self._layer_names:
                apply_kv_compaction_moves(
                    self._kv_caches[layer_name].transpose(1, 2),
                    copies,
                )

            updates[request_id] = self.budget

        return updates

    def _clear_step(self) -> None:
        self._request_ids = []
        self._query_ranges = None
        self._seq_lens = None
        self._block_table = None
