# SPDX-License-Identifier: Apache-2.0
"""Worker-local R-KV query capture and KV compaction."""

from __future__ import annotations

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.integration.vllm.token_drop import (
    TokenDropSpec,
    build_token_drop_algorithm,
)


class RKVWorker:
    """Capture recent queries and compact private FlashAttention KV in place."""

    def __init__(self) -> None:
        self._request_algorithms: dict[str, Any] = {}
        self._max_observation_tokens = 0
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._block_size = 0
        self._kv_caches_validated = False

        self._query_rings: dict[str, torch.Tensor] = {}
        self._query_slots: dict[str, int] = {}
        self._free_query_slots: list[int] = []
        self._query_counts: dict[str, int] = {}
        self._next_query_slot = 0
        self._query_ring_width = 0
        self._query_last_indices: torch.Tensor | None = None
        self._query_write_indices: torch.Tensor | None = None

        self._request_ids: list[str] = []
        self._request_rows: list[int] = []
        self._seq_lens: list[int] | None = None
        self._block_table: torch.Tensor | None = None
        self._is_genuine_decode: list[bool] | None = None
        self._num_decoded_tokens: list[int] | None = None
        self._num_new_tokens: list[int] | None = None
        self._compacted_requests: set[str] = set()
        self._query_hooks_installed = False
        self._query_hook_originals: dict[str, tuple[Any, Any]] = {}

    def _algorithm_from_state(self, state: Any) -> Any:
        spec = TokenDropSpec(
            algorithm=str(state.algorithm),
            config=dict(state.config),
        )
        existing = self._request_algorithms.get(state.request_id)
        if existing is not None:
            return existing

        algorithm = build_token_drop_algorithm(spec)
        observation_tokens = int(algorithm.observation_window_tokens)
        if observation_tokens <= 0:
            raise ValueError(
                "Token-drop algorithm requires a positive observation window"
            )

        self._request_algorithms[state.request_id] = algorithm
        self._max_observation_tokens = max(
            self._max_observation_tokens,
            observation_tokens,
        )
        return algorithm

    def _algorithm_for_request(self, request_id: str) -> Any:
        algorithm = self._request_algorithms.get(request_id)
        if algorithm is None:
            raise RuntimeError(
                f"Missing token-drop algorithm for request {request_id!r}"
            )
        return algorithm

    def is_token_drop_request(self, request_id: str) -> bool:
        return request_id in self._request_algorithms

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # Registration is shared with normal LMCache serving. Do not impose
        # R-KV-specific layout/device constraints until a request opts in.
        self._kv_caches = dict(kv_caches)
        self._layer_names = list(kv_caches)
        self._block_size = 0
        self._kv_caches_validated = False

    def _ensure_kv_caches_compatible(self) -> None:
        if self._kv_caches_validated:
            return
        if not self._kv_caches:
            raise ValueError("R-KV requires KV caches")

        block_size = 0
        for kv_cache in self._kv_caches.values():
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
            block_size = int(kv_cache.shape[2])

        self._block_size = block_size
        self._kv_caches_validated = True

    def drop_requests(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            slot = self._query_slots.pop(request_id, None)
            if slot is not None:
                self._free_query_slots.append(slot)
            self._query_counts.pop(request_id, None)
            self._request_algorithms.pop(request_id, None)
            self._compacted_requests.discard(request_id)

    def prepare_forward(
        self,
        forward_context: Any,
        request_states: list[Any],
    ) -> None:
        if not request_states:
            self.remove_query_hooks()
            self._clear_step()
            return

        self._ensure_kv_caches_compatible()
        algorithms = [self._algorithm_from_state(state) for state in request_states]
        observe_query = [
            algorithm.should_observe_query(
                num_decoded_tokens=int(state.num_decoded_tokens),
                num_new_tokens=int(state.num_new_tokens),
                is_genuine_decode=bool(state.is_genuine_decode),
            )
            for state, algorithm in zip(request_states, algorithms, strict=True)
        ]
        if not any(observe_query):
            self.remove_query_hooks()
            self._clear_step()
            return

        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict) or not attn_metadata:
            raise ValueError("R-KV MVP requires eager attention metadata")

        representative = next(iter(attn_metadata.values()))
        ordered_states = list(request_states)
        request_rows = [int(state.worker_row) for state in ordered_states]
        if any(row < 0 for row in request_rows):
            raise RuntimeError("Token-drop metadata is missing worker row identity")
        if (
            request_rows
            and max(request_rows) + 1 >= representative.query_start_loc.shape[0]
        ):
            raise ValueError("R-KV request/attention row mismatch")

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

        self.install_query_hooks(forward_context.no_compile_layers)
        self.begin_step(
            [state.request_id for state in ordered_states],
            representative,
            physical_seq_lens=physical_seq_lens,
            request_rows=request_rows,
            observe_query=observe_query,
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
        request_rows: list[int] | None = None,
        observe_query: list[bool] | None = None,
        is_genuine_decode: list[bool] | None = None,
        num_decoded_tokens: list[int] | None = None,
        num_new_tokens: list[int] | None = None,
    ) -> None:
        self._ensure_kv_caches_compatible()
        if getattr(attn_metadata, "use_cascade", False):
            raise ValueError("R-KV MVP does not support cascade attention")

        self._clear_step()
        self._request_ids = list(request_ids)
        self._request_rows = (
            list(range(len(request_ids)))
            if request_rows is None
            else list(request_rows)
        )

        query_start_loc = attn_metadata.query_start_loc
        if len(self._request_rows) != len(self._request_ids):
            raise ValueError("R-KV request/row count mismatch")
        if self._request_rows and (
            min(self._request_rows) < 0
            or max(self._request_rows) + 1 >= query_start_loc.shape[0]
        ):
            raise ValueError("R-KV request/query segmentation mismatch")
        if len(physical_seq_lens) != len(self._request_ids):
            raise ValueError("R-KV request/sequence-length mismatch")

        num_reqs = len(self._request_ids)
        for request_id in self._request_ids:
            self._algorithm_for_request(request_id)
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
        observe_query = (
            list(self._is_genuine_decode)
            if observe_query is None
            else list(observe_query)
        )
        if (
            len(self._is_genuine_decode) != num_reqs
            or len(self._num_decoded_tokens) != num_reqs
            or len(self._num_new_tokens) != num_reqs
            or len(observe_query) != num_reqs
        ):
            raise ValueError("R-KV request/step fact count mismatch")

        self._prepare_query_write_plan(query_start_loc, observe_query)
        self._seq_lens = list(physical_seq_lens)
        self._block_table = attn_metadata.block_table

    def install_query_hooks(self, no_compile_layers: dict[str, Any]) -> None:
        if self._query_hooks_installed:
            return

        missing = set(self._layer_names) - set(no_compile_layers)
        if missing:
            raise RuntimeError(f"R-KV attention layers are missing: {sorted(missing)}")

        originals: dict[str, tuple[Any, Any]] = {}
        for layer_name in self._layer_names:
            layer = no_compile_layers[layer_name]
            impl = getattr(layer, "impl", None)
            original_forward = getattr(impl, "forward", None)
            if original_forward is None:
                raise RuntimeError(
                    f"R-KV attention backend is missing for {layer_name}"
                )
            originals[layer_name] = (impl, original_forward)

        for impl, original_forward in originals.values():

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

        self._query_hook_originals = originals
        self._query_hooks_installed = True

    def remove_query_hooks(self) -> None:
        if not self._query_hooks_installed:
            return

        for impl, original_forward in self._query_hook_originals.values():
            impl.forward = original_forward
        self._query_hook_originals.clear()
        self._query_hooks_installed = False

    def _prepare_query_write_plan(
        self,
        query_start_loc: torch.Tensor,
        observe_query: list[bool],
    ) -> None:
        assert self._is_genuine_decode is not None

        active_rows: list[int] = []
        active_slots: list[int] = []
        cursors: list[int] = []
        for request_id, worker_row, is_decode, observe in zip(
            self._request_ids,
            self._request_rows,
            self._is_genuine_decode,
            observe_query,
            strict=True,
        ):
            if not is_decode or not observe:
                continue

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

            algorithm = self._algorithm_for_request(request_id)
            count = self._query_counts[request_id]
            active_rows.append(worker_row)
            active_slots.append(slot)
            cursors.append(count % int(algorithm.observation_window_tokens))
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
        if (
            ring is None
            or ring.shape[0] < self._max_observation_tokens
            or ring.shape[1] < self._query_ring_width
        ):
            new_ring = last_q.new_zeros(
                (
                    self._max_observation_tokens,
                    self._query_ring_width,
                    last_q.shape[1],
                    last_q.shape[2],
                )
            )
            if ring is not None:
                new_ring[: ring.shape[0], : ring.shape[1]] = ring
            ring = new_ring
            self._query_rings[layer_name] = ring

        ring.view(
            self._max_observation_tokens * self._query_ring_width,
            last_q.shape[1],
            last_q.shape[2],
        ).index_copy_(0, self._query_write_indices, last_q)

    def _slots_for_request(self, row: int, seq_len: int) -> torch.Tensor:
        assert self._block_table is not None
        device = self._kv_caches[self._layer_names[0]].device
        positions = torch.arange(seq_len, device=device)
        block_ids = self._block_table[row, positions // self._block_size].long()
        return block_ids * self._block_size + positions % self._block_size

    def _compact_request(
        self,
        request_id: str,
        worker_row: int,
        seq_len: int,
        algorithm: Any,
    ) -> int:
        """Apply one algorithm-selected retained-position set to all KV layers."""
        slots = self._slots_for_request(worker_row, seq_len)
        blocks = slots // self._block_size
        offsets = slots % self._block_size
        observation_tokens = int(algorithm.observation_window_tokens)

        layer_keys: list[torch.Tensor] = []
        layer_queries: list[torch.Tensor] = []
        for name in self._layer_names:
            layer_keys.append(
                self._kv_caches[name][:, 0][blocks, offsets]
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
            layer_queries.append(
                self._query_rings[name][
                    :observation_tokens,
                    self._query_slots[request_id],
                ]
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )

        kept = algorithm.select_kept_positions(layer_keys, layer_queries)
        if not isinstance(kept, torch.Tensor):
            raise RuntimeError("Token-drop algorithm must return a tensor of positions")
        if kept.device != slots.device:
            raise RuntimeError("Token-drop kept positions must stay on the KV device")
        if kept.ndim != 1 or kept.numel() == 0:
            raise RuntimeError(
                "Token-drop kept positions must be a non-empty 1-D tensor"
            )
        if kept.dtype not in (torch.int32, torch.int64):
            raise RuntimeError(
                "Token-drop kept positions must use an integer index dtype"
            )
        if kept.numel() > seq_len:
            raise RuntimeError(
                "Token-drop kept positions exceed the resident KV length"
            )
        if (kept < 0).any() or (kept >= seq_len).any():
            raise RuntimeError(
                "Token-drop kept positions are outside resident KV bounds"
            )
        if kept.numel() > 1 and not torch.all(kept[1:] > kept[:-1]):
            raise RuntimeError(
                "Token-drop kept positions must be unique and in logical order"
            )

        new_resident_len = int(kept.numel())
        source_slots = slots[kept]
        destination_slots = slots[:new_resident_len]
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

        return new_resident_len

    def compact(self) -> dict[str, int]:
        if self._seq_lens is None or self._block_table is None:
            return {}

        assert self._is_genuine_decode is not None
        assert self._num_decoded_tokens is not None
        assert self._num_new_tokens is not None

        updates: dict[str, int] = {}
        for local_row, (request_id, worker_row) in enumerate(
            zip(self._request_ids, self._request_rows, strict=True)
        ):
            algorithm = self._algorithm_for_request(request_id)
            seq_len = self._seq_lens[local_row]
            observation_tokens = int(algorithm.observation_window_tokens)
            query_window_tokens = min(
                self._query_counts.get(request_id, 0),
                observation_tokens,
            )
            if (
                request_id in self._compacted_requests
                and query_window_tokens < observation_tokens
            ):
                raise RuntimeError(
                    "Token-drop algorithm lost its observation window after compaction"
                )

            if not algorithm.should_compact(
                resident_len=seq_len,
                num_decoded_tokens=self._num_decoded_tokens[local_row],
                num_new_tokens=self._num_new_tokens[local_row],
                is_genuine_decode=self._is_genuine_decode[local_row],
                query_window_tokens=query_window_tokens,
            ):
                continue

            updates[request_id] = self._compact_request(
                request_id,
                worker_row,
                seq_len,
                algorithm,
            )
            self._compacted_requests.add(request_id)

        return updates

    def _clear_step(self) -> None:
        self._request_ids = []
        self._request_rows = []
        self._query_last_indices = None
        self._query_write_indices = None
        self._seq_lens = None
        self._block_table = None
        self._is_genuine_decode = None
        self._num_decoded_tokens = None
        self._num_new_tokens = None
