# SPDX-License-Identifier: Apache-2.0
"""Worker-local query forwarding and paged-KV compaction for token dropping."""

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


class TokenDropWorker:
    """Forward request-local Q and apply algorithm-selected KV compaction."""

    def __init__(self) -> None:
        self._request_algorithms: dict[str, Any] = {}
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._block_size = 0
        self._kv_caches_validated = False

        self._request_ids: list[str] = []
        self._request_rows: list[int] = []
        self._seq_lens: list[int] | None = None
        self._block_table: torch.Tensor | None = None
        self._is_genuine_decode: list[bool] | None = None
        self._num_decoded_tokens: list[int] | None = None
        self._num_new_tokens: list[int] | None = None

        self._query_observation_indices: torch.Tensor | None = None
        self._query_observation_slices: list[tuple[str, int, int]] = []

        self._query_hooks_installed = False
        self._query_hook_originals: dict[str, tuple[Any, Any]] = {}

    def _algorithm_from_state(self, state: Any) -> Any:
        existing = self._request_algorithms.get(state.request_id)
        if existing is not None:
            return existing

        algorithm = build_token_drop_algorithm(
            TokenDropSpec(
                algorithm=str(state.algorithm),
                config=dict(state.config),
            )
        )
        self._request_algorithms[state.request_id] = algorithm
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
        # Registration is shared with normal LMCache serving. Validate the
        # token-drop runtime requirements only after a request opts in.
        self._kv_caches = dict(kv_caches)
        self._layer_names = list(kv_caches)
        self._block_size = 0
        self._kv_caches_validated = False

    def _ensure_kv_caches_compatible(self) -> None:
        if self._kv_caches_validated:
            return
        if not self._kv_caches:
            raise ValueError("Token dropping requires KV caches")

        block_size = 0
        for kv_cache in self._kv_caches.values():
            if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
                raise ValueError(
                    "Token dropping requires FlashAttention KV shaped "
                    "[num_blocks, 2, block_size, kv_heads, head_dim]"
                )
            if kv_cache.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("Token dropping requires FP16 or BF16 KV cache")
            if kv_cache.device.type != "cuda":
                raise ValueError("Token dropping requires CUDA-resident KV cache")
            if not kv_cache.is_contiguous():
                raise ValueError(
                    "Token dropping MVP requires contiguous NHD KV layout"
                )
            if block_size and kv_cache.shape[2] != block_size:
                raise ValueError(
                    "Token dropping requires one shared KV block size"
                )
            block_size = int(kv_cache.shape[2])

        self._block_size = block_size
        self._kv_caches_validated = True

    def drop_requests(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self._request_algorithms.pop(request_id, None)

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
            bool(
                algorithm.should_observe_query(
                    num_decoded_tokens=int(state.num_decoded_tokens),
                    num_new_tokens=int(state.num_new_tokens),
                    is_genuine_decode=bool(state.is_genuine_decode),
                )
            )
            for state, algorithm in zip(request_states, algorithms, strict=True)
        ]

        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict) or not attn_metadata:
            raise ValueError("Token dropping MVP requires eager attention metadata")

        representative = next(iter(attn_metadata.values()))
        ordered_states = list(request_states)
        request_rows = [int(state.worker_row) for state in ordered_states]
        if any(row < 0 for row in request_rows):
            raise RuntimeError("Token-drop metadata is missing worker row identity")
        if (
            request_rows
            and max(request_rows) + 1 >= representative.query_start_loc.shape[0]
        ):
            raise ValueError("Token-drop request/attention row mismatch")

        query_lens = [int(state.num_new_tokens) for state in ordered_states]
        if any(state.resident_kv_tokens is None for state in ordered_states):
            raise RuntimeError(
                "Token-drop scheduler metadata is missing resident KV length"
            )
        physical_seq_lens = [
            int(state.resident_kv_tokens) for state in ordered_states
        ]
        if any(
            query_len <= 0 or physical_len < query_len
            for query_len, physical_len in zip(
                query_lens, physical_seq_lens, strict=True
            )
        ):
            raise ValueError("Token dropping received invalid physical/query lengths")

        if any(observe_query):
            self.install_query_hooks(forward_context.no_compile_layers)
        else:
            self.remove_query_hooks()

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
            raise ValueError("Token dropping MVP does not support cascade attention")

        self._clear_step()
        self._request_ids = list(request_ids)
        self._request_rows = (
            list(range(len(request_ids)))
            if request_rows is None
            else list(request_rows)
        )

        query_start_loc = attn_metadata.query_start_loc
        if len(self._request_rows) != len(self._request_ids):
            raise ValueError("Token-drop request/row count mismatch")
        if self._request_rows and (
            min(self._request_rows) < 0
            or max(self._request_rows) + 1 >= query_start_loc.shape[0]
        ):
            raise ValueError("Token-drop request/query segmentation mismatch")
        if len(physical_seq_lens) != len(self._request_ids):
            raise ValueError("Token-drop request/sequence-length mismatch")

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
            [False] * num_reqs if observe_query is None else list(observe_query)
        )
        if (
            len(self._is_genuine_decode) != num_reqs
            or len(self._num_decoded_tokens) != num_reqs
            or len(self._num_new_tokens) != num_reqs
            or len(observe_query) != num_reqs
        ):
            raise ValueError("Token-drop request/step fact count mismatch")

        self._prepare_query_observation_plan(query_start_loc, observe_query)
        self._seq_lens = list(physical_seq_lens)
        self._block_table = attn_metadata.block_table

    def install_query_hooks(self, no_compile_layers: dict[str, Any]) -> None:
        if self._query_hooks_installed:
            return

        missing = set(self._layer_names) - set(no_compile_layers)
        if missing:
            raise RuntimeError(
                f"Token-drop attention layers are missing: {sorted(missing)}"
            )

        originals: dict[str, tuple[Any, Any]] = {}
        for layer_name in self._layer_names:
            layer = no_compile_layers[layer_name]
            impl = getattr(layer, "impl", None)
            original_forward = getattr(impl, "forward", None)
            if original_forward is None:
                raise RuntimeError(
                    f"Token-drop attention backend is missing for {layer_name}"
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
                # Under PIECEWISE cudagraph this backend call executes on every
                # model step. Ignore any padded rows beyond the real batch.
                num_actual_tokens = getattr(
                    attn_metadata, "num_actual_tokens", query.shape[0]
                )
                self.capture_query(
                    attn_layer.layer_name,
                    query[:num_actual_tokens],
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

    def _prepare_query_observation_plan(
        self,
        query_start_loc: torch.Tensor,
        observe_query: list[bool],
    ) -> None:
        active: list[tuple[str, int, int]] = []
        for request_id, worker_row, num_new_tokens, observe in zip(
            self._request_ids,
            self._request_rows,
            self._num_new_tokens or [],
            observe_query,
            strict=True,
        ):
            if observe:
                active.append((request_id, worker_row, int(num_new_tokens)))

        if not active:
            return

        device = query_start_loc.device
        rows = torch.as_tensor(
            [worker_row for _, worker_row, _ in active],
            dtype=torch.long,
            device=device,
        )
        starts = query_start_loc.index_select(0, rows).long()

        index_parts: list[torch.Tensor] = []
        slices: list[tuple[str, int, int]] = []
        offset = 0
        for idx, (request_id, _, num_new_tokens) in enumerate(active):
            if num_new_tokens <= 0:
                raise ValueError(
                    "Token-drop observation requires scheduled tokens for "
                    f"{request_id!r}"
                )
            indices = (
                torch.arange(num_new_tokens, dtype=torch.long, device=device)
                + starts[idx]
            )
            index_parts.append(indices)
            slices.append((request_id, offset, offset + num_new_tokens))
            offset += num_new_tokens

        self._query_observation_indices = torch.cat(index_parts)
        self._query_observation_slices = slices

    def capture_query(self, layer_name: str, query: torch.Tensor) -> None:
        if layer_name not in self._kv_caches:
            return
        if self._query_observation_indices is None:
            return

        head_dim = self._kv_caches[layer_name].shape[-1]
        if query.ndim == 2 and query.shape[1] % head_dim == 0:
            query = query.view(query.shape[0], -1, head_dim)
        elif query.ndim != 3 or query.shape[2] != head_dim:
            raise ValueError(
                "Token-drop query must be shaped [tokens, q_heads * head_dim] "
                "or [tokens, q_heads, head_dim]"
            )

        observed = query.index_select(0, self._query_observation_indices)
        for request_id, start, end in self._query_observation_slices:
            self._algorithm_for_request(request_id).observe_query(
                layer_name,
                observed[start:end],
            )

    def _slots_for_request(self, row: int, seq_len: int) -> torch.Tensor:
        assert self._block_table is not None
        device = self._kv_caches[self._layer_names[0]].device
        positions = torch.arange(seq_len, device=device)
        block_ids = self._block_table[row, positions // self._block_size].long()
        return block_ids * self._block_size + positions % self._block_size

    def _compact_request(
        self,
        worker_row: int,
        seq_len: int,
        algorithm: Any,
    ) -> int:
        """Apply one algorithm-selected retained-position set to all KV layers."""
        slots = self._slots_for_request(worker_row, seq_len)
        blocks = slots // self._block_size
        offsets = slots % self._block_size

        layer_keys = {
            name: (
                self._kv_caches[name][:, 0][blocks, offsets]
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
            for name in self._layer_names
        }
        kept = algorithm.select_kept_positions(layer_keys)

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
            if not algorithm.should_compact(
                resident_len=seq_len,
                num_decoded_tokens=self._num_decoded_tokens[local_row],
                num_new_tokens=self._num_new_tokens[local_row],
                is_genuine_decode=self._is_genuine_decode[local_row],
            ):
                continue

            updates[request_id] = self._compact_request(
                worker_row,
                seq_len,
                algorithm,
            )

        return updates

    def _clear_step(self) -> None:
        self._request_ids = []
        self._request_rows = []
        self._query_observation_indices = None
        self._query_observation_slices = []
        self._seq_lens = None
        self._block_table = None
        self._is_genuine_decode = None
        self._num_decoded_tokens = None
        self._num_new_tokens = None
