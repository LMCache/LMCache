# SPDX-License-Identifier: Apache-2.0
"""Worker-local query forwarding and paged-KV compaction for token dropping."""

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

# Third Party
from vllm.forward_context import ForwardContext
import torch

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPTokenDropRequestState,
)
from lmcache.integration.vllm.token_drop import (
    TokenDropAlgorithm,
    TokenDropSpec,
    build_token_drop_algorithm,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )


class KVView:
    """Read-only request-local GPU KV access, materialized on demand."""

    def __init__(
        self, cache: torch.Tensor, blocks: torch.Tensor, offsets: torch.Tensor
    ):
        self._cache = cache
        self._blocks = blocks
        self._offsets = offsets
        self._keys = None
        self._values = None

    def _read(self, plane: int) -> torch.Tensor:
        return (
            self._cache[:, plane][self._blocks, self._offsets]
            .permute(1, 0, 2)
            .contiguous()
            .unsqueeze(0)
        )

    def get_keys(self) -> torch.Tensor:
        if self._keys is None:
            self._keys = self._read(0)
        return self._keys

    def get_values(self) -> torch.Tensor:
        if self._values is None:
            self._values = self._read(1)
        return self._values


class TokenDropWorker:
    """Forward request-local Q and apply algorithm-selected KV compaction."""

    def __init__(self) -> None:
        self._request_algorithms: dict[str, TokenDropAlgorithm] = {}
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._block_size = 0
        self._kv_caches_validated = False

        self._request_ids: list[str] = []
        self._request_rows: list[int] = []
        self._seq_lens: list[int] | None = None
        self._block_table: torch.Tensor | None = None
        self._compact_now: list[bool] | None = None
        self._num_new_tokens: list[int] | None = None

        self._query_observation_indices: torch.Tensor | None = None
        self._query_observation_slices: list[tuple[str, int, int]] = []
        self._observed_queries: dict[str, torch.Tensor] = {}

        self._query_hooks_installed = False
        self._query_hook_originals: dict[
            str, tuple[FlashAttentionImpl, Callable[..., torch.Tensor]]
        ] = {}

    def _algorithm_from_state(
        self, state: LMCacheMPTokenDropRequestState
    ) -> TokenDropAlgorithm:
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

    def _algorithm_for_request(self, request_id: str) -> TokenDropAlgorithm:
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
                raise ValueError("Token dropping MVP requires contiguous NHD KV layout")
            if block_size and kv_cache.shape[2] != block_size:
                raise ValueError("Token dropping requires one shared KV block size")
            block_size = int(kv_cache.shape[2])

        self._block_size = block_size
        self._kv_caches_validated = True

    def drop_requests(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self._request_algorithms.pop(request_id, None)

    def prepare_forward(
        self,
        forward_context: ForwardContext,
        request_states: list[LMCacheMPTokenDropRequestState],
    ) -> None:
        if not request_states:
            self.remove_query_hooks()
            self._clear_step()
            return

        self._ensure_kv_caches_compatible()
        ordered_states = list(request_states)
        query_lens = [int(state.num_new_tokens) for state in ordered_states]
        if any(state.resident_kv_tokens is None for state in ordered_states):
            raise RuntimeError(
                "Token-drop scheduler metadata is missing resident KV length"
            )
        physical_seq_lens = [
            state.resident_kv_tokens
            for state in ordered_states
            if state.resident_kv_tokens is not None
        ]
        if any(
            query_len <= 0 or physical_len < query_len
            for query_len, physical_len in zip(
                query_lens, physical_seq_lens, strict=True
            )
        ):
            raise ValueError("Token dropping received invalid physical/query lengths")

        algorithms = [self._algorithm_from_state(state) for state in ordered_states]
        observe_counts = []
        compact_now = []
        for state, algorithm, physical_len in zip(
            ordered_states, algorithms, physical_seq_lens, strict=True
        ):
            phase: Literal["prefill", "decode"] = (
                "decode" if state.is_genuine_decode else "prefill"
            )
            decoded_before = int(state.num_decoded_tokens)
            count = algorithm.should_observe_token_queries(phase, decoded_before)
            if type(count) is not int or count < 0:
                raise ValueError("Token-drop Q capture count must be nonnegative int")
            observe_counts.append(min(count, int(state.num_new_tokens)))
            compact_now.append(
                bool(algorithm.should_compact_kv(phase, physical_len, decoded_before))
            )

        if not any(observe_counts) and not any(compact_now):
            self.remove_query_hooks()
            self._clear_step()
            return

        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict) or not attn_metadata:
            raise ValueError("Token dropping MVP requires eager attention metadata")

        representative = cast(
            "FlashAttentionMetadata", next(iter(attn_metadata.values()))
        )
        request_rows = [int(state.worker_row) for state in ordered_states]
        if any(row < 0 for row in request_rows):
            raise RuntimeError("Token-drop metadata is missing worker row identity")
        if (
            request_rows
            and max(request_rows) + 1 >= representative.query_start_loc.shape[0]
        ):
            raise ValueError("Token-drop request/attention row mismatch")

        if any(observe_counts):
            self.install_query_hooks(forward_context.no_compile_layers)
        else:
            self.remove_query_hooks()

        self.begin_step(
            [state.request_id for state in ordered_states],
            representative,
            physical_seq_lens=physical_seq_lens,
            request_rows=request_rows,
            observe_counts=observe_counts,
            compact_now=compact_now,
            num_new_tokens=query_lens,
        )

    def begin_step(
        self,
        request_ids: list[str],
        attn_metadata: "FlashAttentionMetadata",
        *,
        physical_seq_lens: list[int],
        request_rows: list[int] | None = None,
        observe_counts: list[int] | None = None,
        compact_now: list[bool] | None = None,
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

        self._num_new_tokens = (
            [1] * num_reqs if num_new_tokens is None else list(num_new_tokens)
        )
        observe_counts = (
            [0] * num_reqs if observe_counts is None else list(observe_counts)
        )
        self._compact_now = (
            [False] * num_reqs if compact_now is None else list(compact_now)
        )
        if (
            len(self._num_new_tokens) != num_reqs
            or len(observe_counts) != num_reqs
            or len(self._compact_now) != num_reqs
        ):
            raise ValueError("Token-drop request/step fact count mismatch")

        self._prepare_query_observation_plan(query_start_loc, observe_counts)
        self._seq_lens = list(physical_seq_lens)
        self._block_table = attn_metadata.block_table

    def install_query_hooks(self, no_compile_layers: dict[str, "Attention"]) -> None:
        if self._query_hooks_installed:
            return

        missing = set(self._layer_names) - set(no_compile_layers)
        if missing:
            raise RuntimeError(
                f"Token-drop attention layers are missing: {sorted(missing)}"
            )

        originals: dict[
            str, tuple[FlashAttentionImpl, Callable[..., torch.Tensor]]
        ] = {}
        for layer_name in self._layer_names:
            layer = no_compile_layers[layer_name]
            impl = getattr(layer, "impl", None)
            original_forward = getattr(impl, "forward", None)
            if original_forward is None:
                raise RuntimeError(
                    f"Token-drop attention backend is missing for {layer_name}"
                )
            originals[layer_name] = (impl, original_forward)

        for impl, forward_impl in originals.values():

            def forward_with_query_capture(
                attn_layer: "Attention",
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                kv_cache: torch.Tensor,
                attn_metadata: "FlashAttentionMetadata",
                *args: Any,
                _forward: Callable[..., torch.Tensor] = forward_impl,
                **kwargs: Any,
            ) -> torch.Tensor:
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

            setattr(impl, "forward", forward_with_query_capture)  # noqa: B010

        self._query_hook_originals = originals
        self._query_hooks_installed = True

    def remove_query_hooks(self) -> None:
        if not self._query_hooks_installed:
            return

        for impl, original_forward in self._query_hook_originals.values():
            setattr(impl, "forward", original_forward)  # noqa: B010
        self._query_hook_originals.clear()
        self._query_hooks_installed = False

    def _prepare_query_observation_plan(
        self,
        query_start_loc: torch.Tensor,
        observe_counts: list[int],
    ) -> None:
        active: list[tuple[str, int, int, int]] = []
        for request_id, worker_row, num_new_tokens, count in zip(
            self._request_ids,
            self._request_rows,
            self._num_new_tokens or [],
            observe_counts,
            strict=True,
        ):
            if type(count) is not int or count < 0 or count > num_new_tokens:
                raise ValueError("Token-drop invalid Q capture count")
            if count:
                active.append((request_id, worker_row, int(num_new_tokens), count))

        if not active:
            return

        device = query_start_loc.device
        rows = torch.as_tensor(
            [worker_row for _, worker_row, _, _ in active],
            dtype=torch.long,
            device=device,
        )
        starts = query_start_loc.index_select(0, rows).long()

        index_parts: list[torch.Tensor] = []
        slices: list[tuple[str, int, int]] = []
        offset = 0
        for idx, (request_id, _, num_new_tokens, count) in enumerate(active):
            if num_new_tokens <= 0:
                raise ValueError(
                    "Token-drop observation requires scheduled tokens for "
                    f"{request_id!r}"
                )
            indices = (
                torch.arange(count, dtype=torch.long, device=device)
                + starts[idx]
                + num_new_tokens
                - count
            )
            index_parts.append(indices)
            slices.append((request_id, offset, offset + count))
            offset += count

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

        self._observed_queries[layer_name] = query.index_select(
            0,
            self._query_observation_indices,
        )

    def _flush_query_observations(self) -> None:
        if not self._observed_queries:
            return

        for request_id, start, end in self._query_observation_slices:
            self._algorithm_for_request(request_id).observe_token_queries(
                {
                    layer_name: observed[start:end]
                    for layer_name, observed in self._observed_queries.items()
                }
            )
        self._observed_queries.clear()

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
        algorithm: TokenDropAlgorithm,
    ) -> int:
        """Compact each KV head using the positions chosen for its own layer."""
        # Standard
        from collections.abc import Mapping

        slots = self._slots_for_request(worker_row, seq_len)
        blocks = slots // self._block_size
        offsets = slots % self._block_size
        views = {
            name: KVView(self._kv_caches[name], blocks, offsets)
            for name in self._layer_names
        }
        kept_by_layer = algorithm.select_kept_token_positions(views)
        if not isinstance(kept_by_layer, Mapping):
            raise RuntimeError("Token-drop algorithm must return positions by layer")
        if set(kept_by_layer) != set(self._layer_names):
            raise RuntimeError("Token-drop retained layer names mismatch")

        resident_len = None
        for name in self._layer_names:
            indices = kept_by_layer[name]
            kv_heads = int(self._kv_caches[name].shape[3])
            if (
                not isinstance(indices, torch.Tensor)
                or indices.ndim != 2
                or indices.shape[0] != kv_heads
                or indices.device != slots.device
                or indices.dtype not in (torch.int32, torch.int64)
            ):
                raise RuntimeError(
                    "Token-drop positions must be [kv_heads, kept_tokens] "
                    "integer tensors on the KV device"
                )
            count = int(indices.shape[1])
            if count == 0 or count > seq_len:
                raise RuntimeError("Token-drop invalid retained token count")
            if resident_len is None:
                resident_len = count
            elif resident_len != count:
                raise RuntimeError("All heads/layers must retain the same count")
        assert resident_len is not None
        destination = slots[:resident_len]
        dst_blocks = destination // self._block_size
        dst_offsets = destination % self._block_size
        for name, view in views.items():
            indices = kept_by_layer[name].long()
            for plane, values in enumerate((view.get_keys(), view.get_values())):
                # values: [1, kv_heads, current_tokens, head_dim]
                per_head = values[0]
                gather_idx = indices.unsqueeze(-1).expand(-1, -1, per_head.shape[-1])
                retained = torch.gather(per_head, 1, gather_idx)
                self._kv_caches[name][:, plane][dst_blocks, dst_offsets] = (
                    retained.permute(1, 0, 2)
                )
        return resident_len

    def compact(self) -> dict[str, int]:
        if (
            self._seq_lens is None
            or self._block_table is None
            or self._compact_now is None
        ):
            return {}

        self._flush_query_observations()

        updates: dict[str, int] = {}
        for local_row, (request_id, worker_row, compact_now) in enumerate(
            zip(
                self._request_ids,
                self._request_rows,
                self._compact_now,
                strict=True,
            )
        ):
            if not compact_now:
                continue

            updates[request_id] = self._compact_request(
                worker_row,
                self._seq_lens[local_row],
                self._algorithm_for_request(request_id),
            )

        return updates

    def _clear_step(self) -> None:
        self._request_ids = []
        self._request_rows = []
        self._query_observation_indices = None
        self._query_observation_slices = []
        self._observed_queries.clear()
        self._seq_lens = None
        self._block_table = None
        self._compact_now = None
        self._num_new_tokens = None
