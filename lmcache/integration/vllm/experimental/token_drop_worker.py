# SPDX-License-Identifier: Apache-2.0
"""Request-local algorithm decisions and query observation for token dropping."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, cast
import logging

# Third Party
import torch

# First Party
from lmcache.integration.vllm.experimental.token_drop import (
    LMCacheMPTokenDropRequestState,
    TokenDropSpec,
    build_token_drop_algorithm,
)
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    install_token_drop_query_hooks,
    remove_token_drop_query_hooks,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

logger = logging.getLogger(__name__)


class TokenDropWorker:
    """Track request-local Q and upcoming KV compaction decisions."""

    def __init__(self) -> None:
        self._request_algorithms: dict[str, Any] = {}
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []

        self._request_ids: list[str] = []
        self._request_rows: list[int] = []
        self._compact_now: list[bool] | None = None

        self._query_observation_indices: torch.Tensor | None = None
        self._query_observation_slices: list[tuple[str, int, int]] = []
        self._observed_queries: dict[str, torch.Tensor] = {}

        self._original_attention_forwards: dict[str, tuple[object, object]] = {}

    def _algorithm_from_state(self, state: LMCacheMPTokenDropRequestState) -> Any:
        existing = self._request_algorithms.get(state.request_id)
        if existing is not None:
            return existing

        algorithm = build_token_drop_algorithm(
            TokenDropSpec(
                algorithm=state.algorithm,
                config=dict(state.config),
            )
        )
        self._request_algorithms[state.request_id] = algorithm
        return algorithm

    def finish_requests(self, request_ids: set[str]) -> set[str]:
        completed = request_ids & self._request_algorithms.keys()
        self.drop_requests(request_ids)
        return completed

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # Registration is shared with normal LMCache serving. Validate the
        # token-drop runtime requirements only after a request opts in.
        self._kv_caches = dict(kv_caches)
        self._layer_names = list(kv_caches)

    def drop_requests(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self._request_algorithms.pop(request_id, None)

    def prepare_forward(
        self,
        forward_context: ForwardContext,
        request_states: list[LMCacheMPTokenDropRequestState],
    ) -> None:
        # A previous step's hooks must not survive failed preparation.
        remove_token_drop_query_hooks(self)
        if not request_states:
            self._clear_step()
            return

        query_lens = [int(state.num_new_tokens) for state in request_states]
        physical_seq_lens = [state.resident_kv_tokens for state in request_states]
        algorithms = [self._algorithm_from_state(state) for state in request_states]
        observe_counts = []
        compact_now = []
        for state, algorithm, physical_len in zip(
            request_states, algorithms, physical_seq_lens, strict=True
        ):
            decoded_before = int(state.num_decoded_tokens)
            count = int(
                algorithm.should_observe_token_queries(state.phase, decoded_before)
            )
            observe_counts.append(min(count, int(state.num_new_tokens)))
            compact_now.append(
                bool(
                    algorithm.should_compact_kv(
                        state.phase, physical_len, decoded_before
                    )
                )
            )

        if not any(observe_counts) and not any(compact_now):
            self._clear_step()
            return

        attn_metadata = forward_context.attn_metadata
        if not attn_metadata:
            raise ValueError("Token dropping MVP requires eager attention metadata")

        representative = cast(
            "FlashAttentionMetadata", next(iter(attn_metadata.values()))
        )
        request_rows = [int(state.worker_row) for state in request_states]
        try:
            self.begin_step(
                [state.request_id for state in request_states],
                representative,
                request_rows=request_rows,
                observe_counts=observe_counts,
                compact_now=compact_now,
                num_new_tokens=query_lens,
            )
            if any(observe_counts):
                install_token_drop_query_hooks(self, forward_context.no_compile_layers)
        except Exception:
            remove_token_drop_query_hooks(self)
            self._clear_step()
            raise

    def begin_step(
        self,
        request_ids: list[str],
        attn_metadata: "FlashAttentionMetadata",
        *,
        request_rows: list[int],
        observe_counts: list[int],
        compact_now: list[bool],
        num_new_tokens: list[int],
    ) -> None:
        if getattr(attn_metadata, "use_cascade", False):
            raise ValueError("Token dropping MVP does not support cascade attention")

        self._clear_step()
        self._request_ids = list(request_ids)
        self._request_rows = list(request_rows)

        query_start_loc = attn_metadata.query_start_loc
        if self._request_rows and (
            min(self._request_rows) < 0
            or max(self._request_rows) + 1 >= query_start_loc.shape[0]
        ):
            raise ValueError("Token-drop request/query segmentation mismatch")
        self._compact_now = list(compact_now)
        self._prepare_query_observation_plan(
            query_start_loc, observe_counts, num_new_tokens
        )

    def _prepare_query_observation_plan(
        self,
        query_start_loc: torch.Tensor,
        observe_counts: list[int],
        scheduled_tokens: list[int],
    ) -> None:
        active: list[tuple[str, int, int, int]] = []
        for request_id, worker_row, num_new_tokens, count in zip(
            self._request_ids,
            self._request_rows,
            scheduled_tokens,
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
        if self._query_observation_slices and set(self._observed_queries) != set(
            self._layer_names
        ):
            # No KV writes or algorithm history updates yet. Skip this step
            # rather than letting incomplete Q history corrupt selection.
            logger.warning("Skipping token-drop compaction: incomplete Q capture")
            assert self._compact_now is not None
            self._compact_now = [False] * len(self._compact_now)
            self._observed_queries.clear()
            return
        if not self._observed_queries:
            return

        for request_id, start, end in self._query_observation_slices:
            try:
                self._request_algorithms[request_id].observe_token_queries(
                    {
                        layer_name: observed[start:end]
                        for layer_name, observed in self._observed_queries.items()
                    }
                )
            except (ValueError, RuntimeError) as exc:
                # No KV writes have occurred yet. A poisoned CUDA stream,
                # however, must not be treated as a recoverable error.
                observed = next(iter(self._observed_queries.values()))
                if observed.is_cuda:
                    torch.cuda.current_stream(device=observed.device).synchronize()
                assert self._compact_now is not None
                self._compact_now[self._request_ids.index(request_id)] = False
                logger.warning(
                    "Skipping token-drop compaction after Q observation error: %s",
                    exc,
                )
        self._observed_queries.clear()

    def _clear_step(self) -> None:
        self._request_ids = []
        self._request_rows = []
        self._query_observation_indices = None
        self._query_observation_slices = []
        self._observed_queries.clear()
        self._compact_now = None
