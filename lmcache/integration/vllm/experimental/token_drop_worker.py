# SPDX-License-Identifier: Apache-2.0
"""Worker-local query forwarding and paged-KV compaction for token dropping."""

# Standard
from typing import TYPE_CHECKING, Any, Literal, cast
import logging

# Third Party
from vllm.forward_context import ForwardContext
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
from lmcache.v1.platform.stream import current_stream, synchronize_stream

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata


logger = logging.getLogger(__name__)


class TokenDropWorker:
    """Forward request-local Q and apply algorithm-selected KV compaction."""

    def __init__(self) -> None:
        self._request_algorithms: dict[str, Any] = {}
        self._request_generations: dict[str, int] = {}
        # A plugin may mutate its Q history before a recoverable pre-write error.
        # Disable further compaction for that request generation, not its decoding.
        self._failed_generations: dict[str, int] = {}
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._block_size = 0
        self._kv_caches_validated = False

        self._request_ids: list[str] = []
        self._request_rows: list[int] = []
        self._seq_lens: list[int] | None = None
        self._block_table: torch.Tensor | None = None
        self._compact_now: list[bool] | None = None

        self._query_observation_indices: torch.Tensor | None = None
        self._query_observation_slices: list[tuple[str, int, int]] = []
        self._observed_queries: dict[str, torch.Tensor] = {}

        self._original_attention_forwards: dict[str, tuple[object, object, bool]] = {}

    def _algorithm_from_state(self, state: LMCacheMPTokenDropRequestState) -> Any:
        existing = self._request_algorithms.get(state.request_id)
        if (
            existing is not None
            and self._request_generations.get(state.request_id) == state.generation
        ):
            return existing
        if state.generation < 1:
            raise ValueError("Token dropping requires a request generation")
        self.drop_requests({state.request_id})

        algorithm = build_token_drop_algorithm(
            TokenDropSpec(
                algorithm=state.algorithm,
                config=dict(state.config),
            )
        )
        self._request_algorithms[state.request_id] = algorithm
        self._request_generations[state.request_id] = state.generation
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
            if kv_cache.device.type != "cuda":
                raise ValueError("Token dropping requires CUDA-resident KV cache")
            if kv_cache.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("Token dropping requires FP16 or BF16 KV cache")
            if block_size and kv_cache.shape[2] != block_size:
                raise ValueError("Token dropping requires one shared KV block size")
            block_size = int(kv_cache.shape[2])

        self._block_size = block_size
        self._kv_caches_validated = True

    def drop_requests(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self._request_algorithms.pop(request_id, None)
            self._request_generations.pop(request_id, None)
            self._failed_generations.pop(request_id, None)

    def _disable_compaction(self, request_id: str) -> None:
        self._failed_generations[request_id] = self._request_generations[request_id]
        logger.warning(
            "Disabling further token-drop compaction for request %s after a "
            "recoverable pre-write failure",
            request_id,
        )

    def prepare_forward(
        self,
        forward_context: ForwardContext,
        request_states: list[LMCacheMPTokenDropRequestState],
    ) -> None:
        # No previous step's backend hook may survive into a new batch.
        remove_token_drop_query_hooks(self)
        self._clear_step()
        request_states = [
            state
            for state in request_states
            if self._failed_generations.get(state.request_id) != state.generation
        ]
        if not request_states:
            return

        try:
            self._ensure_kv_caches_compatible()
            query_lens = [int(state.num_new_tokens) for state in request_states]
            physical_seq_lens = [state.resident_kv_tokens for state in request_states]
            algorithms = [self._algorithm_from_state(state) for state in request_states]
            observe_counts: list[int] = []
            compact_now: list[bool] = []
            for state, algorithm, physical_len in zip(
                request_states, algorithms, physical_seq_lens, strict=True
            ):
                phase: Literal["prefill", "decode"] = state.phase
                decoded_before = int(state.num_decoded_tokens)
                requested = algorithm.should_observe_token_queries(
                    phase, decoded_before
                )
                if type(requested) is not int or requested < 0:
                    raise ValueError("Token-drop invalid Q observation count")
                observe_counts.append(min(requested, int(state.num_new_tokens)))
                compact_now.append(
                    bool(
                        algorithm.should_compact_kv(phase, physical_len, decoded_before)
                    )
                )

            attn_metadata = forward_context.attn_metadata
            # Auto-selected backends must be checked even on TD no-op steps;
            # an idle step without attention metadata stays a no-op.
            if attn_metadata:
                # Third Party
                from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

                if any(
                    not isinstance(meta, FlashAttentionMetadata)
                    for meta in attn_metadata.values()
                ):
                    raise ValueError("Token dropping requires FlashAttention metadata")

            if not any(observe_counts) and not any(compact_now):
                return
            if not attn_metadata:
                raise ValueError("Token dropping requires attention metadata")

            representative = cast(
                "FlashAttentionMetadata", next(iter(attn_metadata.values()))
            )
            self.begin_step(
                [state.request_id for state in request_states],
                representative,
                physical_seq_lens=physical_seq_lens,
                request_rows=[int(state.worker_row) for state in request_states],
                observe_counts=observe_counts,
                compact_now=compact_now,
                num_new_tokens=query_lens,
            )
            # Only install after all preparation that can fail is complete.
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
        physical_seq_lens: list[int],
        request_rows: list[int],
        observe_counts: list[int],
        compact_now: list[bool],
        num_new_tokens: list[int],
    ) -> None:
        self._ensure_kv_caches_compatible()
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
        self._seq_lens = list(physical_seq_lens)
        self._block_table = attn_metadata.block_table

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
            for request_id, _, _ in self._query_observation_slices:
                self._disable_compaction(request_id)
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
                synchronize_stream(observed.device, current_stream(observed.device))
                assert self._compact_now is not None
                self._compact_now[self._request_ids.index(request_id)] = False
                self._disable_compaction(request_id)
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
        self._seq_lens = None
        self._block_table = None
        self._compact_now = None
