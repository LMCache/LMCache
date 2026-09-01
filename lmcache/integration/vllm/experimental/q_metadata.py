# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connector metadata for the query-tensor capture path.

Kept apart from ``lmcache_mp_metadata`` because the query path is a separable
feature: its producer (the connector's per-step capture window), its buffer
(``lmcache.sdk.qringbuffer``) and its server module (``qstore``) all live in
their own modules, and this dataclass is consumed only by those.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING

# First Party
from lmcache.utils import cdiv, init_logger, round_down
from lmcache.v1.multiprocess.group_view import slice_block_ids_per_group

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestTracker,
    )

logger = init_logger(__name__)


@dataclass
class LMCacheMPQRequestMetadata:
    """One request's token range computed in the current forward step.
    Emitted for every scheduled request, every step, so the worker can attribute
    the step's query rows to token indices and accumulate them across steps.

    Attributes:
        request_id: The request the window belongs to.
        start: First token index computed in this step, inclusive.
        end: Last token index computed in this step, exclusive.
        block_ids: Engine block ids covering ``[block_start, ...)`` in token
            order, indexed by engine KV cache group.
            May cover tokens outside ``[start, end)``.
        block_start: First token index covered by ``block_ids``.
    """

    request_id: str
    start: int
    end: int
    block_ids: list[list[int]]
    block_start: int

    @staticmethod
    def BuildQCaptureWindow(
        tracker: LMCacheMPRequestTracker,
        num_scheduled_tokens: int,
        group_tokens_per_block: list[int],
    ) -> "LMCacheMPQRequestMetadata | None":
        """Build this step's query-capture window for one request.

        Args:
            tracker: the request tracker to build the q metadata from.
            num_scheduled_tokens: tokens scheduled for this request in this step.
            group_tokens_per_block: per-engine-group tokens covered by one
                paged chunk (one block ID) of that group, i.e. the group's
                KV cache spec ``block_size``. Must each divide
                ``lmcache_tokens_per_chunk`` (hybrid models can mix different values).

        Returns:
            The capture window, or None when the request contributes no rows
            or its allocated blocks do not yet cover the computed range.
        """
        num_engine_groups = len(group_tokens_per_block)

        if num_scheduled_tokens <= 0 or not group_tokens_per_block:
            logger.info(
                "No q window for %s: num_scheduled_tokens=%d, num_groups=%d",
                tracker.request_id,
                num_scheduled_tokens,
                num_engine_groups,
            )
            return None

        computed_tokens = tracker.num_scheduled_tokens + max(
            tracker.num_vllm_hit_tokens, tracker.num_lmcache_hit_tokens
        )

        start_token_idx = computed_tokens - num_scheduled_tokens
        end_token_idx = computed_tokens
        if start_token_idx < 0:
            logger.info(
                "No q window for %s: negative start (cum_scheduled=%d, "
                "vllm_hit=%d, lmcache_hit=%d, num_scheduled=%d)",
                tracker.request_id,
                tracker.num_scheduled_tokens,
                tracker.num_vllm_hit_tokens,
                tracker.num_lmcache_hit_tokens,
                num_scheduled_tokens,
            )
            return None

        # Widen to whole blocks: slice_block_ids_per_group rejects unaligned
        # bounds, and a window edge may sit inside a block. Every group's
        # tokens_per_block divides the largest one in practice (all divide the
        # LMCache chunk size), so aligning to the max aligns to all.
        tokens_per_block = max(group_tokens_per_block)
        block_start = round_down(start_token_idx, tokens_per_block)
        block_end = cdiv(end_token_idx, tokens_per_block) * tokens_per_block

        allocated_lengths = tracker.num_allocated_blocks()
        allocated_tokens = min(
            allocated_lengths.get(engine_group_idx, 0)
            * group_tokens_per_block[engine_group_idx]
            for engine_group_idx in range(num_engine_groups)
        )
        if allocated_tokens < block_end:
            logger.info(
                "No q window for %s: allocated_tokens=%d < block_end=%d "
                "(window [%d, %d), blocks=%s)",
                tracker.request_id,
                allocated_tokens,
                block_end,
                start_token_idx,
                end_token_idx,
                allocated_lengths,
            )
            return None

        return LMCacheMPQRequestMetadata(
            request_id=tracker.request_id,
            start=start_token_idx,
            end=end_token_idx,
            block_ids=slice_block_ids_per_group(
                tracker.allocated_block_ids,
                group_tokens_per_block,
                block_start,
                block_end,
            ),
            block_start=block_start,
        )
