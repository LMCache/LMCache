# SPDX-License-Identifier: Apache-2.0
"""Regression tests for bookkeeping bugs in lmcache_mp_connector_0180.

Bug 1 – GetStoreMetadata must include lmcache-hit blocks when computing
         the upper bound (computed_blocks), otherwise num_staging_blocks
         goes negative and new blocks are never stored.

Bug 2 – _process_cached_requests must use the *incremental*
         num_scheduled_tokens from SchedulerOutput, not the cumulative
         num_computed_tokens, to stay consistent with
         _process_new_requests.
"""

# Standard
from dataclasses import dataclass, field
from unittest.mock import MagicMock
import sys
import types

# ---------------------------------------------------------------------------
# Stub out vLLM sub-modules that are absent in the installed version
# (the connector targets vLLM >= 0.18.0, but tests run on 0.7.x).
# We only create stubs for *missing* modules; existing ones are left
# untouched.
# ---------------------------------------------------------------------------

_MISSING_MODULES = [
    "vllm.distributed.kv_transfer.kv_connector.v1",
    "vllm.distributed.kv_transfer.kv_connector.v1.base",
    "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
    "vllm.distributed.kv_events",
    "vllm.v1.attention.backend",
    "vllm.v1.core.sched",
    "vllm.v1.core.sched.output",
]


for _m in _MISSING_MODULES:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

# Provide sentinel classes the connector expects at import time
_base = sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
_base.KVConnectorBase_V1 = type(  # type: ignore[attr-defined]
    "KVConnectorBase_V1",
    (),
    {"__init_subclass__": classmethod(lambda cls, **kw: None)},
)
_base.KVConnectorMetadata = type(  # type: ignore[attr-defined]
    "KVConnectorMetadata", (), {}
)
_base.KVConnectorRole = MagicMock()  # type: ignore[attr-defined]

_sched_out = sys.modules["vllm.v1.core.sched.output"]
_sched_out.SchedulerOutput = MagicMock  # type: ignore[attr-defined]

_attn = sys.modules["vllm.v1.attention.backend"]
_attn.AttentionMetadata = MagicMock  # type: ignore[attr-defined]

# Third Party
# Add KVConnectorOutput to the existing vllm.v1.outputs if missing
import vllm.v1.outputs as _outputs_mod  # noqa: E402

if not hasattr(_outputs_mod, "KVConnectorOutput"):
    _outputs_mod.KVConnectorOutput = MagicMock

# First Party
# Now import the connector module
from lmcache.integration.vllm.lmcache_mp_connector_0180 import (  # noqa: E402
    LMCacheMPConnector,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestTracker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(
    request_id: str = "req-1",
    num_tokens: int = 4096,
    num_block_hashes: int = 64,
    num_allocated_blocks: int = 64,
    num_scheduled_tokens: int = 0,
    num_stored_blocks: int = 0,
    num_lmcache_hit_blocks: int = 0,
    num_vllm_hit_blocks: int = 0,
) -> LMCacheMPRequestTracker:
    """Build a tracker without needing a real vLLM Request."""
    tracker = object.__new__(LMCacheMPRequestTracker)
    tracker.request_id = request_id
    tracker.all_token_ids = list(range(num_tokens))
    tracker.block_hashes = [MagicMock()] * num_block_hashes
    tracker.allocated_block_ids = list(range(num_allocated_blocks))
    tracker.num_scheduled_tokens = num_scheduled_tokens
    tracker.num_stored_blocks = num_stored_blocks
    tracker.num_lmcache_hit_blocks = num_lmcache_hit_blocks
    tracker.num_vllm_hit_blocks = num_vllm_hit_blocks
    return tracker


# ---------------------------------------------------------------------------
# Bug 1: GetStoreMetadata – computed_blocks must include hit blocks
# ---------------------------------------------------------------------------


class TestGetStoreMetadataWithHitBlocks:
    """Verify that lmcache-hit blocks are counted in the upper
    bound so that newly computed blocks can still be stored.

    Scenario (block_size=64, chunk_size=256, blocks_in_chunk=4):
      - Request has 4096 tokens (64 blocks).
      - LMCache lookup hit 60 blocks (3840 tokens).
      - Scheduler schedules the remaining 256 tokens (4 blocks).
      - num_stored_blocks is set to 60 (from the lookup hit).

    Without the fix, computed_blocks = 256 // 64 = 4,
    num_staging_blocks = 4 - 60 = -56 -> no store.

    With the fix, computed_blocks = 4 + 60 = 64,
    num_staging_blocks = 64 - 60 = 4 -> stores 1 chunk.
    """

    BLOCK_SIZE = 64
    BLOCKS_IN_CHUNK = 4  # chunk_size=256, block_size=64

    def test_new_blocks_stored_after_lmcache_hit(self):
        """New blocks after a partial hit must be stored."""
        tracker = _make_tracker(
            num_tokens=4096,
            num_block_hashes=64,
            num_allocated_blocks=64,
            num_scheduled_tokens=256,
            num_stored_blocks=60,
            num_lmcache_hit_blocks=60,
        )

        result = LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker, self.BLOCKS_IN_CHUNK, self.BLOCK_SIZE
        )

        assert result is not None, (
            "GetStoreMetadata must return metadata for the "
            "4 new blocks, but returned None "
            "(regression of Bug 1)."
        )
        assert result.direction == "STORE"
        assert len(result.op.block_ids) == 4
        assert result.op.start == 60 * self.BLOCK_SIZE
        assert result.op.end == 64 * self.BLOCK_SIZE

    def test_no_store_when_nothing_new(self):
        """If all blocks are already stored, return None."""
        tracker = _make_tracker(
            num_tokens=4096,
            num_block_hashes=64,
            num_allocated_blocks=64,
            num_scheduled_tokens=256,
            num_stored_blocks=64,
            num_lmcache_hit_blocks=60,
        )

        result = LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker, self.BLOCKS_IN_CHUNK, self.BLOCK_SIZE
        )
        assert result is None

    def test_hit_blocks_not_re_stored(self):
        """Hit blocks must not be re-stored; only new blocks
        beyond num_stored_blocks are included."""
        tracker = _make_tracker(
            num_tokens=4096,
            num_block_hashes=64,
            num_allocated_blocks=64,
            num_scheduled_tokens=512,  # 8 new blocks
            num_stored_blocks=60,
            num_lmcache_hit_blocks=60,
        )

        result = LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker, self.BLOCKS_IN_CHUNK, self.BLOCK_SIZE
        )

        assert result is not None
        # start should be at block 60, not block 0
        assert result.op.start == 60 * self.BLOCK_SIZE
        # min(64 hashes, 64 alloc, 8+60=68) = 64
        # staging = 64 - 60 = 4 blocks (1 chunk)
        assert len(result.op.block_ids) == 4

    def test_no_hit_blocks_still_works(self):
        """When there are no lmcache hits, store works."""
        tracker = _make_tracker(
            num_tokens=4096,
            num_block_hashes=64,
            num_allocated_blocks=64,
            num_scheduled_tokens=4096,
            num_stored_blocks=0,
            num_lmcache_hit_blocks=0,
        )

        result = LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker, self.BLOCKS_IN_CHUNK, self.BLOCK_SIZE
        )

        assert result is not None
        assert result.op.start == 0
        assert result.op.end == 4096
        assert len(result.op.block_ids) == 64


# ---------------------------------------------------------------------------
# Bug 2: _process_cached_requests – incremental vs cumulative
# ---------------------------------------------------------------------------


@dataclass
class _FakeCachedReqs:
    """Minimal stand-in for scheduled_cached_reqs."""

    req_ids: list[str] = field(default_factory=list)
    new_block_ids: list = field(default_factory=list)
    resumed_req_ids: set[str] = field(default_factory=set)


class TestProcessCachedRequestsIncremental:
    """Verify that _process_cached_requests uses incremental
    num_scheduled_tokens, not cumulative num_computed_tokens.

    Scenario (block_size=64, chunk_size=256, blocks_in_chunk=4):
      - Request has 4096 tokens.
      - Round 1 (new_req): schedules 2048 -> tracker = 2048.
      - Round 2 (cached_req): schedules another 2048.
        - Correct (incremental): tracker += 2048 -> 4096.
        - Wrong (cumulative):   tracker += 4096 -> 6144.
    """

    BLOCK_SIZE = 64
    BLOCKS_IN_CHUNK = 4

    def _make_connector_with_tracker(
        self,
        request_id: str,
        tracker: LMCacheMPRequestTracker,
    ) -> LMCacheMPConnector:
        """Build a connector with a pre-populated tracker."""
        connector = object.__new__(LMCacheMPConnector)
        connector._role = MagicMock()
        connector.vllm_block_size = self.BLOCK_SIZE
        connector.scheduler_adapter = MagicMock()
        connector.scheduler_adapter.num_blocks_per_chunk.return_value = (
            self.BLOCKS_IN_CHUNK
        )
        connector.request_trackers = {request_id: tracker}
        return connector

    def test_incremental_tokens_after_two_rounds(self):
        """After two rounds, tracker.num_scheduled_tokens
        must equal the sum of incremental values."""
        request_id = "req-chunked"

        tracker = _make_tracker(
            request_id=request_id,
            num_tokens=4096,
            num_block_hashes=64,
            num_allocated_blocks=64,
            num_scheduled_tokens=2048,
            num_stored_blocks=32,
            num_lmcache_hit_blocks=0,
        )

        connector = self._make_connector_with_tracker(request_id, tracker)

        cached_reqs = _FakeCachedReqs(
            req_ids=[request_id],
            new_block_ids=[None],
            resumed_req_ids=set(),
        )

        scheduler_output = MagicMock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = cached_reqs
        scheduler_output.num_scheduled_tokens = {request_id: 2048}

        connector.build_connector_meta(scheduler_output)

        assert tracker.num_scheduled_tokens == 4096, (
            "tracker.num_scheduled_tokens should be 4096 "
            f"(2048+2048), got "
            f"{tracker.num_scheduled_tokens} "
            "(regression of Bug 2)."
        )

    def test_store_metadata_correct_after_two_rounds(self):
        """Store metadata after round 2 should cover only
        the newly computed blocks."""
        request_id = "req-chunked-2"

        tracker = _make_tracker(
            request_id=request_id,
            num_tokens=4096,
            num_block_hashes=64,
            num_allocated_blocks=64,
            num_scheduled_tokens=2048,
            num_stored_blocks=32,
            num_lmcache_hit_blocks=0,
        )

        connector = self._make_connector_with_tracker(request_id, tracker)

        cached_reqs = _FakeCachedReqs(
            req_ids=[request_id],
            new_block_ids=[None],
            resumed_req_ids=set(),
        )

        scheduler_output = MagicMock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = cached_reqs
        scheduler_output.num_scheduled_tokens = {request_id: 2048}

        metadata = connector.build_connector_meta(scheduler_output)

        store_reqs = [r for r in metadata.requests if r.direction == "STORE"]
        assert len(store_reqs) == 1

        op = store_reqs[0].op
        assert op.start == 32 * self.BLOCK_SIZE
        assert op.end == 64 * self.BLOCK_SIZE
        assert len(op.block_ids) == 32

    def test_decode_single_token_no_inflation(self):
        """During decode (1 token/step), the tracker should
        increment by exactly 1."""
        request_id = "req-decode"

        tracker = _make_tracker(
            request_id=request_id,
            num_tokens=4097,
            num_block_hashes=64,
            num_allocated_blocks=64,
            num_scheduled_tokens=4096,
            num_stored_blocks=64,
            num_lmcache_hit_blocks=0,
        )

        connector = self._make_connector_with_tracker(request_id, tracker)

        cached_reqs = _FakeCachedReqs(
            req_ids=[request_id],
            new_block_ids=[None],
            resumed_req_ids=set(),
        )

        scheduler_output = MagicMock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = cached_reqs
        scheduler_output.num_scheduled_tokens = {request_id: 1}

        connector.build_connector_meta(scheduler_output)

        assert tracker.num_scheduled_tokens == 4097, (
            "Decode step should add exactly 1 token, "
            f"got {tracker.num_scheduled_tokens}."
        )
