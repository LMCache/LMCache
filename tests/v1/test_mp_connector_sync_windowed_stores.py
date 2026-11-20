# SPDX-License-Identifier: Apache-2.0
"""wait_for_save waits for stores that read sliding-window / recurrent blocks."""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    STORE_PIN_HINT,
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)


def _connector(
    has_windowed_groups: bool,
    sync: bool,
    direction: str = "STORE",
    request_configs: dict | None = None,
) -> tuple[LMCacheMPConnector, MagicMock]:
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.dispatcher = None
    connector._has_windowed_groups = has_windowed_groups
    connector._sync_windowed_stores = sync
    connector._pinned_store_futures = []
    transfer_ctx = MagicMock()
    connector.worker_adapter = MagicMock(transfer_ctx=transfer_ctx, store_futures={})
    metadata = LMCacheMPConnectorMetadata()
    metadata.add_request_metadata(
        cast(
            Any,
            SimpleNamespace(
                request_id="req-0",
                direction=direction,
                op=MagicMock(),
                cache_salt="",
                request_configs=request_configs,
            ),
        )
    )
    connector._get_connector_metadata = lambda: metadata  # type: ignore[method-assign]
    return connector, transfer_ctx


@pytest.mark.parametrize(
    ("has_windowed_groups", "sync", "waits"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_store_waits_only_for_windowed_groups_when_enabled(
    has_windowed_groups: bool, sync: bool, waits: bool
) -> None:
    connector, transfer_ctx = _connector(has_windowed_groups, sync)

    connector.wait_for_save()

    connector.worker_adapter.submit_store_request.assert_called_once()
    assert transfer_ctx.flush_inflight_stores.called is waits


def test_step_without_stores_does_not_wait() -> None:
    connector, transfer_ctx = _connector(True, True, direction="RETRIEVE")

    connector.wait_for_save()

    connector.worker_adapter.submit_store_request.assert_not_called()
    transfer_ctx.flush_inflight_stores.assert_not_called()


def test_store_without_windowed_objects_does_not_wait() -> None:
    connector, transfer_ctx = _connector(True, True)
    connector._store_reads_windowed_blocks = lambda meta: False  # type: ignore[method-assign]

    connector.wait_for_save()

    transfer_ctx.flush_inflight_stores.assert_not_called()


# ------------------------------------------------------------------ #
#  lmcache.mp.pin_windowed_stores                                      #
# ------------------------------------------------------------------ #


def _scheduler() -> tuple[LMCacheMPConnector, MagicMock]:
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.lazy_offload = False
    connector._pin_windowed_stores = True
    pool = MagicMock()
    pool.blocks = {i: f"block-{i}" for i in range(100)}
    connector._gpu_block_pool = pool
    connector.scheduler_adapter = MagicMock(lmcache_tokens_per_chunk=8)
    # Group 0: full attention, 8 tokens/block. Group 1: sliding window of 3
    # tokens over 2-token blocks (2 window blocks). Group 2: Mamba, 2-token
    # blocks, one state block per chunk.
    connector._group_tokens_per_block = [8, 2, 2]
    connector._windowed_group_tokens = {1: 3, 2: 2}
    connector._store_pins = {}
    connector._store_pin_seq = 0
    connector._sched_step = 0
    connector._store_pin_steps = {}
    connector._pin_release_steps = 2
    connector._pinned_boundary_state_blocks = {}
    return connector, pool


def _store(
    start: int, end: int, block_ids: list[list[int]]
) -> LMCacheMPRequestMetadata:
    return LMCacheMPRequestMetadata(
        request_id="req-0",
        direction="STORE",
        op=LoadStoreOp(token_bytes=b"", block_ids=block_ids, start=start, end=end),
        request_configs={"lmcache.tag.x": "y"},
    )


def test_scheduler_pins_window_blocks_for_k_steps() -> None:
    connector, pool = _scheduler()
    # Two full chunks; group 2's first chunk ends on a null state block.
    r_meta = _store(
        0, 16, [[1, 2], [10, 11, 12, 13, 14, 15, 16, 17], [0, 0, 0, 0, 0, 0, 0, 27]]
    )

    connector._pin_store_blocks(r_meta)

    pinned = [12, 13, 16, 17, 27]
    pool.touch.assert_called_once_with([f"block-{i}" for i in pinned])
    assert r_meta.request_configs is not None
    key = r_meta.request_configs[STORE_PIN_HINT]
    assert r_meta.request_configs["lmcache.tag.x"] == "y"
    assert connector._store_pins == {key: pinned}

    connector._release_aged_store_pins()
    pool.free_blocks.assert_not_called()
    connector._release_aged_store_pins()
    pool.free_blocks.assert_called_once_with([f"block-{i}" for i in pinned])
    assert connector._store_pins == {}


def test_scheduler_pins_the_unfull_window_with_its_lead_block() -> None:
    connector, pool = _scheduler()
    # Full chunk [0, 8) plus unfull [8, 14): group 1 blocks 4..6 of the tail.
    r_meta = _store(
        0, 14, [[1, 2], [10, 11, 12, 13, 20, 21, 22], [0, 0, 0, 7, 0, 0, 9]]
    )

    connector._pin_store_blocks(r_meta)

    (blocks,), _ = pool.touch.call_args
    assert blocks == [f"block-{i}" for i in [7, 9, 12, 13, 20, 21, 22]]


def test_scheduler_does_not_pin_when_disabled() -> None:
    connector, pool = _scheduler()
    connector._pin_windowed_stores = False
    r_meta = _store(0, 8, [[1], [10, 11, 12, 13], [0, 0, 0, 7]])

    connector._pin_store_blocks(r_meta)

    pool.touch.assert_not_called()
    assert STORE_PIN_HINT not in (r_meta.request_configs or {})


def test_step_release_unpins_after_k_steps() -> None:
    connector, pool = _scheduler()
    connector._release_aged_store_pins()  # step 1
    r_meta = _store(0, 8, [[1], [10, 11, 12, 13], [0, 0, 0, 7]])
    connector._pin_store_blocks(r_meta)

    connector._release_aged_store_pins()  # step 2
    pool.free_blocks.assert_not_called()
    connector._release_aged_store_pins()  # step 3
    pool.free_blocks.assert_called_once_with([f"block-{i}" for i in [7, 12, 13]])
    assert connector._store_pins == {}


def test_worker_waits_for_pinned_stores_of_earlier_steps() -> None:
    connector, transfer_ctx = _connector(
        True, True, request_configs={STORE_PIN_HINT: "k"}
    )
    future = MagicMock()
    connector.worker_adapter.submit_store_request.side_effect = (
        lambda request_id, *a, **k: connector.worker_adapter.store_futures.update(
            {request_id: future}
        )
    )

    connector.wait_for_save()  # step 1: submits, does not wait for itself
    future.wait.assert_not_called()
    transfer_ctx.flush_inflight_stores.assert_not_called()
    connector._get_connector_metadata = (  # type: ignore[method-assign]
        lambda: LMCacheMPConnectorMetadata()
    )
    connector.wait_for_save()  # step 2: waits for step 1's pinned store
    future.wait.assert_called_once()
