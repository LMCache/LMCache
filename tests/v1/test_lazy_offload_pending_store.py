# SPDX-License-Identifier: Apache-2.0
"""Facade-level tests for the lazy offload pending store.

The pure policy behind EVICTION_AWARE mode is covered in depth by
test_lazy_offload_eviction_aware.py; here the eviction-aware tests only verify the
facade's routing and mode guards. The FIFO sections exercise the legacy
count-triggered placeholder policy kept for compatibility.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import lazy_offload_pending_store as pending_store_mod
from lmcache.integration.vllm.lazy_offload_pending_store import (
    AddOutcome,
    LazyOffloadMode,
    LazyOffloadPendingStore,
)
from lmcache.integration.vllm.lazy_offload_policy.fifo import FIFOOffloadPolicy

FIFO_CONFIG = {"lmcache.mp.lazy_offload_policy": "FIFO"}
EVICTION_AWARE_CONFIG = {"lmcache.mp.lazy_offload_policy": "EVICTION_AWARE"}


def _spy_logger(monkeypatch: pytest.MonkeyPatch, method: str) -> list[str]:
    """Capture the module logger's lines emitted through one method.

    The lmcache logger does not propagate to the root logger
    (``propagate=False``), so pytest's ``caplog`` never sees its records;
    spy on the method instead.

    Args:
        monkeypatch: The fixture used to install the spy.
        method: The logger method to capture, e.g. ``"info"``.

    Returns:
        A list that accumulates the formatted messages as they are logged.
    """
    messages: list[str] = []

    def spy(msg: object, *args: object, **kwargs: object) -> None:
        messages.append(str(msg) % args if args else str(msg))

    monkeypatch.setattr(pending_store_mod.logger, method, spy)
    return messages


def _spy_logger_info(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture the module logger's INFO lines.

    Args:
        monkeypatch: The fixture used to install the spy.

    Returns:
        A list that accumulates the formatted messages as they are logged.
    """
    return _spy_logger(monkeypatch, "info")


def _make_meta(
    request_id: str = "req-0", num_blocks: int = 1, end: int = 256
) -> MagicMock:
    """Helper to create a mock LMCacheMPRequestMetadata."""
    meta = MagicMock()
    meta.request_id = request_id
    meta.op.flat_block_ids = list(range(num_blocks))
    meta.op.end = end
    return meta


def _make_block_hashes(block_ids: list[int]) -> dict[int, bytes]:
    """Helper to create mock block hashes."""
    return {bid: f"hash-{bid}".encode() for bid in block_ids}


def _make_gpu_pool(num_blocks: int = 10) -> MagicMock:
    """Mock BlockPool: hashed blocks, all sitting in the free queue.

    The free queue is a linked list terminated by a tail sentinel, the shape
    vLLM maintains and the one the pool view walks -- it reads ranks by
    following links from the head and stopping at the depth it needs, so a
    mock that only answered ``get_all_free_blocks()`` would report an empty
    queue and every drain would look like an idle step.
    """
    gpu_pool = MagicMock()
    gpu_pool.blocks = {
        bid: MagicMock(block_hash=f"hash-{bid}".encode()) for bid in range(num_blocks)
    }
    tail = SimpleNamespace(block_id=-1, next_free_block=None)
    nodes = [
        SimpleNamespace(block_id=bid, next_free_block=None) for bid in range(num_blocks)
    ]
    for node, following in zip(nodes, [*nodes[1:], tail], strict=True):
        node.next_free_block = following
    gpu_pool.free_block_queue.fake_free_list_head = SimpleNamespace(
        block_id=-1, next_free_block=nodes[0] if nodes else tail
    )
    return gpu_pool


def _drain_store(
    store: LazyOffloadPendingStore,
    new_blocks_allocated: int = 0,
    est_next_step_blocks: int = 0,
) -> pending_store_mod.LazyOffloadDrain:
    return store.drain(
        new_blocks_allocated,
        est_next_step_blocks,
        None,
        set(),
        set(),
    )


# ===========================================================================
# Tests for FIFOOffloadPolicy
# ===========================================================================


class TestFIFOOffloadPolicy:
    def test_configures_threshold(self) -> None:
        assert FIFOOffloadPolicy()._threshold == 100
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        assert policy._threshold == 3

    def test_add_aggregates_one_request_epoch(self) -> None:
        policy = FIFOOffloadPolicy()
        policy.add(_make_meta("req", 1), _make_block_hashes([0]), epoch=2)
        policy.add(_make_meta("req", 2), _make_block_hashes([0, 1]), epoch=2)
        assert len(policy._pending_items["req"].metadatas) == 2

    def test_add_rejects_mixed_epochs_for_one_request(self) -> None:
        policy = FIFOOffloadPolicy()
        policy.add(_make_meta("req"), _make_block_hashes([0]), epoch=2)
        with pytest.raises(RuntimeError, match="mixed store epochs 2 and 3"):
            policy.add(_make_meta("req"), _make_block_hashes([1]), epoch=3)

    def test_threshold_counts_controller_eligible_requests(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        for index in range(3):
            policy.add(_make_meta(f"req-{index}"), _make_block_hashes([index]))

        assert policy.pop_items_for_offload(10, {"req-0"}) == []
        selected = policy.pop_items_for_offload(10, {"req-0", "req-2"})
        assert [item.request_id for item in selected] == ["req-0", "req-2"]
        assert policy.has_pending_request("req-1")

    def test_blocked_request_is_not_eligible_or_popped(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        for request_id, block_id in (("blocked", 0), ("ready", 1)):
            policy.add(_make_meta(request_id), _make_block_hashes([block_id]))

        selected = policy.pop_items_for_offload(10, {"blocked", "ready"}, {"blocked"})
        assert [item.request_id for item in selected] == ["ready"]
        assert policy.has_pending_request("blocked")

    def test_count_caps_fifo_selection(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        finished = {f"req-{index}" for index in range(5)}
        for index in range(5):
            policy.add(_make_meta(f"req-{index}"), _make_block_hashes([index]))

        selected = policy.pop_items_for_offload(2, finished)
        assert [item.request_id for item in selected] == ["req-0", "req-1"]
        assert len(policy._pending_items) == 3

    def test_discard_operations_report_chunk_count(self) -> None:
        policy = FIFOOffloadPolicy()
        policy.add(_make_meta("req", 1), _make_block_hashes([0]))
        policy.add(_make_meta("req", 2), _make_block_hashes([0, 1]))
        assert policy.drop_request("req") == 2
        assert policy.drop_request("req") == 0

        policy.add(_make_meta("req"), _make_block_hashes([0]))
        assert policy.discard_for_reuse("req") == 1
        assert not policy.has_pending_request("req")


# ===========================================================================
# Tests for the LazyOffloadPendingStore facade (FIFO mode + shared plumbing)
# ===========================================================================


class TestLazyOffloadPendingStore:
    def _setup_store_with_gpu_pool(
        self, configs: dict | None = None
    ) -> LazyOffloadPendingStore:
        store = LazyOffloadPendingStore({**FIFO_CONFIG, **(configs or {})})
        store.bind_gpu_block_pool(_make_gpu_pool())
        return store

    def test_selects_and_validates_mode(self) -> None:
        assert LazyOffloadPendingStore().mode is LazyOffloadMode.FIFO
        store = LazyOffloadPendingStore(dict(EVICTION_AWARE_CONFIG))
        assert store.mode is LazyOffloadMode.EVICTION_AWARE
        with pytest.raises(ValueError, match="Unknown offload policy"):
            LazyOffloadPendingStore({"lmcache.mp.lazy_offload_policy": "UNKNOWN"})

    def test_add_requires_bound_gpu_pool(self) -> None:
        store = LazyOffloadPendingStore()
        with pytest.raises(ValueError, match="gpu block pool not bound"):
            store.add(_make_meta("req"))

    def test_fifo_add_snapshots_hashes_and_drains_eligible_request(self) -> None:
        store = self._setup_store_with_gpu_pool(
            {"lmcache.mp.lazy_offload_threshold": 1}
        )
        meta = _make_meta("req", num_blocks=2)
        assert store.add(meta) is AddOutcome.BUFFERED
        assert store.has_pending_request("req")

        (item,) = store.drain(0, 0, None, {"req"}, set()).items
        assert item.metadatas == [(meta, {0: b"hash-0", 1: b"hash-1"})]
        assert not store.has_pending_request("req")

    def test_fifo_facade_respects_threshold_select_count_and_order(self) -> None:
        store = self._setup_store_with_gpu_pool(
            {
                "lmcache.mp.lazy_offload_threshold": 3,
                "lmcache.mp.lazy_offload_select_count": 2,
            }
        )
        finished = {f"req-{index}" for index in range(5)}
        for index in range(5):
            store.add(_make_meta(f"req-{index}"))

        first = store.drain(0, 0, None, finished, set()).items
        second = store.drain(0, 0, None, finished, set()).items
        assert [item.request_id for item in first] == ["req-0", "req-1"]
        assert [item.request_id for item in second] == ["req-2", "req-3"]
        assert store.drain(0, 0, None, finished, set()).items == []

    def test_fifo_facade_routes_drop_and_reuse_discard(self) -> None:
        store = self._setup_store_with_gpu_pool()
        store.add(_make_meta("reset"))
        store.add(_make_meta("reused"))
        assert store.drop_request("reset") == 1
        assert store.discard_for_reuse("reused") == 1


# ===========================================================================
# Tests for LazyOffloadPendingStore in EVICTION_AWARE mode
# ===========================================================================


class TestEvictionAwareMode:
    """Facade routing to the eviction-aware queue and FIFO entry-point guards.

    Policy semantics (pressure, prefix closure, deduplication, lifecycle) are
    covered by test_lazy_offload_eviction_aware.py; these tests stop at the facade."""

    def _setup(
        self, configs: dict | None = None
    ) -> tuple[LazyOffloadPendingStore, MagicMock]:
        store = LazyOffloadPendingStore({**EVICTION_AWARE_CONFIG, **(configs or {})})
        gpu_pool = _make_gpu_pool()
        store.bind_gpu_block_pool(gpu_pool)
        return store, gpu_pool

    def test_add_buffers_hashed_op(self) -> None:
        store, _ = self._setup()
        assert store.add(_make_meta("req-0", num_blocks=2)) is AddOutcome.BUFFERED

    def test_add_skips_unhashed_op(self) -> None:
        store, gpu_pool = self._setup()
        gpu_pool.blocks[1].block_hash = None
        assert store.add(_make_meta("req-0", num_blocks=2)) is (
            AddOutcome.SKIPPED_UNHASHED
        )

    def test_unhashed_skip_names_both_of_its_causes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The warning has to name sliding-window attention, not only
        disabled prefix caching. A hash-less block reaches admission in two
        ways, and on a hybrid-attention model prefix caching is on: an
        operator told to check that setting finds it already correct and has
        nothing left to look at."""
        store, gpu_pool = self._setup()
        gpu_pool.blocks[0].block_hash = None
        warnings = _spy_logger(monkeypatch, "warning")
        meta = _make_meta("req-0", num_blocks=1, end=512)
        # _make_meta leaves `start` a MagicMock, which the warning's %d
        # cannot format; the real logger swallows that, the spy here does
        # not.
        meta.op.start = 0
        assert store.add(meta) is AddOutcome.SKIPPED_UNHASHED
        assert len(warnings) == 1
        assert "req-0" in warnings[0]
        assert "[0, 512)" in warnings[0]
        assert "prefix caching is off" in warnings[0]
        assert "sliding-window" in warnings[0]

    def test_collect_due_under_pressure_emits_op(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        meta = _make_meta("req-0", num_blocks=2)
        store.add(meta)
        result = _drain_store(store, 4, 0)
        assert [entry[0] for item in result.items for entry in item.metadatas] == [meta]

    def test_collect_due_without_pressure_holds(self) -> None:
        store, _ = self._setup()
        store.add(_make_meta("req-0", num_blocks=2))
        assert _drain_store(store, 0, 0).items == []

    def test_drain_reports_policy_neutral_emptied_request(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        result = _drain_store(store, 4, 0)
        assert len(result.items) == 1
        assert result.emptied_request_ids == ["req-0"]
        assert not store.has_pending_request("req-0")

    def test_drop_request_discards_buffered_ops(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))
        assert store.drop_request("req-0") == 1
        assert _drain_store(store, 4, 0).items == []

    def test_discard_for_reuse_routes_to_eviction_queue(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))
        assert store.discard_for_reuse("req-0") == 1
        assert _drain_store(store, 4, 0).items == []

    def test_mark_store_failed_drops_buffered_ops(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))
        assert store.mark_store_failed("req-0") == 1
        assert _drain_store(store, 4, 0).items == []

    def test_stats_reports_the_cumulative_counters(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        _drain_store(store, 4, 0)
        stats = store.stats()
        assert stats.admitted == 1
        assert stats.emitted == 1
        assert stats.dropped_evicted == 0

    def test_stats_counts_a_drop_of_evicted_blocks(self) -> None:
        store, gpu_pool = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        gpu_pool.blocks[0].block_hash = b"reallocated"
        assert _drain_store(store, 4, 0).items == []
        assert store.stats().dropped_evicted == 1

    def test_stats_unavailable_in_fifo_mode_or_before_bind(self) -> None:
        with pytest.raises(ValueError, match="EVICTION_AWARE queue unavailable"):
            LazyOffloadPendingStore().stats()
        fifo = LazyOffloadPendingStore(dict(FIFO_CONFIG))
        fifo.bind_gpu_block_pool(_make_gpu_pool())
        with pytest.raises(ValueError, match="EVICTION_AWARE queue unavailable"):
            fifo.stats()

    def test_evicted_drop_is_logged_at_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One aggregate INFO line per drain with the drop count: the drop
        ledger must be visible in production logs (which rarely run at
        DEBUG) without flooding the scheduler path when a burst evicts a
        large queue at once."""
        store, gpu_pool = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        messages = _spy_logger_info(monkeypatch)
        store.add(_make_meta("req-0", num_blocks=1))
        gpu_pool.blocks[0].block_hash = b"reallocated"
        _drain_store(store, 4, 0)
        (line,) = [m for m in messages if "blocks evicted before drain" in m]
        assert "dropped 1 store op(s)" in line
        assert "req-0" in line

    def test_short_prefix_drop_is_logged_at_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A gate-3 drop is cache-quality loss too, so it gets the same
        aggregate INFO line as the eviction path: the counter alone cannot
        say which request lost its prefix."""
        store, _ = self._setup(
            {
                "lmcache.mp.lazy_offload_horizon_steps": 1.0,
                "lmcache.mp.lazy_offload_min_prefix_tokens": 4096,
            }
        )
        messages = _spy_logger_info(monkeypatch)
        store.add(_make_meta("req-0", num_blocks=1, end=256))
        _drain_store(store, 4, 0)
        assert store.stats().rejected_short_prefix == 1
        (line,) = [m for m in messages if "below the break-even length" in m]
        assert "dropped 1 store op(s)" in line
        assert "req-0 (prefix 256)" in line

    def test_drop_line_reports_the_ops_it_could_not_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drain that drops more ops than the line samples must say so.

        The line names at most eight ops, and the count of the rest is the
        only thing between "these are the ops that were lost" and a reader
        (or a per-request-id grep) concluding that nothing else was. The
        truncation itself is untested at every other layer: on hardware the
        aggregate lines are asserted never to truncate, precisely so that
        the greps stay sound."""
        store, _ = self._setup(
            {
                "lmcache.mp.lazy_offload_horizon_steps": 1.0,
                "lmcache.mp.lazy_offload_min_prefix_tokens": 4096,
            }
        )
        messages = _spy_logger_info(monkeypatch)
        for block_id in range(10):
            meta = _make_meta(f"req-{block_id}", end=256)
            meta.op.flat_block_ids = [block_id]
            store.add(meta)
        _drain_store(store, 40, 0)
        assert store.stats().rejected_short_prefix == 10
        (line,) = [m for m in messages if "below the break-even length" in m]
        assert "dropped 10 store op(s)" in line
        assert line.count("(prefix 256)") == 8
        assert "+2 more" in line

    def test_throttled_drain_that_also_loses_ops_warns_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cap that holds ops back while the queue is dying is a sizing
        error, and the only place it is visible before the cache-hit rate
        falls. Warned once per process: the misconfiguration lasts the whole
        run, so repeating it every step would add noise, and the
        throttled_drains counter carries the recurrence."""
        store, gpu_pool = self._setup(
            {
                "lmcache.mp.lazy_offload_horizon_steps": 1.0,
                "lmcache.mp.lazy_offload_max_drain_per_step": 1,
            }
        )
        warnings = _spy_logger(monkeypatch, "warning")
        # The two live ops must form one contiguous range, or the drain
        # cuts at the gap instead of at the cap and the test would pass on
        # the wrong mechanism.
        for block_id, start, end in ((0, 0, 256), (1, 256, 512)):
            meta = _make_meta("req-live", end=end)
            meta.op.start = start
            meta.op.flat_block_ids = [block_id]
            store.add(meta)
        doomed = _make_meta("req-doomed", end=256)
        doomed.op.start = 0
        doomed.op.flat_block_ids = [2]
        store.add(doomed)
        gpu_pool.blocks[2].block_hash = b"reallocated"
        _drain_store(store, 40, 0)
        assert store.stats().throttled_drains == 1
        assert store.stats().dropped_evicted == 1
        (line,) = [m for m in warnings if "max_drain_per_step" in m]
        assert "held back 1 due store op(s)" in line
        assert "1 op(s) were lost to eviction" in line

        # Same symptoms again, one line total.
        third = _make_meta("req-live", end=768)
        third.op.start = 512
        third.op.flat_block_ids = [4]
        store.add(third)
        again = _make_meta("req-doomed2", end=256)
        again.op.start = 0
        again.op.flat_block_ids = [3]
        store.add(again)
        gpu_pool.blocks[3].block_hash = b"reallocated"
        _drain_store(store, 40, 0)
        assert len([m for m in warnings if "max_drain_per_step" in m]) == 1

    def test_healthy_drain_does_not_warn_about_the_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Losing ops without the cap binding is ordinary pressure, not a
        misconfiguration: warning on it would train operators to raise a
        knob that had nothing to do with the loss."""
        store, gpu_pool = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        warnings = _spy_logger(monkeypatch, "warning")
        store.add(_make_meta("req-0", num_blocks=1))
        gpu_pool.blocks[0].block_hash = b"reallocated"
        _drain_store(store, 4, 0)
        assert store.stats().dropped_evicted == 1
        assert store.stats().throttled_drains == 0
        assert [m for m in warnings if "max_drain_per_step" in m] == []

    def test_skip_of_a_broken_request_logs_at_debug_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken request keeps producing chunks and every one of them is
        rejected; at INFO those would bury the one line that reported the
        cause, so the tail logs at DEBUG."""
        store, gpu_pool = self._setup()
        gpu_pool.blocks[0].block_hash = None
        assert store.add(_make_meta("req-0", num_blocks=1)) is (
            AddOutcome.SKIPPED_UNHASHED
        )
        info = _spy_logger_info(monkeypatch)
        debug = _spy_logger(monkeypatch, "debug")
        assert store.add(_make_meta("req-0", num_blocks=2)) is (
            AddOutcome.SKIPPED_PREFIX_BROKEN
        )
        assert [m for m in info if "req-0" in m] == []
        assert [m for m in debug if "prefix chain is already broken" in m] != []

    def test_drain_logs_the_ledger_periodically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """collect_due logs a ledger line when the counters changed, at
        most once per throttle interval, and logs AGAIN once the interval
        lapses: the log must converge to the true ledger even when a
        force-killed engine never reaches the shutdown hook."""
        clock = [1000.0]
        monkeypatch.setattr(pending_store_mod.time, "monotonic", lambda: clock[0])
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        messages = _spy_logger_info(monkeypatch)
        store.add(_make_meta("req-0", num_blocks=1))
        _drain_store(store, 4, 0)  # changed since start -> logs
        _drain_store(store)  # unchanged -> silent
        store.add(_make_meta("req-1", num_blocks=2))
        _drain_store(store)  # changed, but inside the throttle -> silent
        ledgers = [m for m in messages if m.startswith("Lazy offload counters:")]
        assert len(ledgers) == 1
        assert "admitted=1" in ledgers[0]
        assert "emitted=1" in ledgers[0]
        # The depth belongs on the periodic line, not only on the shutdown
        # one: a force-killed engine never reaches log_final_stats, so this
        # is the line an operator (and the GPU harness) actually reads, and
        # without pending it does not close as an equation.
        assert "pending=0" in ledgers[0]
        clock[0] += 6.0
        _drain_store(store)  # throttle lapsed, change pending -> logs again
        ledgers = [m for m in messages if m.startswith("Lazy offload counters:")]
        assert len(ledgers) == 2
        assert "admitted=2" in ledgers[1]
        assert "pending=0" in ledgers[1]

    def test_log_final_stats_emits_the_counter_ledger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, gpu_pool = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        gpu_pool.blocks[0].block_hash = b"reallocated"
        _drain_store(store, 4, 0)
        messages = _spy_logger_info(monkeypatch)
        store.log_final_stats()
        (line,) = [m for m in messages if "final counters" in m]
        assert "admitted=1" in line
        assert "emitted=0" in line
        assert "dropped_evicted=1" in line

    def test_ledger_reports_the_pending_depth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ledger carries the queue depth at the same instant as the
        counters, so the line closes as an equation: an op that left the
        queue without incrementing any outcome counter would show up as
        admitted > pending + outcomes."""
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        store.add(_make_meta("req-1", num_blocks=1))
        messages = _spy_logger_info(monkeypatch)
        store.log_final_stats()
        (held,) = [m for m in messages if "final counters" in m]
        assert "admitted=2" in held
        assert "emitted=0" in held
        assert "pending=2" in held

        assert len(_drain_store(store, 4, 0).items) == 2
        messages.clear()
        store.log_final_stats()
        (drained,) = [m for m in messages if "final counters" in m]
        assert "admitted=2" in drained
        assert "emitted=2" in drained
        assert "pending=0" in drained

    def test_log_final_stats_is_silent_when_nothing_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unbound or FIFO stores have no counters; shutdown must neither
        raise nor log a bogus ledger."""
        messages = _spy_logger_info(monkeypatch)
        LazyOffloadPendingStore().log_final_stats()
        fifo = LazyOffloadPendingStore(dict(FIFO_CONFIG))
        fifo.bind_gpu_block_pool(_make_gpu_pool())
        fifo.log_final_stats()
        # The constructor's own banner may log; the ledger line must not.
        assert [m for m in messages if "final counters" in m] == []

    def test_rebind_same_pool_is_idempotent(self) -> None:
        store, gpu_pool = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))

        store.bind_gpu_block_pool(gpu_pool)

        # Buffered state survived the redundant bind.
        assert len(_drain_store(store, 4, 0).items) == 1

    def test_rebind_different_pool_raises(self) -> None:
        store, _ = self._setup()
        with pytest.raises(ValueError, match="already bound"):
            store.bind_gpu_block_pool(_make_gpu_pool())
