# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the CB V3 observability emission sites.

These cover the event *contracts* the metrics/tracing subscribers depend on —
what each site publishes and with which metadata — without touching CUDA or the
storage controller.  See
``docs/design/v1/mp_observability/blend_v3_observability.md``.
"""

# Standard
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock
import ast
import inspect
import threading
import time

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, TrimPolicy
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.custom_types import CBMatchResult
from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

_CHUNK = 256


def _make_engine():
    """A BlendV3Module mock with only the attributes these paths touch."""
    eng = MagicMock(spec=v3_mod.BlendV3Module)
    # Instance attrs (set in __init__) are omitted by spec=; add the ones used.
    eng._event_bus = MagicMock()
    eng._token_range_matcher = MagicMock()
    eng._token_range_matcher.chunk_size = _CHUNK
    return eng


def _bind(eng, name):
    """Bind a real BlendV3Module method to the mock engine."""
    return getattr(v3_mod.BlendV3Module, name).__get__(eng)


def _published(eng):
    """Return the published events as a list of (EventType, metadata, sid)."""
    return [
        (
            call.args[0].event_type,
            call.args[0].metadata,
            call.args[0].session_id,
        )
        for call in eng._event_bus.publish.call_args_list
    ]


class TestFingerprintsRegisteredEvent:
    """The registration event feeds ``lmcache_blend.fingerprints_registered``,
    which stayed flat under V3 until this site existed. It must count only the
    chunks the matcher actually indexed."""

    def test_reports_indexed_chunks_and_their_tokens(self):
        eng = _make_engine()

        _bind(eng, "_emit_fingerprints_registered")("req-1", 2)

        events = _published(eng)
        assert len(events) == 1
        event_type, metadata, sid = events[0]
        assert event_type is EventType.CB_FINGERPRINTS_REGISTERED
        assert metadata == {"num_chunks": 2, "num_tokens": 2 * _CHUNK}
        assert sid == "req-1", "must carry the enqueuing store's request id"

    def test_no_event_when_nothing_was_indexed(self):
        """A re-store of known content indexes nothing: no event, so neither
        the counter nor the trace gets a zero-valued registration."""
        eng = _make_engine()

        _bind(eng, "_emit_fingerprints_registered")("req-2", 0)

        assert _published(eng) == []

    def test_sync_drain_reports_the_matcher_count_not_the_job_size(self):
        """The matcher dedups already-registered hashes; the event must follow
        its return value, not ``len(chunk_hashes) - start_chunk_idx``."""
        eng = _make_engine()
        eng._fingerprint_queue = Queue()
        eng._token_range_matcher.on_new_token_hashes.return_value = 1
        eng._emit_fingerprints_registered = _bind(eng, "_emit_fingerprints_registered")
        # Job size says 3 chunks; the matcher only indexed 1 of them.
        eng._fingerprint_queue.put(
            (list(range(3 * _CHUNK)), [b"h0", b"h1", b"h2"], 0, 0, "req-sync")
        )

        _bind(eng, "_drain_fingerprints_sync")()

        assert eng._token_range_matcher.on_new_token_hashes.call_count == 1
        events = _published(eng)
        assert [e[0] for e in events] == [EventType.CB_FINGERPRINTS_REGISTERED]
        assert events[0][1] == {"num_chunks": 1, "num_tokens": _CHUNK}
        assert events[0][2] == "req-sync"

    def test_async_drain_skips_event_for_deduplicated_job(self):
        eng = _make_engine()
        eng._fingerprint_queue = Queue()
        eng._fingerprint_stop = threading.Event()
        eng._pending_fp_lock = threading.Lock()
        eng._pending_fp_hashes = set()
        eng._token_range_matcher.on_new_token_hashes.return_value = 0
        eng._emit_fingerprints_registered = _bind(eng, "_emit_fingerprints_registered")
        eng._fingerprint_queue.put(([1], [b"h0"], 0, 0, "req-dup"))

        worker = threading.Thread(
            target=_bind(eng, "_drain_fingerprint_queue"), daemon=True
        )
        worker.start()
        deadline = time.monotonic() + 2.0
        while (
            eng._token_range_matcher.on_new_token_hashes.call_count < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        eng._fingerprint_stop.set()
        worker.join(timeout=1.0)

        assert eng._token_range_matcher.on_new_token_hashes.call_count == 1
        assert _published(eng) == []

    def test_no_event_when_registration_fails(self):
        """A failed registration means the chunks are not matchable, so the
        counter must not move."""
        eng = _make_engine()
        eng._fingerprint_queue = Queue()
        eng._token_range_matcher.on_new_token_hashes.side_effect = RuntimeError("boom")
        eng._emit_fingerprints_registered = _bind(eng, "_emit_fingerprints_registered")
        eng._fingerprint_queue.put(([1], [b"h0"], 0, 0, "req-fail"))

        _bind(eng, "_drain_fingerprints_sync")()

        assert _published(eng) == []


class TestMatcherRegistrationCount:
    """``on_new_token_hashes`` returns how many chunks it newly indexed -- the
    number the registration event reports."""

    def test_first_store_counts_indexed_chunks_and_restore_counts_zero(self):
        matcher = v3_mod.BlendTokenRangeMatcherV3(chunk_size=4)
        tokens = list(range(16))
        hashes = [b"h0", b"h1", b"h2", b"h3"]

        # start_chunk_idx=1: chunk 0 belongs to the prefix leg.
        assert matcher.on_new_token_hashes(tokens, hashes, 1, 0) == 3
        # Same content again: every hash is already indexed.
        assert matcher.on_new_token_hashes(tokens, hashes, 1, 0) == 0
        # A partially new sequence counts only its new chunks.
        assert (
            matcher.on_new_token_hashes(tokens, [b"h0", b"h1", b"n2", b"n3"], 1, 0) == 2
        )

    def test_no_full_chunk_counts_zero(self):
        matcher = v3_mod.BlendTokenRangeMatcherV3(chunk_size=4)

        assert matcher.on_new_token_hashes([1, 2, 3], [], 0, 0) == 0
        assert matcher.on_new_token_hashes(list(range(4)), [b"h0"], 1, 0) == 0


class TestPrefixLegNoGpuContext:
    """``no_gpu_context`` drives ``lookup_no_gpu_context_errors``; V3 hardcoded
    it to False, so a server with no registered CB KV cache looked healthy."""

    def _key(self):
        return SimpleNamespace(
            request_id="req-ctx",
            model_name="m",
            world_size=2,
            token_ids=[1, 2, 3],
        )

    def test_missing_layout_reports_no_gpu_context(self):
        eng = _make_engine()
        eng._resolve_cb_read_layouts = MagicMock(return_value=None)

        handle, world_size, gids, windows, n_chunks, no_gpu_context = _bind(
            eng, "_submit_prefix_leg"
        )(self._key(), 2, TrimPolicy.PREFIX)

        assert handle is None
        assert world_size == 2
        assert (gids, windows, n_chunks) == ((), (), 0)
        assert no_gpu_context is True
        # The prefix span still opens, so the poll's END has a partner.
        assert [e[0] for e in _published(eng)] == [EventType.CB_PREFIX_LOOKUP_START]

    def test_no_full_chunk_is_not_a_no_gpu_context_error(self):
        """A short prompt yields no chunk to look up — that is not the same as a
        misconfigured server, and must not trip the error counter."""
        eng = _make_engine()
        # Legacy fused layout: one object group, full attention.
        eng._resolve_cb_read_layouts = MagicMock(
            return_value=(
                v3_mod._classify_cb_read_groups(1, ()),
                {0: MagicMock()},
                AttnWindowDesc(num_chunks_in_sw=[-1], world_size=2),
            )
        )
        eng._ctx = MagicMock()
        eng._ctx.token_hasher.compute_chunk_hashes.return_value = []

        handle, _, _, _, _, no_gpu_context = _bind(eng, "_submit_prefix_leg")(
            self._key(), 2, TrimPolicy.PREFIX
        )

        assert handle is None
        assert no_gpu_context is False


class TestPollPrefixLegEvent:
    def test_prefix_end_reports_zero_coverage_without_handle(self):
        eng = _make_engine()
        eng._ctx = MagicMock()
        job = SimpleNamespace(
            prefix_handle=None, prefix_world_size=1, prefix_lock_gids=()
        )

        leading, retained = _bind(eng, "_poll_prefix_leg")(job, "req-p", False)

        assert (leading, retained) == (0, None)
        event_type, metadata, sid = _published(eng)[0]
        assert event_type is EventType.CB_PREFIX_LOOKUP_END
        assert metadata == {"prefix_chunks": 0}
        assert sid == "req-p"


class TestCoordinatorMatchTimeoutEvent:
    def test_deadline_publishes_timed_out_end(self):
        eng = _make_engine()
        coordinator = MagicMock()
        coordinator.poll_match.return_value = v3_mod.PENDING
        eng._coordinator = coordinator
        job = SimpleNamespace(
            coord_submitted=True,
            coord_deadline=time.monotonic() - 1.0,  # already past
        )

        assert _bind(eng, "_poll_coordinator_match")(job, "req-c") == []

        event_type, metadata, _ = _published(eng)[0]
        assert event_type is EventType.CB_COORDINATOR_MATCH_END
        assert metadata == {"matches": 0, "timed_out": True}
        coordinator.take_match.assert_called_once_with("req-c")

    def test_pending_within_deadline_defers_without_event(self):
        eng = _make_engine()
        coordinator = MagicMock()
        coordinator.poll_match.return_value = v3_mod.PENDING
        eng._coordinator = coordinator
        job = SimpleNamespace(
            coord_submitted=True,
            coord_deadline=time.monotonic() + 30.0,
        )

        assert _bind(eng, "_poll_coordinator_match")(job, "req-c2") is None
        assert _published(eng) == []


class TestFingerprintJobTuple:
    """The queued job carries the store's request id, so the registration event
    is attributed to the store and not to whichever request drains the queue."""

    def test_store_enqueues_request_id(self):
        eng = _make_engine()
        eng._transfer_module = MagicMock()
        eng._transfer_module.store.return_value = (b"handle", True)
        eng._transfer_module.get_and_touch_context_entry.return_value = None
        eng._pending_fp_lock = threading.Lock()
        eng._pending_fp_hashes = set()
        eng._coordinator = None
        eng._ctx = MagicMock()
        eng._ctx.session_manager.get_or_create.return_value.get_hashes.return_value = [
            123,
            456,
        ]

        eng._fingerprint_queue = Queue()
        key = SimpleNamespace(
            request_id="req-store",
            worker_id=0,
            token_ids=list(range(512)),
            start=0,
            end=512,
        )

        assert _bind(eng, "store")(key, 0, [[0, 1]], b"evt") == (b"handle", True)

        job = eng._fingerprint_queue.get_nowait()
        assert len(job) == 5, "job tuple must carry the request id"
        assert job[4] == "req-store"
        assert job[2] == 1, "chunk 0 of a position-0 store is skipped"


class TestRetrieveEventsCarryWorkerId:
    """At TP>1 each rank publishes its own retrieve/scatter pair under the
    shared request_id, and the metrics subscriber pairs START/END on
    ``(session_id, worker_id)``. Every retrieve/scatter publish site must
    therefore stamp ``worker_id`` -- one site without it would pair as
    ``(phase, sid, None)`` and never match its partner. The GPU retrieve path
    cannot run in a unit test, so this checks the emission sites in the source.
    """

    _EVENTS = {
        "CB_RETRIEVE_START",
        "CB_RETRIEVE_END",
        "CB_SCATTER_START",
        "CB_SCATTER_END",
    }

    def test_every_retrieve_and_scatter_publish_stamps_worker_id(self):
        tree = ast.parse(inspect.getsource(v3_mod))
        sites: list[tuple[str, int, bool]] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Event"
            ):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            et = kw.get("event_type")
            if not (isinstance(et, ast.Attribute) and et.attr in self._EVENTS):
                continue
            md = kw.get("metadata")
            has_worker = isinstance(md, ast.Dict) and any(
                isinstance(k, ast.Constant) and k.value == "worker_id" for k in md.keys
            )
            sites.append((et.attr, node.lineno, has_worker))

        assert {name for name, _, _ in sites} == self._EVENTS, sites
        missing = [(n, ln) for n, ln, ok in sites if not ok]
        assert missing == [], f"publish sites without worker_id: {missing}"
        # One START and three END sites for retrieve, one START + two END for
        # scatter: keep this in step with blend_v3.cb_retrieve_pre_computed.
        assert len(sites) == 7, sites


class TestLookupOnlyRequestEnds:
    """``cb.request`` closes on ``CB_REQUEST_END``, which V3 otherwise publishes
    only from the retrieve path. A request the connector will never retrieve for
    (miss, prefix-only) must be ended by the lookup itself, or its root span
    leaks until shutdown (seen e2e: 5 of 6 roots never exported)."""

    def _engine_with_finished_job(self, rid, job):
        eng = _make_engine()
        eng._ctx = MagicMock()
        eng._ctx.chunk_size = _CHUNK
        eng._cb_jobs_lock = threading.Lock()
        eng._cb_jobs = {rid: job}
        eng._coordinator = None
        return eng

    def _key(self, rid, n_chunks):
        return SimpleNamespace(
            request_id=rid,
            model_name="m",
            world_size=2,
            token_ids=list(range(n_chunks * _CHUNK)),
        )

    def test_prefix_only_lookup_ends_the_request(self):
        rid = "req-prefix-only"
        # Both legs done: prefix landed 4 chunks, sparse leg had nothing to fetch.
        job = v3_mod._CBUnifiedJob(
            matches=[],
            num_tokens=4 * _CHUNK,
            prefix_chunks=4,
            sparse_started=True,
            non_prefix=[],
        )
        eng = self._engine_with_finished_job(rid, job)

        result = _bind(eng, "cb_unified_lookup")(self._key(rid, 4), 2)

        assert result is not None
        assert result.prefix_coverage_tokens == 4 * _CHUNK
        assert result.non_prefix_segments == []
        events = _published(eng)
        assert [e[0] for e in events] == [
            EventType.CB_LOOKUP_END,
            EventType.CB_REQUEST_END,
        ]
        assert all(sid == rid for _, _, sid in events)
        assert rid not in eng._cb_jobs

    def test_miss_ends_the_request(self):
        rid = "req-miss"
        job = v3_mod._CBUnifiedJob(
            matches=[],
            num_tokens=2 * _CHUNK,
            prefix_chunks=0,
            sparse_started=True,
            non_prefix=[],
        )
        eng = self._engine_with_finished_job(rid, job)

        _bind(eng, "cb_unified_lookup")(self._key(rid, 2), 2)

        assert [e[0] for e in _published(eng)] == [
            EventType.CB_LOOKUP_END,
            EventType.CB_REQUEST_END,
        ]

    def test_lookup_with_retrievable_matches_leaves_request_open(self):
        """A retrieve will follow, and it owns CB_REQUEST_END (per rank at TP>1)."""
        rid = "req-hit"
        match = CBMatchResult(
            old_st=0, old_ed=_CHUNK, cur_st=_CHUNK, cur_ed=2 * _CHUNK, hash=b"h"
        )
        job = v3_mod._CBUnifiedJob(
            matches=[match],
            num_tokens=3 * _CHUNK,
            prefix_chunks=1,
            sparse_started=True,
            non_prefix=[match],
            handle=MagicMock(),
            found_uidx={0, 1},
        )
        eng = self._engine_with_finished_job(rid, job)
        eng._sparse_classify.return_value = [match]
        eng._non_overlapping_after_prefix.return_value = [match]

        result = _bind(eng, "cb_unified_lookup")(self._key(rid, 3), 2)

        assert result.non_prefix_segments == [match]
        assert [e[0] for e in _published(eng)] == [EventType.CB_LOOKUP_END]
