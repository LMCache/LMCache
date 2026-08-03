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
import threading
import time

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, TrimPolicy
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod


def _make_engine():
    """A BlendV3Module mock with only the attributes these paths touch."""
    eng = MagicMock(spec=v3_mod.BlendV3Module)
    # ``_event_bus`` is an instance attr (set in __init__), so spec= omits it.
    eng._event_bus = MagicMock()
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
    which stayed flat under V3 until this site existed."""

    def test_counts_only_registered_chunks(self):
        eng = _make_engine()
        emit = _bind(eng, "_emit_fingerprints_registered")

        # start_chunk_idx=1: chunk 0 of a position-0 store is owned by the
        # standard prefix path and is not registered.
        emit("req-1", [b"h0", b"h1", b"h2"], 1, list(range(768)))

        events = _published(eng)
        assert len(events) == 1
        event_type, metadata, sid = events[0]
        assert event_type is EventType.CB_FINGERPRINTS_REGISTERED
        assert metadata == {"num_chunks": 2, "num_tokens": 768}
        assert sid == "req-1", "must carry the enqueuing store's request id"

    def test_never_reports_negative_chunks(self):
        eng = _make_engine()
        _bind(eng, "_emit_fingerprints_registered")("req-2", [b"h0"], 3, [1, 2])

        assert _published(eng)[0][1]["num_chunks"] == 0

    def test_sync_drain_publishes_after_registration(self):
        eng = _make_engine()
        eng._fingerprint_queue = Queue()
        eng._token_range_matcher = MagicMock()
        eng._emit_fingerprints_registered = _bind(eng, "_emit_fingerprints_registered")
        eng._fingerprint_queue.put(([1, 2], [b"h0", b"h1"], 0, 0, "req-sync"))

        _bind(eng, "_drain_fingerprints_sync")()

        assert eng._token_range_matcher.on_new_token_hashes.call_count == 1
        events = _published(eng)
        assert [e[0] for e in events] == [EventType.CB_FINGERPRINTS_REGISTERED]
        assert events[0][2] == "req-sync"

    def test_no_event_when_registration_fails(self):
        """A failed registration means the chunks are not matchable, so the
        counter must not move."""
        eng = _make_engine()
        eng._fingerprint_queue = Queue()
        eng._token_range_matcher = MagicMock()
        eng._token_range_matcher.on_new_token_hashes.side_effect = RuntimeError("boom")
        eng._emit_fingerprints_registered = _bind(eng, "_emit_fingerprints_registered")
        eng._fingerprint_queue.put(([1], [b"h0"], 0, 0, "req-fail"))

        _bind(eng, "_drain_fingerprints_sync")()

        assert _published(eng) == []


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
