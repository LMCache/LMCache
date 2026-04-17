# SPDX-License-Identifier: Apache-2.0

"""High-level replay driver for storage-level traces.

Usage::

    driver = StorageReplayDriver(sm_config, trace_path)
    result = driver.run(pace=ReplayPace.ASAP)
    driver.close()

The driver:

1. Opens the trace file via :class:`TraceReader`.
2. Constructs a fresh :class:`StorageManager` from the supplied
   :class:`StorageManagerConfig`.  Notably, the replay-side config is
   chosen by the *caller*, not copied from the trace's header.  This
   lets the same trace exercise different L1/L2 configurations —
   e.g., record with a Redis L2 adapter and replay with a
   local-filesystem adapter.
3. Iterates records, decodes their argument dicts via the trace
   codec registry, and dispatches each to a live StorageManager call
   through a :class:`CallDispatcher`.
4. Records per-qualname timings into a :class:`ReplayStatsCollector`.

Replay is deliberately single-threaded: the recorder captures calls
in the order the EventBus drained them, which is already a
linearization of the concurrent production calls.  Replaying
in that same order preserves the observed interleaving without
needing to reconstruct thread identities.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Callable
import hashlib
import json
import time

# First Party
from lmcache.logging import init_logger
from lmcache.tools.trace_replay.dispatch import (
    CallDispatcher,
    ReplayContext,
    build_default_dispatcher,
)
from lmcache.tools.trace_replay.stats import ReplayStatsCollector
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.trace import codecs
from lmcache.v1.mp_observability.trace.reader import TraceReader
from lmcache.v1.mp_observability.trace.recorder import safe_storage_config_dict

# Import to trigger lazy registration of the PrefetchHandle codec.
# Done at import time on the replay side so
# :func:`codecs.decode_args` handles PrefetchHandle values from older
# traces that included them.  PR1's storage_manager.py registers this
# codec on its own import path too; the call is idempotent.
codecs.register_prefetch_handle_codec()

logger = init_logger(__name__)


class ReplayPace(Enum):
    """Pacing strategy for the replay loop."""

    ASAP = "asap"
    """Dispatch each record as fast as possible, ignoring recorded
    inter-call intervals.  Useful for storage-stress and latency
    characterization."""

    REALTIME = "realtime"
    """Honor recorded inter-call intervals using ``time.sleep``.
    Reproduces the original pressure on eviction/prefetch queues.
    Never speeds a trace up — only slows down to match."""


@dataclass
class ReplayResult:
    """Summary returned from :meth:`StorageReplayDriver.run`.

    Attributes:
        records_replayed: Successful dispatches.
        records_skipped: Records with no registered handler (likely
            from a newer trace level).
        records_failed: Records whose handler raised.
        stats: The per-qualname timing collector.  Exposed so the
            caller can export CSV/JSON or inspect individual
            percentiles.
        header_level: ``level`` field read from the trace header.
        header_digest: ``sm_config_digest`` from the trace header.
        replay_config_digest: SHA-256 of the replay-side
            StorageManagerConfig, for mismatch comparisons.  Empty
            string if the driver could not compute it.
    """

    records_replayed: int
    records_skipped: int
    records_failed: int
    stats: ReplayStatsCollector
    header_level: str
    header_digest: str
    replay_config_digest: str


class StorageReplayDriver:
    """Replays a storage-level trace against a live StorageManager.

    The driver owns the StorageManager for its lifetime; call
    :meth:`close` (or use as a context manager) to shut it down.
    """

    def __init__(
        self,
        sm_config: StorageManagerConfig,
        trace_path: str,
        dispatcher: CallDispatcher | None = None,
    ) -> None:
        """Construct a driver.

        Args:
            sm_config: Replay-side StorageManager configuration.
                Determines L1 size, eviction policy, and L2 adapters
                used during replay.  Typically different from the
                recording-side config (the original deployment may
                have used adapters unavailable on the replay host).
            trace_path: Path to a ``.lct`` trace file written by
                :class:`StorageTraceRecorder`.
            dispatcher: Custom dispatcher.  When omitted, a fresh
                :func:`build_default_dispatcher` is used; passing one
                explicitly is useful for tests that register extra
                handlers.
        """
        self._sm_config = sm_config
        self._trace_path = trace_path
        self._dispatcher = dispatcher or build_default_dispatcher()
        self._reader = TraceReader(trace_path)
        self._sm = StorageManager(sm_config)
        self._closed = False

    def __enter__(self) -> StorageReplayDriver:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------

    @property
    def trace_path(self) -> str:
        """Path of the trace file being replayed."""
        return self._trace_path

    @property
    def storage_manager(self) -> StorageManager:
        """The live StorageManager driving replay.

        Exposed primarily for tests that need to introspect residency
        or eviction state after ``run()`` returns.
        """
        return self._sm

    def close(self) -> None:
        """Close the StorageManager and reader.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._sm.close()
        finally:
            self._reader.close()

    # ------------------------------------------------------------------

    def run(
        self,
        pace: ReplayPace = ReplayPace.ASAP,
        on_record: RecordCallback | None = None,
    ) -> ReplayResult:
        """Replay every record in the trace.

        Args:
            pace: Pacing strategy; see :class:`ReplayPace`.
            on_record: Optional per-record callback invoked after
                dispatch with ``(qualname, latency_s, failed)``.
                Used by the CLI's ``--jsonl-out`` feature and by
                bench integrations.

        Returns:
            A :class:`ReplayResult` summarizing the run.
        """
        stats = ReplayStatsCollector()
        context = ReplayContext(sm=self._sm)
        header = self._reader.header
        t_start = time.time()
        stats.mark_start(t_start)

        replayed = skipped = failed = 0
        t_wall_origin = time.monotonic()

        for record in self._reader.records():
            if pace is ReplayPace.REALTIME:
                # Sleep just long enough to align to the recorded
                # offset from the start of replay.  No speedup — if
                # the replay machine is slower than recording, the
                # loop simply runs behind.
                target = t_wall_origin + record.t_mono
                now = time.monotonic()
                if now < target:
                    time.sleep(target - now)

            try:
                decoded_args = codecs.decode_args(record.args)
            except Exception:
                skipped += 1
                logger.warning(
                    "trace replay: failed to decode args for %s; skipping",
                    record.qualname,
                    exc_info=True,
                )
                if on_record is not None:
                    on_record(record.qualname, 0.0, True)
                continue

            if not self._dispatcher.has(record.qualname):
                skipped += 1
                logger.warning(
                    "trace replay: no handler for qualname %r; skipping",
                    record.qualname,
                )
                if on_record is not None:
                    on_record(record.qualname, 0.0, True)
                continue

            t0 = time.monotonic()
            try:
                self._dispatcher.dispatch(record.qualname, context, decoded_args)
                latency = time.monotonic() - t0
                stats.record(record.qualname, latency, failed=False)
                replayed += 1
                if on_record is not None:
                    on_record(record.qualname, latency, False)
            except Exception:
                latency = time.monotonic() - t0
                stats.record(record.qualname, latency, failed=True)
                failed += 1
                logger.warning(
                    "trace replay: handler for %s raised",
                    record.qualname,
                    exc_info=True,
                )
                if on_record is not None:
                    on_record(record.qualname, latency, True)

        # Close any contexts the trace left open (truncated trace,
        # or missing __exit__ records).  Releasing these keeps the
        # StorageManager in a consistent state for a follow-up run
        # or inspection.
        for key_tuple, pending in list(context.open_read_contexts.items()):
            while pending:
                cm = pending.popleft()
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    logger.warning(
                        "trace replay: forced exit of dangling "
                        "read_prefetched_results raised (keys=%d)",
                        len(key_tuple),
                        exc_info=True,
                    )
            context.open_read_contexts.pop(key_tuple, None)

        stats.mark_end(time.time())

        # Digest of the replay-side config so callers can compare
        # against ``header.sm_config_digest``.  Same hashing used by
        # the recorder — see :func:`safe_storage_config_dict`.
        safe = safe_storage_config_dict(self._sm_config)
        replay_digest = hashlib.sha256(
            json.dumps(safe, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return ReplayResult(
            records_replayed=replayed,
            records_skipped=skipped,
            records_failed=failed,
            stats=stats,
            header_level=header.level,
            header_digest=header.sm_config_digest,
            replay_config_digest=replay_digest,
        )


#: Callback signature for per-record hooks during replay.  Arguments
#: are ``(qualname, latency_seconds, failed)``.  Declared at module
#: scope so callers can type-annotate their hooks without importing
#: the driver class.
RecordCallback = Callable[[str, float, bool], None]
