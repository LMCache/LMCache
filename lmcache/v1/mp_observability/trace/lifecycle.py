# SPDX-License-Identifier: Apache-2.0

"""Trace recorder lifecycle helpers.

Used by the cache server entry points (``server.py`` and
``http_server.py``) to construct, register, and tear down trace
recorders alongside the EventBus.
"""

# Future
from __future__ import annotations

# Standard
from datetime import datetime, timezone
import os
import tempfile

# First Party
from lmcache import __version__ as lmcache_version
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.mp_coordinator.api import CACHE_EVENT_SCHEMA_VERSION
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.mp_observability.event_bus import EventBus
from lmcache.v1.mp_observability.trace.recorder import (
    EventsTraceRecorder,
    StorageTraceRecorder,
    TraceRecorder,
)

logger = init_logger(__name__)

#: ``--trace-level`` value that records StorageManager calls for replay.
STORAGE_LEVEL = "storage"
#: ``--trace-level`` value that records the cache-event stream the server
#: emits for the coordinator, whether or not a coordinator is configured.
EVENTS_LEVEL = "events"
#: Every level ``--trace-level`` accepts.
TRACE_LEVELS: tuple[str, ...] = (STORAGE_LEVEL, EVENTS_LEVEL)

# The recorder this process created, if any. The events level is fed by the
# HTTP server's cache-event sink, which is built later than the recorder and
# in another module, so it looks the recorder up here.
_active_recorder: TraceRecorder | None = None


def _default_trace_path() -> str:
    """Mint a timestamped path for an unnamed trace file."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        tempfile.gettempdir(),
        f"lmcache-trace-{os.getpid()}-{stamp}.lct",
    )


def get_active_trace_recorder() -> TraceRecorder | None:
    """Return the recorder :func:`maybe_initialize_trace_recorder` created.

    Returns:
        The recorder, or ``None`` when trace recording is off in this process.
    """
    return _active_recorder


def maybe_initialize_trace_recorder(
    bus: EventBus,
    obs_config: ObservabilityConfig,
    storage_manager_config: StorageManagerConfig,
    instance_id: str = "",
) -> TraceRecorder | None:
    """Construct and register a trace recorder if configured.

    Args:
        bus: The active EventBus to subscribe the recorder to.
        obs_config: Observability config carrying the trace flags.
        storage_manager_config: The StorageManagerConfig in use.  Used
            to populate the trace file's header digest so a replay
            driver can detect mismatched configurations.
        instance_id: This MP server's id, written into the ``events``
            level's header so a fleet capture can be told apart by file.

    Returns:
        The created recorder, or ``None`` when ``obs_config.trace_level``
        is unset.

    Raises:
        ValueError: If the level is not one of :data:`TRACE_LEVELS`.

    The recorder is registered on the bus, so :meth:`EventBus.stop`
    will invoke its ``shutdown`` (which flushes and closes the file).
    Callers do not need to track the returned reference for cleanup;
    it is returned only for testing and observation, and available to
    other modules through :func:`get_active_trace_recorder`.
    """
    global _active_recorder

    level = obs_config.trace_level
    if not level:
        return None
    if level not in TRACE_LEVELS:
        raise ValueError(
            f"unsupported trace level {level!r}; expected one of {TRACE_LEVELS}"
        )

    output_path = obs_config.trace_output or _default_trace_path()
    if obs_config.trace_output is None:
        logger.info(
            "trace recording enabled (level=%s); no --trace-output given, "
            "writing to %s",
            level,
            output_path,
        )

    recorder: TraceRecorder
    if level == STORAGE_LEVEL:
        recorder = StorageTraceRecorder(output_path=output_path)
    else:
        recorder = EventsTraceRecorder(
            output_path=output_path,
            level_meta={
                "instance_id": instance_id,
                "cache_event_schema_version": CACHE_EVENT_SCHEMA_VERSION,
                "lmcache_version": lmcache_version,
            },
        )
    recorder.attach_storage_config(storage_manager_config)
    bus.register_subscriber(recorder)
    _active_recorder = recorder
    return recorder
