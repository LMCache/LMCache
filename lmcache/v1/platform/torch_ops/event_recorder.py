# SPDX-License-Identifier: Apache-2.0

# Standard
import threading

# ---------------------------------------------------------------------------
# Event recorder fallback (no CUDA stream ordering; timestamp immediately)
# ---------------------------------------------------------------------------

_event_lock = threading.Lock()
_event_buffer: list[tuple[str, str, float, dict[str, str], dict[str, int]]] = []


def record_event_on_stream(
    cuda_stream_ptr: int,
    event_type_name: str,
    session_id: str,
    str_metadata: dict[str, str],
    int_metadata: dict[str, int],
) -> None:
    """Fallback: immediately record the event without CUDA stream ordering.

    The wall-clock timestamp is captured at call time (no host-callback).

    Args:
        cuda_stream_ptr: Ignored on non-CUDA path.
        event_type_name: Event type identifier (e.g. "mp.store.start").
        session_id: Session identifier for the event.
        str_metadata: String-valued metadata dict.
        int_metadata: Integer-valued metadata dict.
    """
    # Standard
    import time

    ts = time.time()
    with _event_lock:
        _event_buffer.append(
            (event_type_name, session_id, ts, dict(str_metadata), dict(int_metadata))
        )


def drain_recorded_events() -> list[
    tuple[str, str, float, dict[str, str], dict[str, int]]
]:
    """Fallback: atomically drain and return all pending events.

    Returns:
        List of (event_type_name, session_id, timestamp, str_metadata,
        int_metadata) tuples recorded since the last drain.
    """
    with _event_lock:
        items = list(_event_buffer)
        _event_buffer.clear()
    return items
