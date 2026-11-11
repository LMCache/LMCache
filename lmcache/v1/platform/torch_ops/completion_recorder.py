# SPDX-License-Identifier: Apache-2.0
# Standard
import threading

# ---------------------------------------------------------------------------
# Completion recorder fallback (no CUDA stream ordering; enqueue immediately)
# ---------------------------------------------------------------------------

_completion_lock = threading.Lock()
_completion_buffer: list[tuple[str, bytes]] = []


def record_completion_on_stream(
    cuda_stream_ptr: int, kind: str, payload: bytes
) -> None:
    """Fallback: immediately enqueue the completion without stream ordering.

    Args:
        cuda_stream_ptr: Ignored on non-CUDA path.
        kind: Dispatch key identifying the handler (e.g. "finish_write").
        payload: Opaque msgpack-encoded bytes forwarded to the handler.
    """
    with _completion_lock:
        _completion_buffer.append((kind, payload))


def drain_recorded_completions() -> list[tuple[str, bytes]]:
    """Fallback: atomically drain and return all pending completions.

    Returns:
        List of (kind, payload) pairs recorded since the last drain.
    """
    with _completion_lock:
        items = list(_completion_buffer)
        _completion_buffer.clear()
    return items
