# SPDX-License-Identifier: Apache-2.0

"""Serde metrics subscriber — OTel instruments for CB serde transforms.

Metrics:

- ``lmcache_blend.serde_encode_duration_seconds`` — histogram of KV→bytes
  serialize wall time. Tagged by ``serde_type`` (fp8/naive/cachegen/kivi),
  ``success``, and ``num_objects``.
- ``lmcache_blend.serde_decode_duration_seconds`` — histogram of bytes→KV
  deserialize wall time. Same tags as encode.
- ``lmcache_blend.serde_bytes_in`` — counter of raw bytes fed into serde
  (pre-compression size for encode, compressed size for decode).
- ``lmcache_blend.serde_bytes_out`` — counter of bytes produced by serde
  (compressed size for encode, raw size for decode).
- ``lmcache_blend.serde_failures`` — counter of failed encode/decode ops.
  Tagged by ``serde_type``, ``direction`` (encode/decode), ``failure_reason``.

Together ``bytes_in / bytes_out`` gives the compression ratio per serde type,
enabling dashboards like:

    rate(lmcache_blend_serde_bytes_out_total[5m])
    / rate(lmcache_blend_serde_bytes_in_total[5m])

which should be < 1 for fp8/cachegen and = 1 for naive.
"""

# Future
from __future__ import annotations

# Standard
import logging
from dataclasses import dataclass

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = logging.getLogger(__name__)

_MAX_PENDING_OPS = 10_000


@dataclass
class _PendingSerdeOp:
    """Tracks an in-flight serde operation for duration measurement."""

    direction: str  # "encode" or "decode"
    serde_type: str
    start_timestamp: float
    num_objects: int


class SerdeMetricsSubscriber(EventSubscriber):
    """Maintains OTel instruments for CB serde/transform operations."""

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.blend.serde")

        self._encode_duration = meter.create_histogram(
            "lmcache_blend.serde_encode_duration_seconds",
            description="Duration of CB serde encode (serialize) operations",
            unit="s",
        )
        self._decode_duration = meter.create_histogram(
            "lmcache_blend.serde_decode_duration_seconds",
            description="Duration of CB serde decode (deserialize) operations",
            unit="s",
        )
        self._bytes_in = meter.create_counter(
            "lmcache_blend.serde_bytes_in",
            description=(
                "Bytes fed into serde (pre-compression for encode, "
                "compressed for decode)"
            ),
            unit="By",
        )
        self._bytes_out = meter.create_counter(
            "lmcache_blend.serde_bytes_out",
            description=(
                "Bytes produced by serde (compressed for encode, raw for decode)"
            ),
            unit="By",
        )
        self._failures = meter.create_counter(
            "lmcache_blend.serde_failures",
            description=(
                "Failed CB serde operations, tagged by serde_type/direction/reason"
            ),
        )

        self._pending_ops: dict[str, _PendingSerdeOp] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.CB_SERDE_ENCODE_START: self._on_encode_start,
            EventType.CB_SERDE_ENCODE_END: self._on_encode_end,
            EventType.CB_SERDE_DECODE_START: self._on_decode_start,
            EventType.CB_SERDE_DECODE_END: self._on_decode_end,
        }

    # ------------------------------------------------------------------
    # Encode path
    # ------------------------------------------------------------------

    def _on_encode_start(self, event: Event) -> None:
        key = f"encode:{event.session_id}"
        self._pending_ops[key] = _PendingSerdeOp(
            direction="encode",
            serde_type=event.metadata.get("serde_type", "unknown"),
            start_timestamp=event.timestamp,
            num_objects=event.metadata.get("num_objects", 1),
        )
        self._cap_pending_ops()

    def _on_encode_end(self, event: Event) -> None:
        serde_type = event.metadata.get("serde_type", "unknown")
        success = bool(event.metadata.get("success", True))
        bytes_in = event.metadata.get("bytes_in", 0)
        bytes_out = event.metadata.get("bytes_out", 0)

        # Duration
        pending = self._pending_ops.pop(f"encode:{event.session_id}", None)
        attrs = {
            "serde_type": serde_type,
            "success": success,
            "num_objects": self._num_objects(event, pending),
        }
        if pending is not None and event.timestamp >= pending.start_timestamp:
            self._encode_duration.record(
                event.timestamp - pending.start_timestamp, attrs
            )

        # Bytes counters
        if bytes_in:
            self._bytes_in.add(
                bytes_in, {"serde_type": serde_type, "direction": "encode"}
            )
        if bytes_out:
            self._bytes_out.add(
                bytes_out, {"serde_type": serde_type, "direction": "encode"}
            )

        # Failures
        if not success:
            reason = event.metadata.get("failure_reason", "unknown")
            self._failures.add(
                1,
                {
                    "serde_type": serde_type,
                    "direction": "encode",
                    "failure_reason": reason,
                },
            )

    # ------------------------------------------------------------------
    # Decode path
    # ------------------------------------------------------------------

    def _on_decode_start(self, event: Event) -> None:
        key = f"decode:{event.session_id}"
        self._pending_ops[key] = _PendingSerdeOp(
            direction="decode",
            serde_type=event.metadata.get("serde_type", "unknown"),
            start_timestamp=event.timestamp,
            num_objects=event.metadata.get("num_objects", 1),
        )
        self._cap_pending_ops()

    def _on_decode_end(self, event: Event) -> None:
        serde_type = event.metadata.get("serde_type", "unknown")
        success = bool(event.metadata.get("success", True))
        bytes_in = event.metadata.get("bytes_in", 0)
        bytes_out = event.metadata.get("bytes_out", 0)

        # Duration
        pending = self._pending_ops.pop(f"decode:{event.session_id}", None)
        attrs = {
            "serde_type": serde_type,
            "success": success,
            "num_objects": self._num_objects(event, pending),
        }
        if pending is not None and event.timestamp >= pending.start_timestamp:
            self._decode_duration.record(
                event.timestamp - pending.start_timestamp, attrs
            )

        # Bytes counters
        if bytes_in:
            self._bytes_in.add(
                bytes_in, {"serde_type": serde_type, "direction": "decode"}
            )
        if bytes_out:
            self._bytes_out.add(
                bytes_out, {"serde_type": serde_type, "direction": "decode"}
            )

        # Failures
        if not success:
            reason = event.metadata.get("failure_reason", "unknown")
            self._failures.add(
                1,
                {
                    "serde_type": serde_type,
                    "direction": "decode",
                    "failure_reason": reason,
                },
            )

    def _cap_pending_ops(self) -> None:
        """Evict oldest entries when _pending_ops exceeds the cap."""
        evicted_count = 0
        while len(self._pending_ops) > _MAX_PENDING_OPS:
            oldest_key = next(iter(self._pending_ops))
            del self._pending_ops[oldest_key]
            evicted_count += 1
        if evicted_count:
            logger.warning(
                "_pending_ops exceeded %d entries; evicted %d oldest entries",
                _MAX_PENDING_OPS,
                evicted_count,
            )

    @staticmethod
    def _num_objects(event: Event, pending: _PendingSerdeOp | None) -> int:
        """Return the object count for duration metric attributes."""
        value = event.metadata.get("num_objects")
        if value is None and pending is not None:
            value = pending.num_objects
        return int(value or 0)
