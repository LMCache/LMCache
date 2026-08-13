# SPDX-License-Identifier: Apache-2.0

"""Gather/DMA phase transfer metrics subscriber.

Consumes ``MP_TRANSFER_PHASE_SAMPLES`` events and emits, labeled by
``device_index`` and ``direction``:

  - ``lmcache_mp.transfer_kernel_throughput``  — per-batch-step GB/s of
    gather/scatter kernel sections (paged blocks <-> GPU staging buffers)
  - ``lmcache_mp.transfer_staging_throughput`` — per-batch-step GB/s of
    host<->device DMA staging sections (GPU staging buffers <-> pinned
    host memory)
  - ``lmcache_mp.transfer_phase_bytes`` / ``lmcache_mp.transfer_phase_busy_time``
    — cumulative bytes and busy seconds per phase (extra ``phase`` label).
    ``rate(bytes) / rate(busy_time)`` gives the byte-weighted aggregate
    throughput the per-step histograms cannot provide, and
    ``rate(busy_time)`` against wall time gives each phase's time share.

Sample layout: see ``EventType.MP_TRANSFER_PHASE_SAMPLES``.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Sequence

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.lmcache_native import TransferDirection
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber
from lmcache.v1.platform.ops_types import TransferPhase

# Label names derived from the enum members; pybind11 enums do not raise
# ValueError on unknown values, so name lookup must not rely on construction.
_DIRECTION_NAMES = {
    int(member): name.lower() for name, member in TransferDirection.__members__.items()
}


def _direction_name(direction: Any) -> str:
    """Map a ``TransferDirection`` value to its lowercase label name.

    Falls back to ``str(direction)`` for values a version-skewed native
    module might emit that this build's enum does not know.
    """
    try:
        return _DIRECTION_NAMES.get(direction, str(direction))
    except TypeError:
        return str(direction)


class TransferPhaseSubscriber(EventSubscriber):
    """Records per-phase transfer metrics from executor timing samples."""

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache_mp.perf")
        self._kernel_hist = meter.create_histogram(
            "lmcache_mp.transfer_kernel_throughput",
            description=(
                "Histogram of gather/scatter kernel-section throughput in "
                "GB/s, one sample per plan-executor batch step."
            ),
            unit="GB/s",
        )
        self._staging_hist = meter.create_histogram(
            "lmcache_mp.transfer_staging_throughput",
            description=(
                "Histogram of DMA staging-section throughput in GB/s, one "
                "sample per plan-executor batch step."
            ),
            unit="GB/s",
        )
        self._bytes_counter = meter.create_counter(
            "lmcache_mp.transfer_phase_bytes",
            description=(
                "Cumulative bytes moved per transfer phase (label 'phase'). "
                "Divide its rate by transfer_phase_busy_time's rate for the "
                "byte-weighted aggregate phase throughput."
            ),
            unit="By",
        )
        self._busy_counter = meter.create_counter(
            "lmcache_mp.transfer_phase_busy_time",
            description=(
                "Cumulative stream-clocked busy seconds per transfer phase "
                "(label 'phase'); its rate against wall time is the phase's "
                "time share."
            ),
            unit="s",
        )

    # -- EventSubscriber interface -----------------------------------------

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_TRANSFER_PHASE_SAMPLES: self._on_samples,
        }

    # -- Recording ----------------------------------------------------------

    def _on_samples(self, event: Event) -> None:
        samples = event.metadata.get("samples", ())
        for sample in samples:
            self._record_sample(sample)

    def _record_sample(self, sample: Sequence[Any]) -> None:
        """Record one ``(phase, direction, device_index, ms, nbytes)`` tuple.

        Malformed or degenerate samples (wrong arity, non-numeric fields,
        non-positive time/bytes, unknown phase) are dropped silently so a
        version-skewed native module cannot break the drain thread.
        """
        if len(sample) != 5:
            return
        phase, direction, device_index, elapsed_ms, nbytes = sample
        if not isinstance(elapsed_ms, (int, float)) or not isinstance(
            nbytes, (int, float)
        ):
            return
        if elapsed_ms <= 0 or nbytes <= 0:
            return
        if phase == TransferPhase.KERNEL:
            hist = self._kernel_hist
        elif phase == TransferPhase.STAGING:
            hist = self._staging_hist
        else:
            return
        attrs: dict[str, Any] = {
            "device_index": str(device_index),
            "direction": _direction_name(direction),
        }
        elapsed_s = elapsed_ms / 1e3
        hist.record(nbytes / elapsed_s / 1e9, attributes=attrs)

        phase_attrs: dict[str, Any] = {
            **attrs,
            "phase": TransferPhase(phase).name.lower(),
        }
        self._bytes_counter.add(int(nbytes), attributes=phase_attrs)
        self._busy_counter.add(elapsed_s, attributes=phase_attrs)
