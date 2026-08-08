# SPDX-License-Identifier: Apache-2.0

"""Gather/DMA phase throughput metrics subscriber.

Consumes ``MP_TRANSFER_PHASE_SAMPLES`` events and emits two OTel
histograms in GB/s, labeled by ``device_index`` and ``direction``:

  - ``lmcache_mp.transfer_kernel_throughput``  — gather/scatter kernel
    sections (paged blocks <-> GPU staging buffers)
  - ``lmcache_mp.transfer_staging_throughput`` — host<->device DMA staging
    sections (GPU staging buffers <-> pinned host memory)

Sample layout: see ``EventType.MP_TRANSFER_PHASE_SAMPLES``.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Sequence

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

_PHASE_KERNEL = 0
_PHASE_STAGING = 1
_DIRECTION_NAMES = {0: "h2d", 1: "d2h"}


class TransferPhaseSubscriber(EventSubscriber):
    """Records per-phase transfer throughput from executor timing samples."""

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

        Malformed or degenerate samples are dropped silently so a
        version-skewed native module cannot break the drain thread.
        """
        if len(sample) != 5:
            return
        phase, direction, device_index, elapsed_ms, nbytes = sample
        if elapsed_ms <= 0 or nbytes <= 0:
            return
        if phase == _PHASE_KERNEL:
            hist = self._kernel_hist
        elif phase == _PHASE_STAGING:
            hist = self._staging_hist
        else:
            return
        attrs: dict[str, Any] = {
            "device_index": str(device_index),
            "direction": _DIRECTION_NAMES.get(direction, str(direction)),
        }
        gb_per_s = nbytes / (elapsed_ms / 1e3) / 1e9
        hist.record(gb_per_s, attributes=attrs)
