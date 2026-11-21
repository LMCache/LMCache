# SPDX-License-Identifier: Apache-2.0
"""Probe L2 adapters' backends and route around the ones that are down.

The monitor owns one :class:`BackendState` per registered adapter. It acts only
on adapters that declare ``supports_health_probe``; every other adapter stays
``HEALTHY`` forever, so adapters that have not opted in behave as before.

Each round (``check_once``) probes every managed adapter on its own thread,
bounded by ``probe_timeout_s`` plus a short grace, so a probe that hangs (a
dead NFS mount, a silent peer) delays neither the round nor the other
adapters. A hung probe counts as a failed probe.

State transitions, driven by the probe result:

* ``OK``: failure count resets; ``UNAVAILABLE`` becomes ``HEALTHY``.
* ``UNREACHABLE``: after ``failure_threshold`` consecutive failures a
  ``HEALTHY`` adapter becomes ``UNAVAILABLE``.
* ``RECONNECT_NEEDED``: the adapter's own connection is broken although the
  backend answers. If the adapter supports it, ``reconnect`` runs (immediately
  when ``UNAVAILABLE``, after ``failure_threshold`` consecutive failures when
  ``HEALTHY``), with the state ``RECOVERING`` meanwhile. Otherwise it counts
  like ``UNREACHABLE``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import L2HealthConfig
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.health import (
    BackendState,
    HealthProbe,
    L2ReconnectError,
    ProbeStatus,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

logger = init_logger(__name__)

# How long ``stop`` waits for the loop thread and for a running reconnect.
_STOP_JOIN_TIMEOUT_S = 5.0

# How long past ``probe_timeout_s`` a probe may take before the monitor gives
# up on it. A probe that spends its whole budget returns just after the
# budget is gone; without the grace its result would race the deadline.
_PROBE_GRACE_S = 1.0


@dataclass
class _Entry:
    """Monitor-side record of one adapter. Fields marked "lock" are guarded by
    ``L2HealthMonitor._lock``."""

    adapter: L2AdapterInterface
    name: str
    state: BackendState = BackendState.HEALTHY  # lock
    consecutive_failures: int = 0  # lock
    last_reason: str = ""  # lock
    last_change_unix_s: float = 0.0  # lock
    last_ok_unix_s: float | None = None  # lock
    # Written by the monitor loop thread only.
    probe_thread: threading.Thread | None = None
    # Written by the probe thread just before it exits; read by the loop thread
    # only after the thread has exited.
    probe_result: HealthProbe | None = None
    reconnect_thread: threading.Thread | None = None  # lock


class L2HealthMonitor:
    """Tracks whether each L2 adapter's backend is usable.

    Controllers ask :meth:`is_available` before routing work to an adapter.
    """

    def __init__(self, config: L2HealthConfig) -> None:
        """
        Args:
            config: Probe interval, probe timeout and failure threshold. An
                interval of ``0`` makes :meth:`start` a no-op, so no adapter
                is ever probed.
        """
        self._config = config
        self._lock = threading.Lock()
        self._entries: dict[int, _Entry] = {}
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._event_bus = get_event_bus()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, adapter_id: int, adapter: L2AdapterInterface, name: str) -> None:
        """Start tracking an adapter, initially ``HEALTHY``.

        Args:
            adapter_id: The adapter's stable id, as controllers know it.
            adapter: The adapter to probe.
            name: The adapter's type name, for logs, events and status.
        """
        with self._lock:
            self._entries[adapter_id] = _Entry(
                adapter=adapter, name=name, last_change_unix_s=time.time()
            )

    def unregister(self, adapter_id: int) -> None:
        """Stop tracking an adapter. A reconnect already running is left to
        finish and its outcome is discarded."""
        with self._lock:
            self._entries.pop(adapter_id, None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_available(self, adapter_id: int) -> bool:
        """Whether new work may be routed to the adapter.

        Args:
            adapter_id: The adapter's stable id.

        Returns:
            ``False`` only while a tracked adapter is ``UNAVAILABLE`` or
            ``RECOVERING``. An id the monitor does not track is available.
        """
        with self._lock:
            entry = self._entries.get(adapter_id)
            return entry is None or entry.state is BackendState.HEALTHY

    def report_status(self) -> dict:
        """Return a JSON-serializable snapshot for ``/status``.

        Returns:
            ``{"enabled": bool, "degraded": bool, "adapters": [...]}``. Each
            adapter entry has ``adapter_id`` (int), ``l2_name`` (str), ``state``
            (str, a ``BackendState`` value), ``reason`` (str, the last probe's
            reason), ``consecutive_failures`` (int) and
            ``seconds_since_last_ok`` (float, ``None`` if never probed OK).
            Only adapters that support probing are listed; ``degraded`` is true
            when any of them is not ``HEALTHY``.
        """
        now = time.time()
        with self._lock:
            adapters = [
                {
                    "adapter_id": adapter_id,
                    "l2_name": entry.name,
                    "state": entry.state.value,
                    "reason": entry.last_reason,
                    "consecutive_failures": entry.consecutive_failures,
                    "seconds_since_last_ok": (
                        None
                        if entry.last_ok_unix_s is None
                        else round(now - entry.last_ok_unix_s, 1)
                    ),
                }
                for adapter_id, entry in sorted(self._entries.items())
                if entry.adapter.supports_health_probe
            ]
        return {
            "enabled": self._config.check_interval_s > 0,
            "degraded": any(a["state"] != BackendState.HEALTHY.value for a in adapters),
            "adapters": adapters,
        }

    def get_availability_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter availability in OTel-observation shape.

        Returns:
            ``(1 or 0, {"l2_name": ..., "adapter_index": ...})`` for every
            adapter that supports probing; ``1`` means ``HEALTHY``.
        """
        with self._lock:
            return [
                (
                    int(entry.state is BackendState.HEALTHY),
                    {"l2_name": entry.name, "adapter_index": adapter_id},
                )
                for adapter_id, entry in sorted(self._entries.items())
                if entry.adapter.supports_health_probe
            ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the probe loop thread (does nothing when the interval is 0)."""
        if self._config.check_interval_s <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="l2-health-monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop and wait briefly for a running reconnect, so the
        caller can close the adapters afterwards."""
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
        with self._lock:
            reconnects = [
                e.reconnect_thread for e in self._entries.values() if e.reconnect_thread
            ]
        for thread in reconnects:
            thread.join(timeout=_STOP_JOIN_TIMEOUT_S)

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    def check_once(self) -> None:
        """Run one probe round and apply the results, then return.

        The loop thread calls this every ``check_interval_s``; tests call it
        directly. Adapters being reconnected are skipped.
        """
        with self._lock:
            managed = [
                (adapter_id, entry)
                for adapter_id, entry in self._entries.items()
                if entry.adapter.supports_health_probe
                and entry.state is not BackendState.RECOVERING
            ]

        started: list[tuple[int, _Entry, threading.Thread]] = []
        for adapter_id, entry in managed:
            if entry.probe_thread is not None and entry.probe_thread.is_alive():
                self._apply(
                    adapter_id,
                    entry,
                    HealthProbe(
                        ProbeStatus.UNREACHABLE, "previous probe has not returned"
                    ),
                )
                continue
            entry.probe_result = None
            thread = threading.Thread(
                target=self._run_probe,
                args=(entry,),
                daemon=True,
                name=f"l2-probe-{adapter_id}",
            )
            entry.probe_thread = thread
            thread.start()
            started.append((adapter_id, entry, thread))

        deadline = time.monotonic() + self._config.probe_timeout_s + _PROBE_GRACE_S
        for adapter_id, entry, thread in started:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            result = None if thread.is_alive() else entry.probe_result
            if result is None:
                result = HealthProbe(
                    ProbeStatus.UNREACHABLE,
                    "probe did not return within "
                    f"{self._config.probe_timeout_s + _PROBE_GRACE_S}s",
                )
            self._apply(adapter_id, entry, result)

    def _loop(self) -> None:
        while not self._stop_flag.wait(self._config.check_interval_s):
            try:
                self.check_once()
            except Exception:
                logger.exception("L2 health check round failed")

    def _run_probe(self, entry: _Entry) -> None:
        try:
            entry.probe_result = entry.adapter.probe_health(
                self._config.probe_timeout_s
            )
        except Exception as exc:
            logger.exception("L2 adapter %s probe_health raised", entry.name)
            entry.probe_result = HealthProbe(
                ProbeStatus.UNREACHABLE, f"probe raised {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _apply(self, adapter_id: int, entry: _Entry, probe: HealthProbe) -> None:
        """Fold one probe result into the adapter's state."""
        start_reconnect = False
        with self._lock:
            if self._entries.get(adapter_id) is not entry:
                return
            if entry.state is BackendState.RECOVERING:
                return
            old_state = entry.state
            entry.last_reason = probe.reason
            if probe.status is ProbeStatus.OK:
                entry.consecutive_failures = 0
                entry.last_ok_unix_s = time.time()
                entry.state = BackendState.HEALTHY
            else:
                entry.consecutive_failures += 1
                threshold_reached = (
                    entry.consecutive_failures >= self._config.failure_threshold
                )
                if (
                    probe.status is ProbeStatus.RECONNECT_NEEDED
                    and entry.adapter.supports_reconnect
                ):
                    if old_state is BackendState.UNAVAILABLE or threshold_reached:
                        entry.state = BackendState.RECOVERING
                        start_reconnect = True
                elif old_state is BackendState.HEALTHY and threshold_reached:
                    entry.state = BackendState.UNAVAILABLE
            new_state = entry.state
            if new_state is not old_state:
                entry.last_change_unix_s = time.time()
        if new_state is not old_state:
            self._announce(adapter_id, entry, old_state, new_state, probe.reason)
        if start_reconnect:
            self._start_reconnect(adapter_id, entry)

    def _start_reconnect(self, adapter_id: int, entry: _Entry) -> None:
        thread = threading.Thread(
            target=self._run_reconnect,
            args=(adapter_id, entry),
            daemon=True,
            name=f"l2-reconnect-{adapter_id}",
        )
        with self._lock:
            entry.reconnect_thread = thread
        thread.start()

    def _run_reconnect(self, adapter_id: int, entry: _Entry) -> None:
        try:
            entry.adapter.reconnect()
        except L2ReconnectError as exc:
            logger.warning("L2 adapter %s reconnect failed: %s", entry.name, exc)
            outcome = BackendState.UNAVAILABLE
            reason = f"reconnect failed: {exc}"
        except Exception as exc:
            logger.exception("L2 adapter %s reconnect raised", entry.name)
            outcome = BackendState.UNAVAILABLE
            reason = f"reconnect raised {type(exc).__name__}: {exc}"
        else:
            outcome = BackendState.HEALTHY
            reason = ""
        with self._lock:
            if self._entries.get(adapter_id) is not entry:
                return
            entry.reconnect_thread = None
            entry.state = outcome
            entry.last_reason = reason
            entry.last_change_unix_s = time.time()
            if outcome is BackendState.HEALTHY:
                entry.consecutive_failures = 0
                entry.last_ok_unix_s = time.time()
        self._announce(adapter_id, entry, BackendState.RECOVERING, outcome, reason)

    def _announce(
        self,
        adapter_id: int,
        entry: _Entry,
        old_state: BackendState,
        new_state: BackendState,
        reason: str,
    ) -> None:
        log = logger.warning if new_state is not BackendState.HEALTHY else logger.info
        log(
            "L2 adapter %d (%s): %s -> %s%s",
            adapter_id,
            entry.name,
            old_state.value,
            new_state.value,
            f" ({reason})" if reason else "",
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_ADAPTER_HEALTH_CHANGED,
                metadata={
                    "adapter_index": adapter_id,
                    "l2_name": entry.name,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "reason": reason,
                },
            )
        )
