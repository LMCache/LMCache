# SPDX-License-Identifier: Apache-2.0
"""Experimental, single-model DRAM retention from measured demand profiles."""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
import json
import math
import threading
import time

# Third Party
from sortedcontainers import SortedList

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import EvictionAction, EvictionDestination
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventCallback,
    EventSubscriber,
)

logger = init_logger(__name__)
_MODES = {"full", "no-frequency", "no-distance", "no-cost", "observe", "two-hit"}


class ProfileStatistics(TypedDict):
    """Read-only copied policy counters; not completed physical evictions."""

    mode: str
    requests: int
    history_keys: int
    decisions: int


@dataclass(frozen=True)
class ProfileConfig:
    """Frozen calibration for one model and chunk size, not a fleet policy.

    ``prefix_recompute_ms[n-1]`` is measured engine prefill time for n chunks,
    calibrated without LMCache or GPU APC. It is not client TTFT.
    The policy uses its per-chunk average as a heuristic saved-work weight,
    not as a calibrated future-use probability or an optimal cost estimate.
    """

    model_name: str
    chunk_size: int
    prefix_recompute_ms: tuple[float, ...]
    mode: str = "full"
    history_keys: int = 65536

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_name, str)
            or not self.model_name
            or type(self.chunk_size) is not int
            or self.chunk_size < 1
        ):
            raise ValueError(
                "model_name and a positive integer chunk_size are required"
            )
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}")
        if (
            type(self.history_keys) is not int
            or not 1 <= self.history_keys <= 1_000_000
        ):
            raise ValueError("history_keys must be an integer in [1, 1000000]")
        if not 1 <= len(self.prefix_recompute_ms) <= 4096 or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in self.prefix_recompute_ms
        ):
            raise ValueError(
                "prefix_recompute_ms requires 1..4096 finite positive costs"
            )

    @classmethod
    def load(cls, path: str) -> ProfileConfig:
        """Load bounded operator calibration JSON; reject absent/invalid input.

        Args:
            path: File containing this dataclass's fields and schema_version=1.
                Optional metadata records experimental provenance only.

        Returns:
            Validated immutable calibration.

        Raises:
            OSError: The file cannot be opened.
            ValueError: The schema, size or field values are invalid.
        """
        with Path(path).open("rb") as source:
            data = source.read((1 << 20) + 1)
        if len(data) > 1 << 20:
            raise ValueError("profiling calibration exceeds 1 MiB")
        record = json.loads(data)
        fields = {
            "model_name",
            "chunk_size",
            "prefix_recompute_ms",
            "mode",
            "history_keys",
        }
        if not isinstance(record, dict) or set(record) - fields - {
            "schema_version",
            "metadata",
        }:
            raise ValueError("invalid profiling calibration fields")
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != 1
        ):
            raise ValueError("profiling schema_version must be 1")
        try:
            costs = record["prefix_recompute_ms"]
            if not isinstance(costs, list):
                raise ValueError("prefix_recompute_ms must be a list")
            return cls(
                model_name=record["model_name"],
                chunk_size=record["chunk_size"],
                prefix_recompute_ms=tuple(costs),
                mode=record.get("mode", "full"),
                history_keys=record.get("history_keys", 65536),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("missing or invalid profiling calibration") from exc


@dataclass
class _History:
    ordinal: int
    frequency: float
    reuse_distance: int
    cost_ms: float
    position: int
    parent: bytes | None
    first_access: float
    last_access: float


class ProfiledEvictionPolicy(EvictionPolicy, EventSubscriber):
    """Keep useful observed prefixes in DRAM; use ordinary eviction ownership.

    Demand MP_LOOKUP events update bounded key-only ghost history, including
    misses. Reuse distance counts distinct tracked logical chunks, not accesses
    or wall time. A forgotten chunk is cold. At pressure, rank resident keys by
    frequency * measured average recompute cost / (1 + distance/resident_count).
    Child scores cannot exceed a resident ancestor's score; ties evict suffixes
    first. This is a heuristic, not expected utility or a universal improvement.

    Only the configured model, empty salt, TP1 shard and group zero are profiled in
    this experimental TP1 deployment. Unknown keys remain ordinary low-priority
    LRU candidates. It does not claim multi-tenant or cross-model qualification.
    Callbacks and snapshots share one lock; sorting and eligibility callbacks
    run outside it. Returned actions do not remove keys or release buffer locks.
    Distances describe delivered events only: exact demand history requires
    loss-free delivery. Independent policy retirement is not supported; the
    process-lifetime bus must drain and stop before invoking shutdown.
    """

    def __init__(
        self,
        config: ProfileConfig,
        event_bus: EventBus,
        *,
        clock: Callable[[], float] = time.monotonic,
        register: bool = True,
    ) -> None:
        """Register demand observation on an already configured live event bus.

        Args:
            config: Measured calibration and the selected ablation mode.
            event_bus: Owner of subscriber shutdown; initialize it only once.
            clock: Process-local monotonic clock, injectable for deterministic tests.
            register: False when a transactional owner will subscribe on commit.
        """
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._resident: OrderedDict[ObjectKey, None] = OrderedDict()
        self._history: OrderedDict[bytes, _History] = OrderedDict()
        self._stack: SortedList[int] = SortedList()
        self._ordinal = 0
        self._requests = 0
        self._decisions = 0
        self._destinations: list[EvictionDestination] = []
        if register:
            event_bus.register_subscriber(self)

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Observe only demand lookups, including misses, on the drain thread."""
        return {EventType.MP_LOOKUP: self._observe}

    def shutdown(self) -> None:
        """Release key-only profile and residency metadata after bus drain."""
        with self._lock:
            self._history.clear()
            self._stack.clear()
            self._resident.clear()

    def statistics(self) -> ProfileStatistics:
        """Return bounded-profile counters, not physical eviction completion.

        Returns:
            Mode, observed requests, history entries and selection batches.
        """
        with self._lock:
            return {
                "mode": self._config.mode,
                "requests": self._requests,
                "history_keys": len(self._history),
                "decisions": self._decisions,
            }

    def profile(self, chunk_hash: bytes) -> dict[str, int | float] | None:
        """Return a copied observation for diagnostics, never a live state view.

        Args:
            chunk_hash: Logical chunk in this calibration's model namespace.

        Returns:
            Frequency, last distinct-key reuse distance, mean-cost weight and
            prefix position, or None when history no longer contains it.
        """
        with self._lock:
            item = self._history.get(chunk_hash)
            if item is None:
                return None
            return {
                "frequency": item.frequency,
                "reuse_distance": item.reuse_distance,
                "cost_ms": item.cost_ms,
                "position": item.position,
            }

    def register_eviction_destination(self, destination: EvictionDestination) -> None:
        """Use the first destination registered by the existing controller."""
        with self._lock:
            if destination not in self._destinations:
                self._destinations.append(destination)

    def on_keys_created(self, keys: list[ObjectKey]) -> None:
        """Track reserved/admitted keys in the same prefix-aware order as LRU."""
        with self._lock:
            for key in reversed(keys):
                self._resident[key] = None
                self._resident.move_to_end(key)

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        """Refresh resident ordering without counting reads twice as demand."""
        with self._lock:
            for key in reversed(keys):
                if key in self._resident:
                    self._resident.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        """Forget completed residency; retain bounded demand ghost history."""
        with self._lock:
            for key in keys:
                self._resident.pop(key, None)

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """Select unlocked victims; actual deletion stays with the controller.

        Args:
            expected_ratio: Fraction of tracked residents requested, clamped
                to [0,1], with at least one for a positive ratio.
            key_eligible_filter: Existing lock/eligibility check; called with
                no policy lock held. The manager revalidates actual deletion.
            cache_salt: Ignored, matching non-isolated LRU.

        Returns:
            Zero or one action, without speculative residency deletion.
        """
        with self._lock:
            keys = list(self._resident)
            count = (
                min(len(keys), max(1, int(len(keys) * min(1.0, expected_ratio))))
                if expected_ratio > 0
                else 0
            )
            if not count:
                return []
            profiles: dict[bytes, tuple[float, int, bytes | None]] = {}
            profiled_keys: set[ObjectKey] = set()
            for key in keys:
                item = self._history.get(key.chunk_hash)
                if (
                    key.model_name == self._config.model_name
                    and not key.cache_salt
                    and key.kv_rank == ObjectKey.ComputeKVRank(1, 0, 1, 0)
                    and key.object_group_id == 0
                    and not key.identity.tags
                    and item is not None
                ):
                    age = len(self._stack) - self._stack.bisect_right(item.ordinal)
                    distance = max(age, item.reuse_distance)
                    cost = self._retention_cost_ms(key, item)
                    if self._config.mode == "no-cost":
                        cost = 1.0
                    penalty = (
                        1.0
                        if self._config.mode == "no-distance"
                        else 1 + distance / max(1, len(keys))
                    )
                    frequency = (
                        1
                        if self._config.mode == "no-frequency"
                        else self._frequency_weight(item)
                    )
                    profiles[key.chunk_hash] = (
                        frequency * cost / penalty,
                        item.position,
                        item.parent,
                    )
                    profiled_keys.add(key)
            destination = (
                self._destinations[0]
                if self._destinations
                else EvictionDestination.DISCARD
            )
            self._decisions += 1
            counters = (self._requests, len(self._history), self._decisions)
        if self._rank_snapshot(keys, profiled_keys):
            scores: dict[bytes, float] = {}
            for chunk, (score, _, parent) in sorted(
                profiles.items(), key=lambda pair: pair[1][1]
            ):
                scores[chunk] = (
                    min(score, scores.get(parent, 0.0)) if parent is not None else score
                )
            keys.sort(
                key=lambda key: (
                    scores.get(key.chunk_hash, 0.0) if key in profiled_keys else 0.0,
                    -profiles[key.chunk_hash][1] if key in profiled_keys else 0,
                )
            )
        victims = []
        for key in keys:
            if key_eligible_filter is None or key_eligible_filter(key):
                victims.append(key)
                if len(victims) == count:
                    break
        logger.info(
            "Profiling eviction %s",
            json.dumps(
                {
                    "mode": self._config.mode,
                    "requests": counters[0],
                    "history_keys": counters[1],
                    "decisions": counters[2],
                    "selected": len(victims),
                    "profiled_residents": len(profiled_keys),
                }
            ),
        )
        return (
            [EvictionAction(keys=victims, destination=destination)] if victims else []
        )

    def _observe(self, event: Event) -> None:
        metadata = event.metadata
        hashes = metadata.get("chunk_hashes")
        if (
            metadata.get("model_name") != self._config.model_name
            or metadata.get("chunk_size") != self._config.chunk_size
            or not isinstance(hashes, list)
            or not hashes
            or len(hashes) > len(self._config.prefix_recompute_ms)
            or any(not isinstance(chunk, bytes) for chunk in hashes)
            or len(set(hashes)) != len(hashes)
            or metadata.get("tags", {}) != {}
            or metadata.get("cache_salt", "") != ""
            or metadata.get("world_size", 1) != 1
        ):
            return
        cost = self._config.prefix_recompute_ms[len(hashes) - 1] / len(hashes)
        with self._lock:
            now = self._clock()
            for position, chunk in enumerate(hashes):
                previous = self._history.get(chunk)
                parent = hashes[position - 1] if position else None
                if previous is not None and (
                    previous.parent != parent or previous.position != position
                ):
                    return
            self._requests += 1
            parent = None
            for position, chunk in enumerate(hashes):
                old = self._history.pop(chunk, None)
                distance = (
                    0
                    if old is None
                    else len(self._stack) - self._stack.bisect_right(old.ordinal)
                )
                if old is not None:
                    self._stack.remove(old.ordinal)
                self._ordinal += 1
                self._history[chunk] = _History(
                    self._ordinal,
                    self._next_frequency(old, now),
                    distance,
                    cost if old is None else (old.cost_ms + cost) / 2,
                    position,
                    parent,
                    now if old is None else old.first_access,
                    now,
                )
                self._stack.add(self._ordinal)
                if len(self._history) > self._config.history_keys:
                    _, forgotten = self._history.popitem(last=False)
                    self._stack.remove(forgotten.ordinal)
                parent = chunk

    def _retention_cost_ms(self, key: ObjectKey, item: _History) -> float:
        return item.cost_ms

    def _frequency_weight(self, item: _History) -> float:
        return item.frequency

    def _next_frequency(self, old: _History | None, now: float) -> float:
        return 1 if old is None else old.frequency + 1

    def _ranking_enabled(self) -> bool:
        return self._config.mode != "observe"

    def _rank_snapshot(self, keys: list[ObjectKey], profiled: set[ObjectKey]) -> bool:
        return self._ranking_enabled()
