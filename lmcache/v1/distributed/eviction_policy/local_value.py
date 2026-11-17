# SPDX-License-Identifier: Apache-2.0
"""Opt-in local DRAM retention and L2 admission using measured marginal costs."""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import math
import time

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.profiled import (
    ProfileConfig,
    ProfiledEvictionPolicy,
    _History,
)
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.storage_controllers.store_policy import StorePolicy
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.mp_observability.event import Event
from lmcache.v1.mp_observability.event_bus import EventBus


def model_revision(path: Path) -> str:
    """Hash model files recursively; reject unbound index references/dir links."""
    if not (path / "config.json").is_file():
        raise ValueError("local policy requires an immutable local model directory")
    inventory = {}
    references: list[str] = []
    for entry in sorted(path.rglob("*")):
        if entry.is_dir() and entry.is_symlink():
            raise ValueError("symlinked model directories are not qualified")
        if entry.is_file():
            with entry.open("rb") as source:
                inventory[str(entry.relative_to(path))] = hashlib.file_digest(
                    source, "sha256"
                ).hexdigest()
            if entry.name.endswith((".safetensors.index.json", ".bin.index.json")):
                index = json.loads(entry.read_bytes())
                mapping = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(mapping, dict) or any(
                    not isinstance(value, str) for value in mapping.values()
                ):
                    raise ValueError("model index has an invalid weight map")
                references.extend(mapping.values())
    for reference in references:
        name = Path(reference)
        if name.is_absolute() or ".." in name.parts or str(name) not in inventory:
            raise ValueError("model index references an unbound file")
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, kw_only=True)
class LocalValueConfig(ProfileConfig):
    """Costs for complete contiguous prefixes, in comparable milliseconds.

    Arrays have the same geometry and monotone nonnegative cumulative costs.
    Their first differences price disjoint prefix increments, not repeated
    whole-prefix savings. Metadata in the JSON must retain raw calibration;
    these tables alone do not establish measurement provenance.
    """

    prefix_fetch_l1_ms: tuple[float, ...]
    prefix_fetch_l2_ms: tuple[float, ...]
    prefix_write_l2_ms: tuple[float, ...]
    model_revision: str
    dtypes: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    half_life_s: float = 30.0
    horizon_s: float = 60.0
    config_ttl_s: float = 3600.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.model_revision, str)
            or len(self.model_revision) != 64
            or any(c not in "0123456789abcdef" for c in self.model_revision)
            or not isinstance(self.dtypes, tuple)
            or not self.dtypes
            or len(self.dtypes) != len(self.shapes)
            or any(
                not isinstance(dtype, str) or not dtype.startswith("torch.")
                for dtype in self.dtypes
            )
            or not isinstance(self.shapes, tuple)
            or any(
                not isinstance(shape, tuple)
                or not shape
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                for shape in self.shapes
            )
        ):
            raise ValueError("model revision, dtype and immutable layout are required")
        for values in (
            self.prefix_recompute_ms,
            self.prefix_fetch_l1_ms,
            self.prefix_fetch_l2_ms,
            self.prefix_write_l2_ms,
        ):
            if (
                not isinstance(values, tuple)
                or len(values) != len(self.prefix_recompute_ms)
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                    for value in values
                )
                or any(
                    left > right
                    for left, right in zip(values, values[1:], strict=False)
                )
            ):
                raise ValueError(
                    "cost tables must have equal length and monotone costs"
                )
        for value in (self.half_life_s, self.horizon_s, self.config_ttl_s):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("policy lifetimes must be finite positive numbers")

    @classmethod
    def load(cls, path: str) -> LocalValueConfig:
        """Load bounded schema-2 JSON; invalid or missing costs fail startup."""
        with Path(path).open("rb") as source:
            data = source.read((1 << 20) + 1)
        if len(data) > 1 << 20:
            raise ValueError("local calibration exceeds 1 MiB")
        record = json.loads(data)
        if (
            not isinstance(record, dict)
            or type(record.get("schema_version")) is not int
            or record.pop("schema_version") != 2
        ):
            raise ValueError("local value calibration requires schema_version=2")
        record.pop("metadata", None)
        try:
            for name in (
                "prefix_recompute_ms",
                "prefix_fetch_l1_ms",
                "prefix_fetch_l2_ms",
                "prefix_write_l2_ms",
            ):
                if not isinstance(record[name], list):
                    raise ValueError("cost arrays must be lists")
                record[name] = tuple(record[name])
            record["dtypes"] = tuple(record["dtypes"])
            record["shapes"] = tuple(tuple(shape) for shape in record["shapes"])
            config = cls(**record)
            if config.model_revision != model_revision(Path(config.model_name)):
                raise ValueError("model bytes differ from cost calibration")
            return config
        except (KeyError, TypeError) as exc:
            raise ValueError("missing or unknown local calibration fields") from exc

    def first_copy_gain_ms(self, position: int, reuses: float) -> float:
        """Price a disjoint increment's first L2 copy, including its write."""
        return reuses * (
            self._increment(self.prefix_recompute_ms, position)
            - self._increment(self.prefix_fetch_l2_ms, position)
        ) - self._increment(self.prefix_write_l2_ms, position)

    def retention_gain_ms(
        self, position: int, reuses: float, *, disk_backed: bool
    ) -> float:
        """Price DRAM versus the known alternative, without double counting."""
        alternative = (
            self.prefix_fetch_l2_ms if disk_backed else self.prefix_recompute_ms
        )
        return reuses * (
            self._increment(alternative, position)
            - self._increment(self.prefix_fetch_l1_ms, position)
        )

    @staticmethod
    def _increment(values: tuple[float, ...], position: int) -> float:
        return values[position] - (values[position - 1] if position else 0.0)


class LocalValuePolicy(ProfiledEvictionPolicy, StorePolicy, L2AdapterListener):
    """Share one demand history between native local controller decisions.

    L2 callbacks maintain bounded *advisory* backing knowledge, not ownership.
    A forgotten/unknown backing gets no disk-copy discount. No policy callback
    removes L1 objects, releases locks or declares a skipped store successful.
    Only one L1 and one fixed L2 are supported by the construction boundary.
    """

    def __init__(
        self,
        config: LocalValueConfig,
        event_bus: EventBus,
        *,
        clock: Callable[[], float] = time.monotonic,
        register: bool = True,
    ) -> None:
        """Construct bounded state; the event bus owns subscriber shutdown."""
        self._costs = config
        self._loaded_at = clock()
        self._backed: OrderedDict[ObjectKey, None] = OrderedDict()
        self._layout_compatible = True
        super().__init__(config, event_bus, clock=clock, register=register)

    @property
    def config_ttl_s(self) -> float:
        """Maximum age of this process's frozen cost snapshot."""
        return self._costs.config_ttl_s

    def select_store_targets(
        self, keys: list[ObjectKey], adapters: list[L2AdapterDescriptor]
    ) -> dict[int, list[ObjectKey]]:
        """Skip known negative-value L2 writes; missing/stale data uses default.

        This executes after L1 materialization, and cannot avoid a D2H copy.
        Known suffixes require a selected or already-backed parent. Unknown
        identities retain the original write-through behavior.
        """
        if (
            len(adapters) != 1
            or self._config.mode == "observe"
            or not self._costs_fresh()
        ):
            return {adapter.index: list(keys) for adapter in adapters}
        with self._lock:
            selected: list[ObjectKey] = []
            selected_hashes: set[bytes] = set()
            ordered = sorted(
                keys,
                key=lambda key: (
                    self._history[key.chunk_hash].position if self._known(key) else -1
                ),
            )
            for key in ordered:
                if not self._known(key):
                    selected.append(key)
                    continue
                item = self._history[key.chunk_hash]
                age = len(self._stack) - self._stack.bisect_right(item.ordinal)
                penalty = (
                    1.0
                    if self._config.mode == "no-distance"
                    else 1 + max(age, item.reuse_distance) / max(1, len(self._resident))
                )
                reuses = (
                    1.0
                    if self._config.mode == "no-frequency"
                    else self._frequency_weight(item)
                ) / penalty
                gain = self._costs.first_copy_gain_ms(item.position, reuses)
                if self._config.mode == "no-cost":
                    gain = reuses - 1.0
                elif self._config.mode == "two-hit":
                    gain = item.frequency - 1.5
                parent_backed = self._ancestors_available(key, item, selected_hashes)
                if gain > 0 and parent_backed:
                    selected.append(key)
                    selected_hashes.add(key.chunk_hash)
        return {adapters[0].index: selected} if selected else {}

    def select_l1_deletions(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        """Keep L1 deletion entirely with its native eviction controller."""
        return []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        """Record completed copies, never proposed or failed writes."""
        with self._lock:
            for key in keys:
                self._backed[key] = None
                self._backed.move_to_end(key)
                while len(self._backed) > self._config.history_keys:
                    self._backed.popitem(last=False)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        """Internal storage reads are not additional demand observations."""

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        """Forget backing after the owning adapter acknowledges deletion."""
        with self._lock:
            for key in keys:
                self._backed.pop(key, None)

    def shutdown(self) -> None:
        """Clear derived metadata only after native producers/bus stop."""
        super().shutdown()
        with self._lock:
            self._backed.clear()

    def _known(self, key: ObjectKey) -> bool:
        return (
            self._layout_compatible
            and key.model_name == self._config.model_name
            and not key.cache_salt
            and not key.identity.tags
            and key.kv_rank == ObjectKey.ComputeKVRank(1, 0, 1, 0)
            and key.object_group_id == 0
            and key.chunk_hash in self._history
        )

    def _retention_cost_ms(self, key: ObjectKey, item: _History) -> float:
        return self._costs.retention_gain_ms(
            item.position, 1.0, disk_backed=key in self._backed
        )

    def _frequency_weight(self, item: _History) -> float:
        now = self._clock()
        decayed = item.frequency * math.exp2(
            -max(0.0, now - item.last_access) / self._costs.half_life_s
        )
        return (
            max(0.0, decayed - 1.0)
            * self._costs.horizon_s
            / max(self._costs.horizon_s, now - item.first_access)
        )

    def _next_frequency(self, old: _History | None, now: float) -> float:
        if old is None:
            return 1.0
        if self._config.mode == "two-hit":
            return old.frequency + 1.0
        return 1.0 + old.frequency * math.exp2(
            -max(0.0, now - old.last_access) / self._costs.half_life_s
        )

    def _ranking_enabled(self) -> bool:
        return self._config.mode not in {"observe", "two-hit"} and self._costs_fresh()

    def _rank_snapshot(self, keys: list[ObjectKey], profiled: set[ObjectKey]) -> bool:
        return (
            len(keys) == len(profiled)
            and all(self._known(key) for key in keys)
            and self._ranking_enabled()
        )

    def _costs_fresh(self) -> bool:
        return (
            self._layout_compatible
            and 0 <= self._clock() - self._loaded_at <= self._costs.config_ttl_s
        )

    def _observe(self, event: Event) -> None:
        if event.metadata.get("model_name") != self._config.model_name:
            return
        if any(
            field not in event.metadata
            for field in ("cache_salt", "world_size", "tags")
        ):
            with self._lock:
                self._layout_compatible = False
            return
        if (
            event.metadata["cache_salt"] != ""
            or event.metadata["world_size"] != 1
            or event.metadata["tags"] != {}
        ):
            return
        if event.metadata.get("dtypes") != list(
            self._costs.dtypes
        ) or event.metadata.get("shapes") != [
            list(shape) for shape in self._costs.shapes
        ]:
            with self._lock:
                self._layout_compatible = False
            return
        super()._observe(event)

    def _ancestors_available(
        self, key: ObjectKey, item: _History, selected: set[bytes]
    ) -> bool:
        parent = item.parent
        remaining = item.position
        while parent is not None and remaining > 0:
            if (
                parent not in selected
                and replace(key, identity=replace(key.identity, chunk_hash=parent))
                not in self._backed
            ):
                return False
            ancestor = self._history.get(parent)
            if ancestor is None or ancestor.position != remaining - 1:
                return False
            parent = ancestor.parent
            remaining -= 1
        return parent is None and remaining == 0
