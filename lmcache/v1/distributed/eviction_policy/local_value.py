# SPDX-License-Identifier: Apache-2.0
"""Opt-in local DRAM retention and L2 admission using measured marginal costs."""

# Future
from __future__ import annotations

# Standard
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import hashlib
import json
import math
import sys
import time

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.l2_adapters.base import CapacitySnapshot

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.profiled import (
    ProfileConfig,
    ProfiledEvictionPolicy,
    _History,
)
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.storage_controllers.store_policy import (
    CapacityAdmission,
    StorePolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.mp_observability.event import Event
from lmcache.v1.mp_observability.event_bus import EventBus

logger = init_logger(__name__)


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
    value_model: Literal["expected_reuses", "reuse_probability"] = "expected_reuses"
    recompute_fraction: float = 0.0
    l2_capacity_policy: Literal["native", "value"] = "native"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.l2_capacity_policy not in ("native", "value"):
            raise ValueError("unsupported l2_capacity_policy")
        if self.l2_capacity_policy == "value" and self.mode != "full":
            raise ValueError("value capacity requires full mode")
        if self.value_model not in ("expected_reuses", "reuse_probability"):
            raise ValueError("unsupported value_model")
        if self.value_model == "reuse_probability" and self.mode == "no-cost":
            raise ValueError("no-cost is a count-domain control; use expected_reuses")
        if (
            type(self.recompute_fraction) not in (int, float)
            or not math.isfinite(self.recompute_fraction)
            or not 0 <= self.recompute_fraction <= 1
        ):
            raise ValueError("recompute_fraction must be finite and in [0,1]")
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
        """Price first L2 copy using the model's reuse weight and write charge.

        ``reuses`` is a count weight in expected_reuses mode, or a probability
        in reuse_probability mode. No capacity opportunity cost is included.
        """
        return self.l2_retention_gain_ms(position, reuses) - self._increment(
            self.prefix_write_l2_ms, position
        )

    def l2_retention_gain_ms(self, position: int, reuses: float) -> float:
        """Price an existing L2 block; its completed write is a sunk cost."""
        return reuses * (
            (1 - self.recompute_fraction)
            * self._increment(self.prefix_recompute_ms, position)
            - self._increment(self.prefix_fetch_l2_ms, position)
        )

    def retention_gain_ms(
        self, position: int, reuses: float, *, disk_backed: bool
    ) -> float:
        """Price DRAM versus the known alternative, without double counting."""
        alternative = (
            self._increment(self.prefix_fetch_l2_ms, position)
            if disk_backed
            else (1 - self.recompute_fraction)
            * self._increment(self.prefix_recompute_ms, position)
        )
        return reuses * (
            alternative - self._increment(self.prefix_fetch_l1_ms, position)
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
        self._store_evaluated = 0
        self._store_admitted = 0
        self._store_fallback = 0
        super().__init__(config, event_bus, clock=clock, register=register)

    @property
    def config_ttl_s(self) -> float:
        """Maximum age of this process's frozen cost snapshot."""
        return self._costs.config_ttl_s

    @property
    def requires_value_capacity(self) -> bool:
        """Enable the exclusive single-writer, uniform-block capacity contract."""
        return self._costs.l2_capacity_policy == "value"

    def store_statistics(self) -> dict[str, int | str]:
        """Return evaluated plan-key counts; repeated re-plans count separately.

        Admitted means proposed, not completed I/O. Fallback keys are excluded
        from evaluated counts. Ranking counts exclude stale/unknown LRU fallback.
        """
        with self._lock:
            return {
                "value_model": self._costs.value_model,
                "l2_capacity_policy": self._costs.l2_capacity_policy,
                "store_evaluated_keys": self._store_evaluated,
                "store_admitted_keys": self._store_admitted,
                "store_rejected_keys": self._store_evaluated - self._store_admitted,
                "store_fallback_keys": self._store_fallback,
                "ranked_decisions": self._ranked_decisions,
            }

    def plan_capacity_admission(
        self, keys: list[ObjectKey], block_bytes: int, snapshot: CapacitySnapshot
    ) -> CapacityAdmission:
        """Price incoming net benefit against eligible resident opportunity cost.

        Uses the owner's inventory, never advisory backing history. Each incoming
        block pays its write cost; resident writes are sunk. At pressure, only a
        strictly better incoming block can replace the lowest-value unlocked
        leaf. Earlier selected blocks and their ancestors cannot be victims.
        Unknown/stale profiles fail closed; the native/default mode is unchanged.

        Args:
            keys: Uniform-size read-locked candidate blocks (duplicates allowed).
            block_bytes: Positive logical payload bytes per block.
            snapshot: Exclusive owner's atomic resident/pending/busy inventory.

        Returns:
            A pure proposal. The controller must delete, refresh, and re-plan
            before storing; this method does not promise deletion or capacity.

        Raises:
            ValueError: The snapshot has no positive finite uniform-block budget.
        """
        if not self.requires_value_capacity:
            return super().plan_capacity_admission(keys, block_bytes, snapshot)
        if (
            type(block_bytes) is not int
            or block_bytes <= 0
            or type(snapshot.capacity_bytes) is not int
            or snapshot.capacity_bytes <= 0
            or type(snapshot.pending_bytes) is not int
            or snapshot.pending_bytes < 0
            or any(size != block_bytes for size in snapshot.resident_sizes.values())
        ):
            raise ValueError("value capacity requires a positive uniform-block budget")
        admitted: list[ObjectKey] = []
        victims: list[ObjectKey] = []
        reason = "active"
        watermark = 0.0
        with self._lock:
            if not self._costs_fresh():
                reason = "stale_or_layout"
            else:
                now = self._clock()
                residents = set(snapshot.resident_sizes)
                remaining = set(residents)
                used = sum(snapshot.resident_sizes.values()) + snapshot.pending_bytes
                scale = max(1, len(residents))
                scored = {
                    candidate: self._l2_gross_value(candidate, now, scale)
                    for candidate in residents
                    if self._known(candidate)
                }
                # Per-decision dependency counts; the adapter remains the owner.
                children: Counter[ObjectKey] = Counter()
                for resident in residents | set(snapshot.unavailable):
                    if self._known(resident):
                        parent = self._capacity_parent(resident)
                        if parent is not None:
                            children[parent] += 1
                ordered = sorted(
                    dict.fromkeys(keys),
                    key=lambda candidate: (
                        self._history[candidate.chunk_hash].position
                        if self._known(candidate)
                        else -1
                    ),
                )
                for candidate in ordered:
                    if (
                        candidate in residents
                        or candidate in snapshot.unavailable
                        or not self._known(candidate)
                        or not self._capacity_ancestors(candidate, remaining)
                    ):
                        continue
                    item = self._history[candidate.chunk_hash]
                    net = self._costs.first_copy_gain_ms(
                        item.position, self._capacity_weight(candidate, now, scale)
                    )
                    if not net > 0:
                        continue
                    victim: ObjectKey | None = None
                    if used + block_bytes > snapshot.capacity_bytes:
                        # Unknown ancestry cannot justify deleting a known parent.
                        if len(scored) != len(residents):
                            continue
                        eligible = [
                            resident
                            for resident in remaining
                            if resident in residents
                            and resident not in snapshot.unavailable
                            and resident not in admitted
                            and not children[resident]
                            and not self._is_capacity_ancestor(resident, candidate)
                        ]
                        if not eligible:
                            continue
                        victim = min(
                            eligible,
                            key=lambda resident: (
                                scored[resident],
                                -self._history[resident.chunk_hash].position,
                                resident.chunk_hash,
                            ),
                        )
                        watermark = max(0.0, scored[victim])
                        if not net > watermark:
                            continue
                        # A single replacement cannot repair pre-existing overage.
                        if used > snapshot.capacity_bytes:
                            continue
                    if victim is not None:
                        remaining.remove(victim)
                        victims.append(victim)
                        used -= block_bytes
                        parent = self._capacity_parent(victim)
                        if parent is not None:
                            children[parent] -= 1
                    remaining.add(candidate)
                    parent = self._capacity_parent(candidate)
                    if parent is not None:
                        children[parent] += 1
                    admitted.append(candidate)
                    used += block_bytes
            if reason == "active":
                known = sum(self._known(key) for key in keys)
                self._store_evaluated += known
                self._store_admitted += len(admitted)
                self._store_fallback += len(keys) - known
            else:
                self._store_fallback += len(keys)
        logger.info(
            "Profiling capacity %s",
            json.dumps(
                {
                    "reason": reason,
                    "candidates": len(keys),
                    "admitted": len(admitted),
                    "victims": len(victims),
                    "watermark_ms": watermark,
                    "resident_bytes": sum(snapshot.resident_sizes.values()),
                    "pending_bytes": snapshot.pending_bytes,
                    "capacity_bytes": snapshot.capacity_bytes,
                }
            ),
        )
        return CapacityAdmission(tuple(admitted), tuple(victims))

    def _l2_gross_value(self, key: ObjectKey, now: float, scale: int) -> float:
        return self._costs.l2_retention_gain_ms(
            self._history[key.chunk_hash].position,
            self._capacity_weight(key, now, scale),
        )

    def _capacity_weight(self, key: ObjectKey, now: float, scale: int) -> float:
        item = self._history[key.chunk_hash]
        age = len(self._stack) - self._stack.bisect_right(item.ordinal)
        weight = self._frequency_weight_at(item, now)
        weight /= 1 + max(age, item.reuse_distance) / scale
        if self._costs.value_model == "reuse_probability":
            weight = -math.expm1(-weight)
        return weight

    def _capacity_parent(self, key: ObjectKey) -> ObjectKey | None:
        parent = self._history[key.chunk_hash].parent
        return (
            replace(key, identity=replace(key.identity, chunk_hash=parent))
            if parent is not None
            else None
        )

    def _capacity_ancestors(self, key: ObjectKey, available: set[ObjectKey]) -> bool:
        item = self._history[key.chunk_hash]
        remaining = item.position
        while item.parent is not None and remaining > 0:
            parent = replace(
                key, identity=replace(key.identity, chunk_hash=item.parent)
            )
            if parent not in available or not self._known(parent):
                return False
            item = self._history[parent.chunk_hash]
            remaining -= 1
            if item.position != remaining:
                return False
        return item.parent is None and remaining == 0

    def _is_capacity_ancestor(self, ancestor: ObjectKey, key: ObjectKey) -> bool:
        item = self._history[key.chunk_hash]
        remaining = item.position
        while item.parent is not None and remaining > 0:
            if item.parent == ancestor.chunk_hash:
                return True
            parent = self._history.get(item.parent)
            if parent is None:
                return True
            item = parent
            remaining -= 1
        return False

    def select_store_targets(
        self, keys: list[ObjectKey], adapters: list[L2AdapterDescriptor]
    ) -> dict[int, list[ObjectKey]]:
        """Skip known negative-value L2 writes; missing/stale data uses default.

        This executes after L1 materialization, and cannot avoid a D2H copy.
        Known suffixes require a selected or already-backed parent. Unknown
        identities retain the original write-through behavior.
        """
        if self.requires_value_capacity:
            # Authoritative prefix/capacity checks run after L1 read reservation.
            return {adapter.index: list(keys) for adapter in adapters}
        if (
            len(adapters) != 1
            or self._config.mode == "observe"
            or not self._costs_fresh()
        ):
            with self._lock:
                self._store_fallback += len(keys)
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
                    self._store_fallback += 1
                    continue
                self._store_evaluated += 1
                item = self._history[key.chunk_hash]
                age = len(self._stack) - self._stack.bisect_right(item.ordinal)
                reuses = self._reuse_weight(
                    item, max(age, item.reuse_distance), len(self._resident)
                )
                gain = self._costs.first_copy_gain_ms(item.position, reuses)
                if self._config.mode == "no-cost":
                    gain = reuses - 1.0
                elif self._config.mode == "two-hit":
                    gain = item.frequency - 1.5
                parent_backed = self._ancestors_available(key, item, selected_hashes)
                if gain > 0 and parent_backed:
                    selected.append(key)
                    selected_hashes.add(key.chunk_hash)
                    self._store_admitted += 1
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
        return self._frequency_weight_at(item, self._clock())

    def _frequency_weight_at(self, item: _History, now: float) -> float:
        if self._costs.value_model == "reuse_probability":
            return self._repeat_intensity(item, now)
        decayed = item.frequency * math.exp2(
            -max(0.0, now - item.last_access) / self._costs.half_life_s
        )
        return (
            max(0.0, decayed - 1.0)
            * self._costs.horizon_s
            / max(self._costs.horizon_s, now - item.first_access)
        )

    def _reuse_weight(self, item: _History, distance: int, residents: int) -> float:
        weight = super()._reuse_weight(item, distance, residents)
        return (
            -math.expm1(-weight)
            if self._costs.value_model == "reuse_probability"
            else weight
        )

    def _repeat_intensity(self, item: _History, now: float) -> float:
        """Estimate lambda*H from decayed repeat events and time exposure.

        The first observation starts exposure; it is not evidence of a repeat.
        A Poisson model and distance thinning are assumptions, not calibration.
        """
        elapsed = max(0.0, now - item.first_access)
        count = item.frequency * math.exp2(
            -max(0.0, now - item.last_access) / self._costs.half_life_s
        )
        repeats = max(0.0, count - math.exp2(-elapsed / self._costs.half_life_s))
        if repeats == 0:
            return 0.0
        if elapsed == 0:
            return math.inf
        decay_age = math.log(2) * (elapsed / self._costs.half_life_s)
        scaled_exposure = -math.expm1(-decay_age)
        # The small-age series avoids relative error in subnormal decay ratios.
        log_exposure = (
            math.log(elapsed) + math.log1p(-decay_age / 2 + decay_age * decay_age / 6)
            if decay_age < 1e-5
            else math.log(scaled_exposure)
            + math.log(self._costs.half_life_s)
            - math.log(math.log(2))
        )
        # Keep both exposure and lambda*H scaled across the finite input domain.
        log_intensity = (
            math.log(repeats) + math.log(self._costs.horizon_s) - log_exposure
        )
        return (
            math.inf
            if log_intensity > math.log(sys.float_info.max)
            else math.exp(log_intensity)
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
