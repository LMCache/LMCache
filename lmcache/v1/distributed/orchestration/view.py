# SPDX-License-Identifier: Apache-2.0
"""Typed cluster view state for KV cache orchestration (see #4291).

The orchestration layer builds a global picture of the cluster by folding the
event stream published by each node. That stream is lossy on purpose. When the
event bus queue is full, ``publish()`` drops the event and only bumps a
counter, so the fold sees a stream that thins out exactly when the cluster is
busy.

Losing an event costs different amounts depending on how the state is folded,
and this module makes that difference part of the type rather than something
each policy author has to rediscover.

A :attr:`FoldKind.CONVERGENT` field assigns a value per key, so a dropped
event leaves the key holding an older value until the next event for that key
repairs it. The error is bounded and self healing, which is why an LRU view
can be folded from a lossy stream safely.

A :attr:`FoldKind.ACCUMULATIVE` field adds and subtracts, so a dropped event
is a permanent offset that nothing in the system corrects. A quota view folded
this way drifts silently, and since the matching action deletes keys, acting
on a drifted total deletes a tenant's cache over quota it is not using.

:class:`View` therefore tracks how many events the bus reported dropping, and
:meth:`View.evaluate` records which fields a trigger read so an accumulative
read taken while the view is degraded can be reported rather than acted on.
"""

# Standard
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "FoldKind",
    "ViewField",
    "View",
    "TriggerVerdict",
    "ViewFieldError",
]


class ViewFieldError(ValueError):
    """Raised when a field is used in a way its fold kind does not allow."""


class FoldKind(Enum):
    """How a view field combines the events folded into it.

    Attributes:
        CONVERGENT: Assignment per key, last writer wins. A dropped event
            leaves a stale value that the next event for the same key
            repairs, so the field tolerates a lossy stream.
        ACCUMULATIVE: Running total per key. A dropped event shifts the total
            permanently, so the field is only trustworthy while no drops have
            occurred since its last reconciliation.
    """

    CONVERGENT = "convergent"
    ACCUMULATIVE = "accumulative"


class ViewField:
    """One named piece of cluster state, folded according to its kind.

    A field is created through :meth:`View.declare` rather than directly, so
    that it stays attached to the drop accounting of its owning view.
    """

    def __init__(self, name: str, fold_kind: FoldKind, owner: "View") -> None:
        """Initialize the field.

        Args:
            name: Identifier used when reporting untrusted reads.
            fold_kind: How events combine into this field.
            owner: The view that owns the drop accounting for this field.
        """
        self._name = name
        self._fold_kind = fold_kind
        self._owner = owner
        self._values: dict[str, float] = {}
        # Drop count at the last point this field is known to have matched
        # reality. A convergent field never consults it.
        self._clean_at_drops = owner.dropped_events_seen

    @property
    def name(self) -> str:
        """Identifier of this field."""
        return self._name

    @property
    def fold_kind(self) -> FoldKind:
        """How events combine into this field."""
        return self._fold_kind

    @property
    def trusted(self) -> bool:
        """Whether the current contents can be acted on.

        A convergent field is always trusted, since a dropped event only makes
        a key look older than it is. An accumulative field is trusted only
        while no events have been dropped since its last reconciliation.
        """
        if self._fold_kind is FoldKind.CONVERGENT:
            return True
        return self._owner.dropped_events_seen == self._clean_at_drops

    def set(self, key: str, value: float) -> None:
        """Assign a value for one key.

        Args:
            key: Key the value belongs to.
            value: New value, replacing any previous one.

        Raises:
            ViewFieldError: If the field is accumulative, where assignment
                would silently discard the running total.
        """
        if self._fold_kind is not FoldKind.CONVERGENT:
            raise ViewFieldError(
                f"field '{self._name}' is accumulative, use add() or reconcile()"
            )
        self._values[key] = value

    def add(self, key: str, delta: float) -> None:
        """Add a signed delta to one key's running total.

        Args:
            key: Key the delta belongs to.
            delta: Amount to add, negative to subtract.

        Raises:
            ViewFieldError: If the field is convergent, where a delta has no
                defined meaning.
        """
        if self._fold_kind is not FoldKind.ACCUMULATIVE:
            raise ViewFieldError(
                f"field '{self._name}' is convergent, use set()"
            )
        self._values[key] = self._values.get(key, 0.0) + delta

    def reconcile(self, absolute_values: Mapping[str, float]) -> None:
        """Replace the contents with values taken from an authoritative source.

        This is how an accumulative field recovers from dropped events. The
        caller supplies absolute values rather than deltas, for example an
        occupancy figure reported by each node, and the field is marked
        trustworthy as of the current drop count.

        Args:
            absolute_values: Complete mapping of key to value. Keys absent
                here are removed.
        """
        self._values = dict(absolute_values)
        self._clean_at_drops = self._owner.dropped_events_seen

    def get(self, key: str) -> float:
        """Return the value for one key, or ``0.0`` when it is unknown.

        Args:
            key: Key to read.

        Returns:
            The folded value, defaulting to ``0.0``.
        """
        return self._values.get(key, 0.0)

    def items(self) -> list[tuple[str, float]]:
        """Return every key and value currently folded into this field.

        Returns:
            Key and value pairs in insertion order.
        """
        return list(self._values.items())


@dataclass(frozen=True)
class TriggerVerdict:
    """Outcome of evaluating a trigger against the view.

    Attributes:
        fired: What the trigger returned.
        untrusted_fields: Names of accumulative fields the trigger read while
            the view was degraded. Empty when every read was trustworthy.
    """

    fired: bool
    untrusted_fields: tuple[str, ...] = field(default=())

    @property
    def trustworthy(self) -> bool:
        """Whether every field the trigger read could be relied on.

        Independent of :attr:`fired`, because a trigger that stays quiet on
        drifted state is no more informative than one that fires on it.
        """
        return not self.untrusted_fields

    @property
    def actionable(self) -> bool:
        """Whether the caller may run the action for this verdict."""
        return self.fired and self.trustworthy


class View:
    """A global picture of the cluster, folded from the node event stream.

    The view owns its fields and the drop accounting they consult. Feed it the
    bus counter with :meth:`observe_dropped_events` as events are folded, then
    evaluate triggers through :meth:`evaluate` so that reads of accumulative
    state taken after a drop are reported instead of acted on.
    """

    def __init__(self) -> None:
        """Initialize an empty view with no observed drops."""
        self._fields: dict[str, ViewField] = {}
        self._dropped_events_seen: int = 0
        self._reads: list[str] = []
        self._recording: bool = False

    @property
    def dropped_events_seen(self) -> int:
        """Total events the bus has reported dropping, as last observed."""
        return self._dropped_events_seen

    @property
    def degraded(self) -> bool:
        """Whether any accumulative field has become untrustworthy."""
        return any(not f.trusted for f in self._fields.values())

    def observe_dropped_events(self, total_dropped: int) -> None:
        """Record the bus drop counter.

        Pass ``EventBus.dropped_events_count`` here whenever events are
        folded. The counter is cumulative, so it never decreases.

        Args:
            total_dropped: Cumulative number of events the bus discarded.

        Raises:
            ValueError: If the counter moves backwards, which would mean two
                different buses are being folded into one view.
        """
        if total_dropped < self._dropped_events_seen:
            raise ValueError(
                "drop counter went backwards, from "
                f"{self._dropped_events_seen} to {total_dropped}"
            )
        self._dropped_events_seen = total_dropped

    def declare(self, name: str, fold_kind: FoldKind) -> ViewField:
        """Create a field on this view.

        Args:
            name: Identifier for the field.
            fold_kind: How events combine into it.

        Returns:
            The newly created field.

        Raises:
            ViewFieldError: If a field of that name already exists.
        """
        if name in self._fields:
            raise ViewFieldError(f"field '{name}' is already declared")
        created = ViewField(name, fold_kind, self)
        self._fields[name] = created
        return created

    def field(self, name: str) -> ViewField:
        """Return a declared field, recording the read while evaluating.

        Args:
            name: Identifier of the field.

        Returns:
            The field.

        Raises:
            ViewFieldError: If no field of that name was declared.
        """
        if name not in self._fields:
            raise ViewFieldError(f"field '{name}' is not declared")
        if self._recording:
            self._reads.append(name)
        return self._fields[name]

    def evaluate(self, trigger: Callable[["View"], bool]) -> TriggerVerdict:
        """Run a trigger and report whether its reads can be acted on.

        Every field the trigger reaches through :meth:`field` is recorded, and
        any accumulative field that has seen a drop since its last
        reconciliation is named in the verdict. A trigger that fires on
        untrusted state is reported rather than suppressed, so the caller can
        log it, reconcile, and evaluate again.

        Args:
            trigger: Predicate over this view.

        Returns:
            The trigger result together with the untrusted fields it read.
        """
        self._reads = []
        self._recording = True
        try:
            fired = trigger(self)
        finally:
            self._recording = False
        untrusted = tuple(
            dict.fromkeys(
                name for name in self._reads if not self._fields[name].trusted
            )
        )
        return TriggerVerdict(fired=fired, untrusted_fields=untrusted)
