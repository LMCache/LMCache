# SPDX-License-Identifier: Apache-2.0
"""The contract a piece of coordinator state implements to be persisted.

Only the contract lives here. Where the bytes go, and when they are
written, belong to whatever stores them -- a component knows its own
shape and nothing else.
"""

# Standard
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Protocol, runtime_checkable


class PersistenceType(str, Enum):
    """Which durable artifact a piece of coordinator state belongs in.

    The two differ in cadence and in what can rebuild them.
    ``CHECKPOINT`` state is derived from the cache-event stream, so it
    rides with a periodic checkpoint and is disposable. ``METADATA``
    state is set by an operator and nothing can reconstruct it, so it is
    written the moment it changes.
    """

    CHECKPOINT = "checkpoint"
    METADATA = "metadata"


@runtime_checkable
class DurableComponent(Protocol):
    """Coordinator state that outlives the process by serializing itself.

    An implementation owns its section end to end: it names it, says
    which artifact it belongs in, and is the only code that understands
    the shape in between.
    """

    @property
    def name(self) -> str:
        """Name of this component's section in its artifact."""
        ...

    @property
    def persistence_type(self) -> PersistenceType:
        """Which artifact this component's state rides in."""
        ...

    def capture(self) -> Mapping[str, object]:
        """Return the current state in the form the artifact holds.

        Plain data only: nested dicts, lists and tuples of scalars,
        strings and bytes. Domain objects would make every artifact
        writer know what a section means, and a section's shape is the
        component's business alone.

        Copies, never references into live state. Ingest is quiesced for
        the capture but released before the artifact is encoded, so a
        returned reference would be serialized while the component is
        being mutated -- a torn section, or an iteration error, long
        after this returns.
        """
        ...

    def restore(self, state: Mapping[str, object]) -> None:
        """Replace the current state with a captured one.

        Args:
            state: A :meth:`capture` value, as decoded from the artifact;
                implementations know their own shape.
        """
        ...


@runtime_checkable
class Durability(Protocol):
    """Something that has durable state -- its own, or another object's.

    Structural rather than inherited: holding durable state is a property
    of a class, not of the package it lives in, so a view or controller
    that has none says nothing at all and a class outside both can still
    be captured.
    """

    def get_durable_components(self) -> Sequence[DurableComponent]:
        """Return the state that needs to outlive the process.

        Usually the object itself. One that owns others -- a controller
        with a quota table and an eviction policy -- returns them
        alongside, so a caller collects state without knowing what
        anything is made of. Each component carries the
        ``persistence_type`` that decides which artifact it goes in.
        """
        ...
