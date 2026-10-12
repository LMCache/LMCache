# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the persistence tests."""

# Standard
from collections.abc import Mapping, Sequence

# First Party
from lmcache.v1.mp_coordinator.persistence.durable_component import DurableComponent
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock


def capture_consistently(
    quiesce: QuiesceLock,
    components: Sequence[DurableComponent],
    timeout: float = 5.0,
) -> dict[str, Mapping[str, object]]:
    """Read ``components`` with ingest held still.

    Stands in for the checkpoint module, which owns this in production.

    Args:
        quiesce: The lock the ingest path holds while applying.
        components: The state to capture.
        timeout: Seconds to wait for an in-flight batch.

    Returns:
        Each component's ``capture`` keyed by its ``name``.
    """
    with quiesce.quiesced(timeout):
        return {component.name: component.capture() for component in components}
