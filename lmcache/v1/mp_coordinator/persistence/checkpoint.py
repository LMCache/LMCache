# SPDX-License-Identifier: Apache-2.0
"""Directory checkpointing: load on boot, rewrite on a timer.

Covers only state derived from the event stream; operator intent lives in
``metadata_persister.py``. Best-effort throughout — an unusable checkpoint
starts the coordinator cold and a failed write retries next tick, because
losing any of this costs hit rate, never correctness.

See ``docs/design/v1/mp_coordinator/key_directory.md``.
"""

# Standard
from collections.abc import Iterator, Mapping, Sequence
import asyncio
import itertools

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate, InstanceStreamStats
from lmcache.v1.mp_coordinator.key_directory import DirectorySnapshot, KeyDirectory
from lmcache.v1.mp_coordinator.persistence.snapshot_codec import (
    SnapshotFormatError,
    read_snapshot,
    write_snapshot,
)
from lmcache.v1.mp_coordinator.persistence.store import (
    ArtifactNotFoundError,
    ArtifactStore,
    DurableComponent,
)

logger = init_logger(__name__)


def load_checkpoint(
    store: ArtifactStore,
    directory: KeyDirectory,
    gate: EventGate,
    broadcaster: CacheEventBroadcaster,
    components: Sequence[DurableComponent] = (),
) -> None:
    """Load the stored checkpoint into ``directory``, the ingest gate, and
    the other consumers — or leave them all cold.

    Call once at startup, after ``enable_blend_lookup`` and after the
    broadcaster's consumers are registered.

    Args:
        store: Where the checkpoint is stored.
        directory: The empty directory to populate.
        gate: The unused ingest gate, whose cursors decide whether
            restored L1 placements can still be fenced.
        broadcaster: Fan-out to the other consumers, which the restored
            placements are replayed into.
        components: State that rides with the placements because the replay
            cannot reconstruct it — restored *after* the replay, so it
            replaces whatever that left behind.

    Raises:
        ValueError: If ``directory`` or ``gate`` already holds state.
    """
    try:
        with store.open_read() as f:
            snapshot, cursors, sections = read_snapshot(f)
    except ArtifactNotFoundError:
        logger.info("No coordinator checkpoint at %s; starting cold", store.location())
        return
    except (OSError, SnapshotFormatError) as e:
        logger.warning("Ignoring coordinator checkpoint %s: %s", store.location(), e)
        return
    directory.restore(snapshot)
    gate.restore_cursors(cursors)
    for batch in _replay_batches(snapshot):
        broadcaster.broadcast(batch)
    for component in components:
        if component.name in sections:
            component.restore(sections[component.name])
    stats = directory.stats()
    logger.info(
        "Restored %d keys / %d placements / %d stream cursors%s from checkpoint %s",
        stats.num_keys,
        stats.num_placements,
        len(cursors),
        "".join(f" / {key}" for key in sections),
        store.location(),
    )


async def save_checkpoint(
    store: ArtifactStore,
    directory: KeyDirectory,
    gate: EventGate,
    components: Sequence[DurableComponent] = (),
) -> None:
    """Write the coordinator's restorable state to ``store``.

    Captures on the calling thread, briefly holding each source's lock,
    then encodes and writes off-thread. Write failures are logged rather
    than raised.

    Args:
        store: Where to put the checkpoint.
        directory: The directory to capture.
        gate: The ingest gate, whose cursors are captured alongside.
        components: State captured into the trailing sections.
    """
    # Cursors first: a gate ahead of the directory would drop the gap as
    # duplicates on restore.
    cursors = gate.stats()
    snapshot = directory.snapshot()
    sections = {c.name: c.capture() for c in components}
    try:
        await asyncio.to_thread(_write_snapshot, store, snapshot, cursors, sections)
    except OSError as e:
        logger.warning(
            "Failed to write coordinator checkpoint %s: %s", store.location(), e
        )
        return
    logger.debug("Checkpointed %d keys to %s", len(snapshot.keys), store.location())


# -- Internals ----------------------------------------------------------------


def _replay_batches(snapshot: DirectorySnapshot) -> Iterator[CacheEventBatch]:
    """Yield restored placements as ``STORE`` batches, one per run of
    same-identity placements.

    ``restore`` fills the directory alone, so without this the coordinator
    comes back with a full directory and zero accounted bytes. Order does
    not matter: what the replay rebuilds is per-salt byte totals, and
    anything order-dependent (the eviction LRU) restores itself from its
    own section afterwards.
    """
    placements = [
        (key, placement)
        for key, (_, key_placements) in snapshot.keys.items()
        for placement in key_placements
    ]
    grouped = itertools.groupby(
        placements,
        key=lambda item: (
            item[1].instance_id,
            item[1].incarnation,
            item[1].tier,
            item[1].backend,
            item[1].shared,
        ),
    )
    for seq, (identity, run) in enumerate(grouped, start=1):
        instance_id, incarnation, tier, backend, shared = identity
        yield CacheEventBatch(
            instance_id=instance_id,
            incarnation=incarnation,
            seq=seq,
            event_type=CacheEventType.STORE,
            tier=tier,
            backend=backend,
            shared=shared,
            entries=[
                CacheEventEntry(
                    key=key.to_encoded_object_key(), size_bytes=placement.size_bytes
                )
                for key, placement in run
            ],
        )


def _write_snapshot(
    store: ArtifactStore,
    snapshot: DirectorySnapshot,
    cursors: Mapping[str, InstanceStreamStats],
    sections: Mapping[str, Mapping[str, object]],
) -> None:
    """Encode every capture into a replacement of ``store``'s contents."""
    with store.open_write() as f:
        write_snapshot(snapshot, cursors, sections, f)
