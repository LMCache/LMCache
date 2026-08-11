# SPDX-License-Identifier: Apache-2.0
"""Durable operator intent: L2 pins and per-``cache_salt`` quotas.

Unlike the directory checkpoint, nothing can rebuild this — hence a
separate artifact, written whenever it changes, as JSON small enough to
read with ``cat`` when a tenant asks why their cache was evicted.
Components register with the persister and serialize themselves (the
shape ``CacheEventBroadcaster.register_consumer`` uses), so the persister
knows the document and nothing about pins or quotas.

See ``docs/design/v1/mp_coordinator/key_directory.md``.
"""

# Standard
from collections.abc import Mapping
import asyncio
import json
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import PersistenceType
from lmcache.v1.mp_coordinator.persistence.store import (
    ArtifactNotFoundError,
    ArtifactStore,
    DurableComponent,
)

logger = init_logger(__name__)

_FORMAT_VERSION = 1


class MetadataFormatError(Exception):
    """The stored document is malformed or of an unknown version."""


class MetadataPersister:
    """Saves and restores every registered :class:`DurableComponent`.

    Writes are synchronous with the change that caused them, so a pin is
    durable by the time its response returns.

    Args:
        store: Where the document lives.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._components: list[DurableComponent] = []

    def register(self, component: DurableComponent) -> None:
        """Add ``component`` to the document; call before :meth:`load`.

        Args:
            component: The state to persist alongside the rest. Its
                ``persistence_type`` must be ``METADATA``.

        Raises:
            ValueError: If the component belongs in the checkpoint instead.
                Derived state here would be rewritten on every operator
                change and reloaded ahead of the replay that owns it.
        """
        if component.persistence_type is not PersistenceType.METADATA:
            raise ValueError(
                f"{component.name!r} is {component.persistence_type.value} state; "
                f"the metadata document takes only metadata components"
            )
        self._components.append(component)

    def load(self) -> None:
        """Restore every registered component, or leave them empty.

        Call once at startup, before the directory checkpoint replays
        placements: restored pins must be in place before their keys
        enter the eviction LRU. An unusable document is logged and
        skipped, leaving eviction disarmed until the controller re-syncs.
        """
        try:
            with self._store.open_read() as stream:
                sections, saved_at = _decode(stream.read())
        except ArtifactNotFoundError as e:
            # The store's own wording covers both "not configured at all"
            # and "configured but nothing written yet".
            logger.info("Starting with no metadata: %s", e)
            return
        except (OSError, MetadataFormatError) as e:
            logger.warning("Ignoring metadata %s: %s", self._store.location(), e)
            return
        try:
            for component in self._components:
                if component.name in sections:
                    component.restore(sections[component.name])
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Ignoring metadata %s: %s", self._store.location(), e)
            return
        logger.info(
            "Restored %s from %s, captured %.0fs ago; the controller remains "
            "the authority and should re-sync",
            ", ".join(component.name for component in self._components),
            self._store.location(),
            max(0.0, time.time() - saved_at),
        )

    async def save(self) -> None:
        """Write every registered component's current state.

        Call from each handler that changes durable state, after the
        change. Write failures are logged, not raised: a full disk costs
        the change's durability, not the request.
        """
        sections = {
            component.name: component.capture() for component in self._components
        }
        try:
            await asyncio.to_thread(_write, self._store, _encode(sections))
        except OSError as e:
            logger.warning("Failed to write metadata %s: %s", self._store.location(), e)


# -- Internals ----------------------------------------------------------------


def _write(store: ArtifactStore, document: bytes) -> None:
    """Replace ``store``'s artifact with ``document``."""
    with store.open_write() as stream:
        stream.write(document)


def _encode(sections: Mapping[str, Mapping[str, object]]) -> bytes:
    """Serialize the document: version, capture time, then each section."""
    return json.dumps(
        {
            "version": _FORMAT_VERSION,
            "saved_at": time.time(),
            "components": sections,
        },
        indent=2,
    ).encode("utf-8")


def _decode(raw: bytes) -> tuple[Mapping[str, Mapping[str, object]], float]:
    """Parse a document written by :func:`_encode`.

    Args:
        raw: The stored bytes.

    Returns:
        The sections keyed by ``name``, and the capture time.

    Raises:
        MetadataFormatError: If the bytes are not a document of a version
            this build reads.
    """
    try:
        body = json.loads(raw)
        version = body["version"]
        if version != _FORMAT_VERSION:
            raise MetadataFormatError(
                f"unsupported metadata version {version} "
                f"(this build reads {_FORMAT_VERSION})"
            )
        sections = body["components"]
        if not isinstance(sections, dict):
            raise MetadataFormatError("components must be an object")
        return sections, float(body["saved_at"])
    except MetadataFormatError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise MetadataFormatError(f"malformed metadata: {e}") from e
