# SPDX-License-Identifier: Apache-2.0
"""The coordinator's metadata document: what an operator set by hand.

Pins and quotas are small, change rarely, and nothing can reconstruct
them -- so they are written the moment they change rather than on a
timer, and in JSON rather than msgpack, because the one person likely to
read this file by hand is the operator who wrote its contents.
"""

# Standard
from collections.abc import Mapping
from typing import cast
import json
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    DurableComponent,
    PersistenceType,
)
from lmcache.v1.mp_coordinator.persistence.store import (
    ArtifactNotFoundError,
    ArtifactStore,
)

logger = init_logger(__name__)

_FORMAT_VERSION = 1


class MetadataFormatError(Exception):
    """A metadata document is malformed or of an unknown version."""


class MetadataPersister:
    """Keeps the metadata document in step with its components.

    The document is the coordinator's own: its shape is checked, but the
    values inside a section are taken as written, and the component that
    owns a section is the only thing that understands it.
    """

    def __init__(self, store: ArtifactStore) -> None:
        """Args:
        store: Where the document lives.
        """
        self._store = store
        self._components: list[DurableComponent] = []

    def register(self, component: DurableComponent) -> None:
        """Add ``component`` to the document; call before :meth:`load`.

        Args:
            component: The state to persist. Its ``persistence_type``
                must be ``METADATA``.

        Raises:
            ValueError: If the component belongs in the checkpoint --
                derived state here would be rewritten on every operator
                change and reloaded ahead of the checkpoint that owns it.
        """
        if component.persistence_type is not PersistenceType.METADATA:
            raise ValueError(
                f"{component.name!r} is {component.persistence_type.value} state; "
                f"the metadata document takes only metadata components"
            )
        self._components.append(component)

    def load(self) -> None:
        """Restore every registered component from the document.

        Call once at startup. A missing or unreadable document leaves the
        components as they are: an operator can re-apply their intent,
        which is better than refusing to boot.
        """
        try:
            with self._store.open_read() as stream:
                document = self._decode(stream.read())
        except ArtifactNotFoundError:
            logger.info("Starting with no metadata: %s", self._store.location())
            return
        except (OSError, MetadataFormatError) as e:
            logger.warning("Ignoring metadata %s: %s", self._store.location(), e)
            return

        sections = cast("Mapping[str, Mapping[str, object]]", document["components"])
        restored = []
        for component in self._components:
            section = sections.get(component.name)
            if section is None:
                continue
            try:
                component.restore(section)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("Ignoring %r section: %s", component.name, e)
                continue
            restored.append(component.name)
        logger.info(
            "Restored %s from %s, captured %.0fs ago",
            ", ".join(restored) or "nothing",
            self._store.location(),
            max(0.0, time.time() - cast("float", document["saved_at"])),
        )

    def save(self) -> None:
        """Replace the document with the components' current state.

        Called by whatever changed one of them, so a ``200`` from an API
        that sets a pin or a quota means the change survives a restart.
        Failures are logged rather than raised.
        """
        document = {
            "version": _FORMAT_VERSION,
            "saved_at": time.time(),
            "components": {c.name: c.capture() for c in self._components},
        }
        try:
            with self._store.open_write() as stream:
                stream.write(json.dumps(document, indent=2).encode())
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to write metadata %s: %s", self._store.location(), e)

    # -- Internals ----------------------------------------------------------------

    def _decode(self, raw: bytes) -> Mapping[str, object]:
        """Parse a document and check its shape.

        Args:
            raw: The stored bytes.

        Returns:
            The document.

        Raises:
            MetadataFormatError: If it is not JSON, is not an object, or
                carries an unsupported version.
        """
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as e:
            raise MetadataFormatError(f"not JSON: {e}") from e
        if not isinstance(document, dict):
            raise MetadataFormatError(f"holds a {type(document).__name__}")
        if document.get("version") != _FORMAT_VERSION:
            raise MetadataFormatError(
                f"unsupported version {document.get('version')!r} "
                f"(this build reads {_FORMAT_VERSION})"
            )
        if not isinstance(document.get("components"), dict):
            raise MetadataFormatError("no components object")
        if not isinstance(document.get("saved_at"), (int, float)):
            raise MetadataFormatError("no saved_at timestamp")
        return cast("Mapping[str, object]", document)
