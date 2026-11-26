# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating SerdeProcessor instances from SerdeConfig.

Each serde type registers itself here so it can be referenced by name
in L2 adapter configs:

    {
      "type": "fs",
      "base_path": "/cache",
      "serde": {"type": "fp8", "fp8_dtype": "float8_e4m3fn"}
    }

A serde that needs to know what an object holds (e.g. a recurrent-state
quantizer) registers with ``register_content_aware_serde_factory`` instead;
its factory also receives a ``PartContentResolver``.
"""

# Standard
from typing import TYPE_CHECKING, Callable

# First Party
from lmcache.v1.common.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.group_view import PartContentDesc
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.serde.base import (
        PartContentResolver,
        SerdeConfig,
        SerdeProcessor,
    )

logger = init_logger(__name__)

# name -> factory(kwargs) -> SerdeProcessor.
# Factories receive the type-specific kwargs (everything except "type").
_SERDE_FACTORY_REGISTRY: dict[str, Callable[[dict[str, object]], "SerdeProcessor"]] = {}

# name -> factory(kwargs, resolver) -> SerdeProcessor, for content-aware serdes.
_CONTENT_AWARE_SERDE_FACTORY_REGISTRY: dict[
    str, Callable[[dict[str, object], "PartContentResolver"], "SerdeProcessor"]
] = {}


def _resolve_no_content(_key: "ObjectKey", _part: int) -> "PartContentDesc | None":
    """Resolver for a storage manager built without registration contexts."""
    return None


def register_serde_factory(
    name: str, factory: Callable[[dict[str, object]], "SerdeProcessor"]
) -> None:
    """Register a serde factory under a type name.

    Args:
        name: Serde type name (used in the JSON config ``"type"`` field).
        factory: Callable that takes the type-specific kwargs dict and
            returns a SerdeProcessor instance.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    _check_unregistered(name)
    _SERDE_FACTORY_REGISTRY[name] = factory


def register_content_aware_serde_factory(
    name: str,
    factory: Callable[[dict[str, object], "PartContentResolver"], "SerdeProcessor"],
) -> None:
    """Register a factory for a serde that needs to know what objects hold.

    Shares the type-name space with :func:`register_serde_factory`.

    Args:
        name: Serde type name (used in the JSON config ``"type"`` field).
        factory: Callable that takes the type-specific kwargs dict and a
            resolver of object part contents, and returns a SerdeProcessor
            instance. The resolver returns ``None`` for unknown contents.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    _check_unregistered(name)
    _CONTENT_AWARE_SERDE_FACTORY_REGISTRY[name] = factory


def get_registered_serde_types() -> list[str]:
    """Return the list of registered serde type names."""
    return [*_SERDE_FACTORY_REGISTRY, *_CONTENT_AWARE_SERDE_FACTORY_REGISTRY]


def create_serde_processor(
    config: "SerdeConfig",
    part_content_resolver: "PartContentResolver | None" = None,
) -> "SerdeProcessor":
    """Build a SerdeProcessor from a SerdeConfig.

    Args:
        config: Serde configuration. ``config.type`` must name a registered
            serde; ``config.kwargs`` is forwarded to the factory.
        part_content_resolver: Resolver of object part contents, handed to a
            content-aware serde; ``None`` gives it one that knows nothing.
            Other serdes ignore it.

    Returns:
        A SerdeProcessor instance ready to be passed to a controller.

    Raises:
        ValueError: If ``config.type`` names an unregistered serde.
    """
    factory = _SERDE_FACTORY_REGISTRY.get(config.type)
    if factory is not None:
        return factory(config.kwargs)
    content_aware_factory = _CONTENT_AWARE_SERDE_FACTORY_REGISTRY.get(config.type)
    if content_aware_factory is None:
        known = ", ".join(sorted(get_registered_serde_types())) or "(none)"
        raise ValueError(f"Unknown serde type {config.type!r}. Registered: {known}")
    return content_aware_factory(
        config.kwargs, part_content_resolver or _resolve_no_content
    )


def _check_unregistered(name: str) -> None:
    """Raise ``ValueError`` if ``name`` names a registered serde of either kind."""
    if name in _SERDE_FACTORY_REGISTRY or name in _CONTENT_AWARE_SERDE_FACTORY_REGISTRY:
        raise ValueError(f"Serde type already registered: {name!r}")
