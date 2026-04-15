# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating SerdeProcessor instances from config dicts.

Each serde type registers itself here so it can be referenced by name
in L2 adapter configs:

    {
      "type": "fs",
      "base_path": "/cache",
      "serde": {"type": "fp8", "fp8_dtype": "float8_e4m3fn"}
    }
"""

# Standard
from typing import Callable

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import SerdeProcessor

logger = init_logger(__name__)

# name -> factory(dict) -> SerdeProcessor
_SERDE_FACTORY_REGISTRY: dict[str, Callable[[dict[str, object]], SerdeProcessor]] = {}


def register_serde_factory(
    name: str, factory: Callable[[dict[str, object]], SerdeProcessor]
) -> None:
    """Register a serde factory under a type name.

    Args:
        name: Serde type name (used in the JSON config ``"type"`` field).
        factory: Callable that takes the serde config dict and returns
            a SerdeProcessor instance.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    if name in _SERDE_FACTORY_REGISTRY:
        raise ValueError(f"Serde type already registered: {name!r}")
    _SERDE_FACTORY_REGISTRY[name] = factory


def get_registered_serde_types() -> list[str]:
    """Return the list of registered serde type names."""
    return list(_SERDE_FACTORY_REGISTRY)


def create_serde_processor(config: dict[str, object]) -> SerdeProcessor:
    """Build a SerdeProcessor from a config dict.

    The dict must include a ``"type"`` field naming a registered serde.
    All other keys are forwarded to the type-specific factory.

    Args:
        config: Serde config dict (e.g., ``{"type": "fp8", ...}``).

    Returns:
        A SerdeProcessor instance ready to be passed to a controller.

    Raises:
        ValueError: If ``"type"`` is missing or names an unregistered type.
    """
    serde_type = config.get("type")
    if serde_type is None:
        raise ValueError("Serde config missing 'type' field")
    if not isinstance(serde_type, str):
        actual = type(serde_type).__name__
        raise ValueError(f"Serde 'type' must be a string, got {actual}")
    factory = _SERDE_FACTORY_REGISTRY.get(serde_type)
    if factory is None:
        known = ", ".join(sorted(_SERDE_FACTORY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown serde type {serde_type!r}. Registered: {known}")
    return factory(config)


# ---------------------------------------------------------------------------
# Built-in factories
# ---------------------------------------------------------------------------


def _create_fp8_serde(config: dict[str, object]) -> SerdeProcessor:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.serde.fp8 import (
        Fp8QuantizationDeserializer,
        Fp8QuantizationSerializer,
    )

    dtype_name = str(config.get("fp8_dtype", "float8_e4m3fn"))
    fp8_dtype = getattr(torch, dtype_name, None)
    if fp8_dtype is None:
        raise ValueError(f"Unknown torch dtype: {dtype_name!r}")

    max_workers = int(str(config.get("max_workers", 1)))
    return AsyncSerdeProcessor(
        Fp8QuantizationSerializer(fp8_dtype),
        Fp8QuantizationDeserializer(fp8_dtype),
        max_workers=max_workers,
    )


register_serde_factory("fp8", _create_fp8_serde)
