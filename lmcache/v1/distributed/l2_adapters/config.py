# SPDX-License-Identifier: Apache-2.0

"""
Configuration for L2 adapters.

Supports multiple adapter instances (including multiple instances of the same
adapter type with different configs) via repeatable --l2-adapter <JSON>.
Each JSON object must include "type" (adapter type name) and type-specific keys.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar
import argparse
import json

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

T = TypeVar("T", bound="L2AdapterConfigBase")

# -----------------------------------------------------------------------------
# Registry: adapter type name -> config class
# -----------------------------------------------------------------------------

_L2_ADAPTER_CONFIG_REGISTRY: dict[str, type[L2AdapterConfigBase]] = {}


def register_l2_adapter_type(name: str, config_cls: type[L2AdapterConfigBase]) -> None:
    """
    Register an L2 adapter config class under a type name.

    The type name is used in JSON specs as the "type" field. Each adapter
    module should call this at import time.

    Args:
        name: Adapter type name (e.g. "disk", "redis").
        config_cls: Config class that can parse from dict via from_dict().
    """
    if name in _L2_ADAPTER_CONFIG_REGISTRY:
        raise ValueError(f"L2 adapter type already registered: {name!r}")
    _L2_ADAPTER_CONFIG_REGISTRY[name] = config_cls


def get_registered_l2_adapter_types() -> list[str]:
    """Return the list of registered adapter type names."""
    return list(_L2_ADAPTER_CONFIG_REGISTRY)


def get_type_name_for_config(config: L2AdapterConfigBase) -> str:
    """
    Reverse-lookup the registered type name for a config instance.

    Args:
        config: An L2 adapter config instance.

    Returns:
        The registered type name string (e.g., "mock", "disk").

    Raises:
        ValueError: If the config's class is not registered.
    """
    for name, cls in _L2_ADAPTER_CONFIG_REGISTRY.items():
        if type(config) is cls:
            return name
    raise ValueError(f"Unregistered L2 adapter config type: {type(config).__name__}")


# -----------------------------------------------------------------------------
# Base config class for a single L2 adapter
# -----------------------------------------------------------------------------


class L2AdapterConfigBase(ABC):
    """
    Base class for per-adapter configs.

    Each adapter type (e.g. disk, redis) defines a config class that:
    - Subclasses this base.
    - Implements from_dict() to parse a dict (from JSON) into an instance.
    - Is registered via register_l2_adapter_type("type_name", ConfigClass).
    """

    @classmethod
    @abstractmethod
    def from_dict(cls: type[T], d: dict) -> T:
        """
        Build a config instance from a dict (e.g. from parsed JSON).

        The dict will contain the "type" key used for dispatch; the concrete
        class may ignore it. All other keys are type-specific.

        Args:
            d: Adapter spec dict (must include type-specific keys).

        Returns:
            An instance of the config class.

        Raises:
            ValueError: If required keys are missing or values are invalid.
        """
        ...

    @classmethod
    @abstractmethod
    def help(cls) -> str:
        """
        Return a help string describing the config fields for this adapter type.

        This is used in command-line help to explain the expected JSON format for
        each adapter type.

        Returns:
            A help string describing the config fields for this adapter type.
        """
        ...


### Detailed config classes for each L2 adapter
class MockL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for a mock L2 adapter (for testing).

    Fields:
    - max_size_gb: maximum size of the adapter in GB.
    - mock_bandwidth_gb: simulated bandwidth in GB/sec (for testing load times).
    """

    def __init__(self, max_size_gb: float, mock_bandwidth_gb: float):
        self.max_size_gb = max_size_gb
        self.mock_bandwidth_gb = mock_bandwidth_gb

    @classmethod
    def from_dict(cls, d: dict) -> MockL2AdapterConfig:
        max_size_gb = d.get("max_size_gb")
        if not isinstance(max_size_gb, (int, float)) or max_size_gb <= 0:
            raise ValueError("max_size_gb must be a positive number")

        mock_bandwidth_gb = d.get("mock_bandwidth_gb")
        if not isinstance(mock_bandwidth_gb, (int, float)) or mock_bandwidth_gb <= 0:
            raise ValueError("mock_bandwidth_gb must be a positive number")

        return cls(max_size_gb=max_size_gb, mock_bandwidth_gb=mock_bandwidth_gb)

    @classmethod
    def help(cls) -> str:
        return (
            "Mock L2 adapter config fields:\n"
            "- max_size_gb (float): maximum size of the adapter in GB (required, >0)\n"
            "- mock_bandwidth_gb (float): simulated bandwidth in GB/sec (required, >0)"
        )


register_l2_adapter_type("mock", MockL2AdapterConfig)


_VALID_NIXL_BACKENDS = ("GDS", "GDS_MT", "POSIX", "HF3FS", "OBJ")
_FILE_BACKENDS = ("GDS", "GDS_MT", "POSIX", "HF3FS")


class NixlStoreL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for a Nixl-store-based L2 adapter.

    Fields:
    - backend: Nixl storage backend (GDS, GDS_MT, POSIX, HF3FS, OBJ).
    - backend_params: Backend-specific parameters as a dict of string key-value
      pairs. For file-based backends (GDS, GDS_MT, POSIX, HF3FS), must include
      "file_path". May also include "use_direct_io" (default "false") and other
      backend-specific keys.
    - pool_size: Number of storage descriptors to pre-allocate (must be > 0).
    """

    def __init__(
        self,
        backend: str,
        backend_params: dict[str, str],
        pool_size: int,
    ):
        if backend in _FILE_BACKENDS:
            if "file_path" not in backend_params:
                raise ValueError(
                    f"backend_params must include 'file_path' "
                    f"for file-based backend {backend!r}"
                )
            if "use_direct_io" not in backend_params:
                raise ValueError(
                    f"backend_params must include 'use_direct_io' "
                    f"for file-based backend {backend!r}"
                )
        self.backend = backend
        self.backend_params = backend_params
        self.pool_size = pool_size

    @classmethod
    def from_dict(cls, d: dict) -> NixlStoreL2AdapterConfig:
        backend = d.get("backend")
        if backend not in _VALID_NIXL_BACKENDS:
            raise ValueError(
                f"backend must be one of {_VALID_NIXL_BACKENDS}, got {backend!r}"
            )

        backend_params = d.get("backend_params", {})
        if not isinstance(backend_params, dict):
            raise ValueError("backend_params must be a dict of string key-value pairs")

        pool_size = d.get("pool_size")
        if not isinstance(pool_size, int) or pool_size <= 0:
            raise ValueError("pool_size must be a positive integer")

        return cls(
            backend=backend,
            backend_params=backend_params,
            pool_size=pool_size,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Nixl store L2 adapter config fields:\n"
            "- backend (str): Nixl storage backend, one of "
            f"{_VALID_NIXL_BACKENDS} (required)\n"
            "- backend_params (dict): backend-specific string key-value pairs "
            "(optional, default {}). File-based backends require file_path. "
            "Optional keys include 'use_direct_io' (default 'false') and "
            "'file_size' (int, size in bytes of each storage file slot; "
            "defaults to the L1 page size if not set).\n"
            "- pool_size (int): number of storage descriptors to pre-allocate "
            "(required, >0)"
        )


register_l2_adapter_type("nixl_store", NixlStoreL2AdapterConfig)

# -----------------------------------------------------------------------------
# Main config: list of adapter configs (order = adapter order)
# -----------------------------------------------------------------------------


@dataclass
class L2AdaptersConfig:
    """
    Main config for L2 adapters.

    Holds an ordered list of adapter configs. Each element corresponds to one
    L2 adapter instance (e.g. two disk adapters with different paths appear
    as two entries).
    """

    adapters: list[L2AdapterConfigBase]
    """ Ordered list of adapter configs; one per L2 adapter instance. """


# -----------------------------------------------------------------------------
# Command-line: add args and parse to config
# -----------------------------------------------------------------------------

_L2_ADAPTER_ARG_DEST = "l2_adapter"


def add_l2_adapters_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add L2 adapter configuration arguments to an existing parser.

    Adds a repeatable --l2-adapter <JSON> argument. Each JSON object specifies
    one adapter: it must include "type" (registered adapter type name) and
    type-specific keys. Order of arguments is the order of adapters.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with L2 adapter arguments added.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_l2_adapters_args(parser)
        >>> args = parser.parse_args(["--l2-adapter", '{"type":"disk","path":"/data"}'])
        >>> config = parse_args_to_l2_adapters_config(args)
    """
    group = parser.add_argument_group(
        "L2 Adapters",
        "L2 adapter instances. Each --l2-adapter is a JSON object with 'type' and "
        "type-specific keys. Repeat for multiple adapters.",
    )
    group.add_argument(
        "--l2-adapter",
        dest=_L2_ADAPTER_ARG_DEST,
        action="append",
        default=[],
        type=str,
        metavar="JSON",
        help='Adapter spec as JSON with a "type" field and adapter-specific configs'
        ', e.g. \'{"type":"disk","path":"/data"}\'.'
        "Repeat for multiple adapters."
        "Supported adapters are: ["
        + ", ".join(sorted(get_registered_l2_adapter_types()))
        + "].",
    )
    return parser


def parse_args_to_l2_adapters_config(args: argparse.Namespace) -> L2AdaptersConfig:
    """
    Build L2AdaptersConfig from parsed command-line arguments.

    Expects args to have the attribute added by add_l2_adapters_args (a list
    of JSON strings). Each string is parsed; the "type" field selects the
    config class from the registry, and from_dict() builds the config instance.

    Args:
        args: Parsed arguments (e.g. from parser.parse_args()).

    Returns:
        L2AdaptersConfig with one entry per --l2-adapter argument.

    Raises:
        KeyError: If an adapter "type" is not registered.
        ValueError: If JSON is invalid or a config class raises from_dict().
    """
    raw_list = getattr(args, _L2_ADAPTER_ARG_DEST, None)
    if raw_list is None:
        raw_list = []

    adapter_configs: list[L2AdapterConfigBase] = []
    for i, raw in enumerate(raw_list):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for --l2-adapter #{i + 1}: {e}") from e

        if not isinstance(d, dict):
            raise ValueError(
                f"--l2-adapter #{i + 1}: expected a JSON object, got {type(d).__name__}"
            )

        type_name = d.get("type")
        if type_name is None:
            raise ValueError(f"--l2-adapter #{i + 1}: missing 'type' field")
        if type_name not in _L2_ADAPTER_CONFIG_REGISTRY:
            known = ", ".join(sorted(_L2_ADAPTER_CONFIG_REGISTRY)) or "(none)"
            raise ValueError(
                f"--l2-adapter #{i + 1}: unknown adapter type "
                f"{type_name!r}. Known: {known}"
            )

        config_cls = _L2_ADAPTER_CONFIG_REGISTRY[type_name]
        try:
            adapter_configs.append(config_cls.from_dict(d))
        except (TypeError, ValueError) as e:
            logger.error(
                "Error parsing --l2-adapter #%d (type %r): %s",
                i + 1,
                type_name,
                e,
            )
            logger.error(
                "Adapter config help for %s adapter:\n"
                "---------------------\n"
                "%s\n"
                "---------------------\n\n",
                type_name,
                config_cls.help(),
            )
            raise ValueError(f"--l2-adapter #{i + 1} ({type_name!r}): {e}") from e

    return L2AdaptersConfig(adapters=adapter_configs)
