# SPDX-License-Identifier: Apache-2.0
"""Configuration and factory for the native NIXL L2 adapter."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any
import re

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory

logger = init_logger(__name__)

_BACKEND_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


class NixlNativeL2AdapterConfig(L2AdapterConfigBase):
    """Configure the built-in native NIXL connector.

    Args:
        backend: NIXL backend plugin name, such as ``POSIX`` or ``OBJ``.
        backend_params: String-to-string parameters forwarded to NIXL.
        num_workers: Number of native connector workers and NIXL agents.
        max_capacity_gb: Capacity used for L2 accounting. Zero disables the
            capacity limit.
    """

    def __init__(
        self,
        backend: str,
        backend_params: dict[str, str],
        num_workers: int = 4,
        max_capacity_gb: float = 0,
    ) -> None:
        self.backend = backend
        self.backend_params = dict(backend_params)
        self.num_workers = num_workers
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NixlNativeL2AdapterConfig":
        """Parse and validate a ``nixl_native`` JSON object.

        Args:
            d: Adapter configuration mapping.

        Returns:
            A validated native NIXL configuration.

        Raises:
            ValueError: If a generic or storage-specific field is invalid.
        """
        backend = d.get("backend")
        if not isinstance(backend, str) or not _BACKEND_PATTERN.fullmatch(backend):
            raise ValueError(
                "backend must be an uppercase NIXL plugin identifier "
                "(for example 'POSIX' or 'OBJ')"
            )

        if "storage_type" in d:
            raise ValueError(
                "storage_type is not configurable; the NIXL backend's "
                "FILE_SEG or OBJ_SEG capability selects the storage strategy"
            )

        raw_params = d.get("backend_params", {})
        if not isinstance(raw_params, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_params.items()
        ):
            raise ValueError("backend_params must be a dict of string key-value pairs")
        backend_params = dict(raw_params)

        num_workers = d.get("num_workers", 4)
        if isinstance(num_workers, bool) or not isinstance(num_workers, int):
            raise ValueError("num_workers must be a positive integer")
        if num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if (
            isinstance(max_capacity_gb, bool)
            or not isinstance(max_capacity_gb, (int, float))
            or max_capacity_gb < 0
        ):
            raise ValueError("max_capacity_gb must be a non-negative number")

        return cls(
            backend=backend,
            backend_params=backend_params,
            num_workers=num_workers,
            max_capacity_gb=float(max_capacity_gb),
        )

    @classmethod
    def help(cls) -> str:
        """Return command-line help for the adapter configuration."""
        return (
            "Native NIXL L2 adapter fields:\n"
            "- backend (str): uppercase NIXL plugin name (required)\n"
            "- backend_params (dict[str, str]): parameters forwarded to NIXL; "
            "FILE_SEG backends require file_path and accept use_direct_io\n"
            "- num_workers (int): worker/agent count (default 4, >0)\n"
            "- max_capacity_gb (float): accounting capacity (default 0)"
        )


def _create_nixl_native_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "L1MemoryDesc | None" = None,
) -> L2AdapterInterface:
    """Create a native NIXL adapter and register the complete L1 arena.

    Args:
        config: Validated :class:`NixlNativeL2AdapterConfig`.
        l1_memory_desc: Base address, size, and alignment of the L1 arena.

    Returns:
        The native connector wrapped as an L2 adapter.

    Raises:
        ValueError: If the L1 descriptor is invalid or configured eviction is
            unsupported by the inferred storage strategy.
        RuntimeError: If the optional C++ extension is unavailable.
    """
    if l1_memory_desc is None:
        raise ValueError("nixl_native requires an L1MemoryDesc")
    if l1_memory_desc.ptr <= 0 or l1_memory_desc.size <= 0:
        raise ValueError("nixl_native requires a non-empty L1 memory arena")
    if l1_memory_desc.align_bytes <= 0:
        raise ValueError("nixl_native requires a positive L1 alignment")

    try:
        # First Party
        from lmcache.lmcache_nixl import LMCacheNixlClient
    except ImportError as exc:
        raise RuntimeError(
            "nixl_native requires the optional C++ extension built with NIXL "
            ">= 1.3 development files. Set BUILD_WITH_NIXL=1, "
            "NIXL_INCLUDE_DIR, and NIXL_LIBRARY_DIR, then reinstall LMCache."
        ) from exc

    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        NativeConnectorL2Adapter,
    )

    if not isinstance(config, NixlNativeL2AdapterConfig):
        raise TypeError("config must be NixlNativeL2AdapterConfig")
    native_client = LMCacheNixlClient(
        backend=config.backend,
        backend_params=config.backend_params,
        num_workers=config.num_workers,
        l1_base=l1_memory_desc.ptr,
        l1_size=l1_memory_desc.size,
        l1_alignment=l1_memory_desc.align_bytes,
    )
    try:
        if config.eviction_config is not None and not native_client.supports_delete:
            raise ValueError(
                f"{native_client.storage_type} storage does not support eviction"
            )
        status: dict[str, Any] = {
            "backend": config.backend,
            "storage_type": native_client.storage_type,
            "num_workers": config.num_workers,
            "supports_query": native_client.supports_query,
            "supports_delete": native_client.supports_delete,
            "supports_direct_io": native_client.supports_direct_io,
            "atomic_publication": native_client.atomic_publication,
        }
        if native_client.storage_type == "FILE":
            status["file_path"] = config.backend_params["file_path"]
            status["use_direct_io"] = native_client.supports_direct_io
    except Exception:
        native_client.close()
        raise
    logger.info(
        "Created native NIXL adapter (backend=%s, storage_type=%s, workers=%d)",
        config.backend,
        native_client.storage_type,
        config.num_workers,
    )
    return NativeConnectorL2Adapter(
        native_client,
        max_capacity_gb=config.max_capacity_gb,
        type_name="nixl_native",
        extra_status=status,
    )


register_l2_adapter_type("nixl_native", NixlNativeL2AdapterConfig)
register_l2_adapter_factory("nixl_native", _create_nixl_native_l2_adapter)
