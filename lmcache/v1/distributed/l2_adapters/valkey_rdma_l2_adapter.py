# SPDX-License-Identifier: Apache-2.0
"""
Valkey L2 adapter that moves KV cache chunks over RDMA.

The ``valkey`` adapter with its transport swapped: every store and load is an
RDMA transfer between the L1 arena and a Valkey node running the
valkey-large-object module, through ``valkey-glide-sync``'s RDMA API. Batching,
event fds, locking, eviction accounting, and wire keys are inherited from
``valkey_l2_adapter.py``; the bytes move in
``lmcache/v1/storage_backend/valkey/rdma_worker_pool.py``.

Registered as ``"valkey_rdma"``. The factory requires the L1 memory
descriptor, because that arena is what every worker registers.

Example ``--l2-adapter``::

    {"type": "valkey_rdma",
     "startup_nodes": "kv.internal:6379",
     "num_workers": 8,
     "ttl_seconds": 3600,
     "rdma_provider": "efa-direct"}
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.valkey_l2_adapter import (
    ValkeyL2Adapter,
    ValkeyL2AdapterConfig,
)
from lmcache.v1.storage_backend.valkey.rdma_worker_pool import (
    RDMA_PROVIDERS,
    ValkeyRdmaWorkerPool,
)
from lmcache.v1.storage_backend.valkey.worker_pool import ValkeyWorkerPool

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

logger = init_logger(__name__)

#: The adapter's registered type name.
ADAPTER_TYPE: str = "valkey_rdma"

DEFAULT_RDMA_PROVIDER: str = "efa-direct"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ValkeyRdmaL2AdapterConfig(ValkeyL2AdapterConfig):
    """
    Config for the Valkey RDMA L2 adapter.

    Every ``ValkeyL2AdapterConfig`` field applies, plus:
      * ``rdma_provider``: ``"efa-direct"`` (default) for EFA hardware, or
        ``"tcp"`` for libfabric's software provider (development and tests).
      * ``rdma_interface``: Fabric domain to pin to on a multi-card host.

    Two inherited fields differ here: ``ttl_seconds`` is applied with a
    separate ``EXPIRE`` after each store, and ``request_timeout`` does not
    bound a transfer, since a posted RDMA write cannot be called off.
    """

    def __init__(
        self,
        *args: Any,
        rdma_provider: str = DEFAULT_RDMA_PROVIDER,
        rdma_interface: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the config, validating the RDMA fields.

        Args:
            *args: Positional arguments of ``ValkeyL2AdapterConfig``.
            rdma_provider: ``"efa-direct"`` or ``"tcp"``.
            rdma_interface: Optional fabric domain; must not be empty if set.
            **kwargs: Keyword arguments of ``ValkeyL2AdapterConfig``.

        Raises:
            ValueError: If any validation fails.
        """
        super().__init__(*args, **kwargs)
        if rdma_provider not in RDMA_PROVIDERS:
            raise ValueError(
                f"rdma_provider must be one of {RDMA_PROVIDERS}, got {rdma_provider!r}"
            )
        if rdma_interface is not None and not rdma_interface:
            raise ValueError("rdma_interface must not be empty if set")
        self.rdma_provider: str = rdma_provider
        self.rdma_interface: Optional[str] = rdma_interface

    @classmethod
    def help(cls) -> str:
        """Return human-readable help describing the JSON config fields."""
        inherited = super().help().split("\n", 1)[1]
        return (
            "Valkey RDMA L2 adapter config fields (needs a Valkey node running "
            "the valkey-large-object module and valkey-glide-sync with RDMA "
            "support):\n"
            f"{inherited}\n"
            "- rdma_provider (str, default 'efa-direct'): 'efa-direct' for "
            "EFA hardware, or 'tcp' for libfabric's software provider "
            "(development and tests only).\n"
            "- rdma_interface (str, optional): fabric domain to pin to on a "
            "host with more than one card; omit to let the provider choose."
        )

    @classmethod
    def _kwargs_from_dict(cls, d: dict) -> dict[str, Any]:
        """Parse the inherited fields, then the RDMA ones.

        Args:
            d: Parsed JSON dict.

        Returns:
            Constructor kwargs for :meth:`from_dict`.

        Raises:
            ValueError: If a field has the wrong type.
        """
        kwargs = super()._kwargs_from_dict(d)

        provider_raw = d.get("rdma_provider", DEFAULT_RDMA_PROVIDER)
        if not isinstance(provider_raw, str):
            raise ValueError("rdma_provider must be a string")
        kwargs["rdma_provider"] = provider_raw

        interface_raw = d.get("rdma_interface")
        if interface_raw is not None and not isinstance(interface_raw, str):
            raise ValueError("rdma_interface must be a string or omitted")
        kwargs["rdma_interface"] = interface_raw

        return kwargs


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ValkeyRdmaL2Adapter(ValkeyL2Adapter):
    """
    ``ValkeyL2Adapter`` whose bytes move over RDMA.

    Everything above the transport is inherited; the pool registers the L1
    arena on each worker and transfers windows of it.
    """

    def __init__(
        self, config: ValkeyRdmaL2AdapterConfig, l1_memory_desc: "L1MemoryDesc"
    ) -> None:
        """Initialize the adapter and warm up worker clients.

        Args:
            config: The adapter configuration.
            l1_memory_desc: The L1 arena every object must live in.

        Raises:
            RuntimeError: If the glide client cannot do RDMA on this machine
                or a connection fails.
        """
        self._rdma_config: ValkeyRdmaL2AdapterConfig = config
        self._l1_memory_desc = l1_memory_desc
        super().__init__(config)

    def report_status(self) -> dict[str, Any]:
        """Return the inherited status snapshot plus the RDMA settings."""
        status = super().report_status()
        status["type"] = ADAPTER_TYPE
        status["rdma_provider"] = self._rdma_config.rdma_provider
        status["rdma_interface"] = self._rdma_config.rdma_interface
        status["l1_arena_bytes"] = self._l1_memory_desc.size
        return status

    def _create_pool(self, config: ValkeyL2AdapterConfig) -> ValkeyWorkerPool:
        """Build a ``ValkeyRdmaWorkerPool`` over the L1 arena."""
        return ValkeyRdmaWorkerPool(
            l1_base=self._l1_memory_desc.ptr,
            l1_size=self._l1_memory_desc.size,
            rdma_provider=self._rdma_config.rdma_provider,
            rdma_interface=self._rdma_config.rdma_interface,
            **self._pool_kwargs(config),
        )


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------


def _create_valkey_rdma_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Factory entry-point registered under type name ``"valkey_rdma"``.

    Unlike the ``valkey`` factory this one requires ``l1_memory_desc``: every
    transfer is addressed relative to the arena it describes.

    Args:
        config: Must be a ``ValkeyRdmaL2AdapterConfig``.
        l1_memory_desc: The L1 arena descriptor from the storage manager.

    Returns:
        A ready ``ValkeyRdmaL2Adapter``.

    Raises:
        TypeError: If ``config`` is not a ``ValkeyRdmaL2AdapterConfig``.
        ValueError: If ``l1_memory_desc`` is missing or describes an empty
            arena.
    """
    if not isinstance(config, ValkeyRdmaL2AdapterConfig):
        raise TypeError(
            "_create_valkey_rdma_l2_adapter expected ValkeyRdmaL2AdapterConfig, "
            f"got {type(config).__name__}"
        )
    if l1_memory_desc is None:
        raise ValueError(
            "The valkey_rdma L2 adapter needs the L1 memory descriptor, but none "
            "was provided; the L1 memory manager in use does not expose its "
            "arena, so RDMA cannot address it."
        )
    if l1_memory_desc.ptr == 0 or l1_memory_desc.size <= 0:
        raise ValueError(
            "The valkey_rdma L2 adapter was given an invalid L1 memory descriptor "
            f"(ptr={l1_memory_desc.ptr:#x}, size={l1_memory_desc.size}); RDMA "
            "cannot address an arena that does not exist."
        )
    return ValkeyRdmaL2Adapter(config, l1_memory_desc)


register_l2_adapter_type(ADAPTER_TYPE, ValkeyRdmaL2AdapterConfig)
register_l2_adapter_factory(ADAPTER_TYPE, _create_valkey_rdma_l2_adapter)
