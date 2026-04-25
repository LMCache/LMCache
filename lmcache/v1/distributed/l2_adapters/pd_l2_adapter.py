# SPDX-License-Identifier: Apache-2.0
"""
PD (Prefill-Decode) L2 adapter config and implementation.

Stores the configuration needed to connect a sender (prefill) node to a
receiver (decode) node via a staging buffer and a transfer channel (NIXL or
mock_memory).
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, Optional
import os
import threading

if TYPE_CHECKING:
    from lmcache.native_storage_ops import Bitmap
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class PdL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for the PD (Prefill-Decode) L2 adapter.

    Fields:
    - role: 'sender' (prefill) or 'receiver' (decode).
    - peer_host: hostname or IP of the remote peer.
    - peer_init_port: per-TP-rank list of init ports on the peer.
    - peer_alloc_port: per-TP-rank list of alloc ports on the peer.
    - proxy_host: proxy notification host (default: '127.0.0.1').
    - proxy_port: proxy notification port (default: 6688).
    - buffer_size: staging buffer size in bytes (default: 1 GiB).
    - buffer_device: device for the staging buffer, 'cpu' or 'cuda'
      (default: 'cpu').
    - transfer_channel: transfer backend, 'nixl' or 'mock_memory'
      (default: 'nixl').
    - nixl_backends: NIXL transport backends (default: ['tcp']).
    """

    def __init__(
        self,
        role: str,
        peer_host: str,
        peer_init_port: list[int],
        peer_alloc_port: list[int],
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 6688,
        buffer_size: int = 1073741824,
        buffer_device: str = "cpu",
        transfer_channel: str = "nixl",
        nixl_backends: list[str] | None = None,
    ):
        """Initialize PdL2AdapterConfig.

        Args:
            role: 'sender' (prefill node) or 'receiver' (decode node).
            peer_host: Hostname or IP address of the remote peer.
            peer_init_port: Per-TP-rank list of initialization port numbers
                on the peer.
            peer_alloc_port: Per-TP-rank list of alloc port numbers on the
                peer.
            proxy_host: Proxy notification host (default: '127.0.0.1').
            proxy_port: Proxy notification port (default: 6688).
            buffer_size: Staging buffer size in bytes.
            buffer_device: Device for the staging buffer ('cpu' or 'cuda').
            transfer_channel: Transfer backend ('nixl' or 'mock_memory').
            nixl_backends: List of NIXL transport backend names.  Defaults
                to ``['tcp']``.
        """
        super().__init__()
        self.role = role
        self.peer_host = peer_host
        self.peer_init_port = peer_init_port
        self.peer_alloc_port = peer_alloc_port
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.buffer_size = buffer_size
        self.buffer_device = buffer_device
        self.transfer_channel = transfer_channel
        self.nixl_backends = nixl_backends or ["tcp"]

    @classmethod
    def from_dict(cls, d: dict) -> "PdL2AdapterConfig":
        """Build a PdL2AdapterConfig from a dict (e.g. parsed JSON).

        Args:
            d: Adapter spec dict.  Must include ``role``, ``peer_host``,
                ``peer_init_port``, and ``peer_alloc_port``.

        Returns:
            A new PdL2AdapterConfig instance.

        Raises:
            ValueError: If a required field is missing or a value is
                outside the allowed set.
        """
        role = d.get("role")
        if role not in ("sender", "receiver"):
            raise ValueError("role must be 'sender' or 'receiver', got %r" % role)

        peer_host = d.get("peer_host")
        if not isinstance(peer_host, str) or not peer_host:
            raise ValueError("peer_host must be a non-empty string")

        peer_init_port = d.get("peer_init_port")
        if not peer_init_port:
            raise ValueError("peer_init_port is required")

        peer_alloc_port = d.get("peer_alloc_port")
        if not peer_alloc_port:
            raise ValueError("peer_alloc_port is required")

        buffer_device = d.get("buffer_device", "cpu")
        if buffer_device not in ("cpu", "cuda"):
            raise ValueError(
                "buffer_device must be 'cpu' or 'cuda', got %r" % buffer_device
            )

        transfer_channel = d.get("transfer_channel", "nixl")
        if transfer_channel not in ("nixl", "mock_memory"):
            raise ValueError(
                "transfer_channel must be 'nixl' or 'mock_memory', got %r"
                % transfer_channel
            )

        proxy_host = d.get("proxy_host", "127.0.0.1")
        if not isinstance(proxy_host, str) or not proxy_host:
            raise ValueError("proxy_host must be a non-empty string")

        cfg = cls(
            role=role,
            peer_host=peer_host,
            peer_init_port=list(peer_init_port),
            peer_alloc_port=list(peer_alloc_port),
            proxy_host=proxy_host,
            proxy_port=int(d.get("proxy_port", 6688)),
            buffer_size=int(d.get("buffer_size", 1073741824)),
            buffer_device=buffer_device,
            transfer_channel=transfer_channel,
            nixl_backends=list(d.get("nixl_backends", ["tcp"])),
        )
        return cfg

    @classmethod
    def help(cls) -> str:
        """Return a help string describing PdL2AdapterConfig fields.

        Returns:
            A multi-line string listing all config fields with types,
            default values, and whether they are required.
        """
        return (
            "PD L2 adapter config fields:\n"
            "- role (str): 'sender' or 'receiver' (required)\n"
            "- peer_host (str): remote peer hostname or IP (required)\n"
            "- peer_init_port (list[int]): per-TP-rank init ports (required)\n"
            "- peer_alloc_port (list[int]): per-TP-rank alloc ports (required)\n"
            "- proxy_host (str): proxy notification host (default: '127.0.0.1')\n"
            "- proxy_port (int): proxy notification port (default: 6688)\n"
            "- buffer_size (int): staging buffer size in bytes (default: 1073741824)\n"
            "- buffer_device (str): 'cpu' or 'cuda' (default: 'cpu')\n"
            "- transfer_channel (str): 'nixl' or 'mock_memory' (default: 'nixl')\n"
            "- nixl_backends (list[str]): NIXL transport backends (default: ['tcp'])"
        )


register_l2_adapter_type("pd", PdL2AdapterConfig)


class PdL2Adapter(L2AdapterInterface):
    """
    PD (Prefill-Decode) L2 adapter.

    Connects a sender (prefill) node to a receiver (decode) node via a staging
    buffer and transfer channel. This skeleton provides eventfd plumbing and
    type structure; actual I/O logic will be implemented in later PRs.
    """

    def __init__(
        self,
        config: PdL2AdapterConfig,
        l1_memory_desc: Optional["L1MemoryDesc"] = None,
    ):
        """Initialize PdL2Adapter.

        Args:
            config: PdL2AdapterConfig instance with role, peer info, and buffer
                settings.
            l1_memory_desc: Optional L1 memory descriptor (unused in this
                skeleton; required by adapters that register L1 memory with
                external backends in PR 5/7).

        Raises:
            OSError: If eventfd creation fails.
        """
        # PD adapter does not support aggregate eviction (pass 0 for capacity)
        super().__init__(max_capacity_bytes=0)

        logger.info(
            "Initializing PdL2Adapter: role=%s, peer_host=%s",
            config.role,
            config.peer_host,
        )

        self._config = config
        self._role = config.role

        # Create three distinct eventfds for store, lookup, load events
        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK)

        # Stubs for resources initialized in PR 5/7
        self._staging_allocator: Optional[Any] = None
        self._transfer_channel: Optional[Any] = None
        self._zmq_context: Optional[Any] = None

        # Shutdown coordination
        self._stop_flag = threading.Event()

    #####################
    # Event Fd Interface
    #####################

    def get_store_event_fd(self) -> int:
        """Return the event fd for store operation completion signals.

        Returns:
            File descriptor for store events.
        """
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the event fd for lookup and lock operation completion signals.

        Returns:
            File descriptor for lookup and lock events.
        """
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        """Return the event fd for load operation completion signals.

        Returns:
            File descriptor for load events.
        """
        return self._load_efd

    #####################
    # Store Interface
    #####################

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a store task (stub; implementation in PR 4/7).

        Args:
            keys: List of keys to be stored.
            objects: List of memory objects to be stored.

        Returns:
            Task ID for the submitted store task.

        Raises:
            NotImplementedError: Always (placeholder for PR 4/7).
        """
        raise NotImplementedError("PdL2Adapter.submit_store_task: impl in PR 4/7")

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """Pop completed store tasks (stub; implementation in PR 4/7).

        Returns:
            Mapping from task ID to success flag.

        Raises:
            NotImplementedError: Always (placeholder for PR 4/7).
        """
        raise NotImplementedError(
            "PdL2Adapter.pop_completed_store_tasks: impl in PR 4/7"
        )

    #####################
    # Lookup and Lock Interface
    #####################

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
    ) -> L2TaskId:
        """Submit a lookup and lock task (stub; implementation in PR 4/7).

        Args:
            keys: List of keys to look up and lock.

        Returns:
            Task ID for the submitted lookup and lock task.

        Raises:
            NotImplementedError: Always (placeholder for PR 4/7).
        """
        raise NotImplementedError(
            "PdL2Adapter.submit_lookup_and_lock_task: impl in PR 4/7"
        )

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> "Bitmap | None":
        """Query lookup and lock result (stub; implementation in PR 4/7).

        Args:
            task_id: Task ID of the lookup and lock task.

        Returns:
            Bitmap indicating success/failure per key, or None if not complete.

        Raises:
            NotImplementedError: Always (placeholder for PR 4/7).
        """
        raise NotImplementedError(
            "PdL2Adapter.query_lookup_and_lock_result: impl in PR 4/7"
        )

    def submit_unlock(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """Submit an unlock task (stub; implementation in PR 4/7).

        Args:
            keys: List of keys to unlock.

        Raises:
            NotImplementedError: Always (placeholder for PR 4/7).
        """
        raise NotImplementedError("PdL2Adapter.submit_unlock: impl in PR 4/7")

    #####################
    # Load Interface
    #####################

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a load task (stub; implementation in PR 4/7).

        Args:
            keys: List of keys to load.
            objects: List of memory objects as load buffers.

        Returns:
            Task ID for the submitted load task.

        Raises:
            NotImplementedError: Always (placeholder for PR 4/7).
        """
        raise NotImplementedError("PdL2Adapter.submit_load_task: impl in PR 4/7")

    def query_load_result(self, task_id: L2TaskId) -> "Bitmap | None":
        """Query load result (stub; implementation in PR 4/7).

        Args:
            task_id: Task ID of the load task.

        Returns:
            Bitmap indicating success/failure per key, or None if not complete.

        Raises:
            NotImplementedError: Always (placeholder for PR 4/7).
        """
        raise NotImplementedError("PdL2Adapter.query_load_result: impl in PR 4/7")

    #####################
    # Cleanup Interface
    #####################

    def close(self) -> None:
        """Close the adapter and release all resources.

        Closes the three eventfds (store, lookup, load) and sets the stop
        flag for graceful shutdown coordination. Safe to call multiple times.
        """
        # Early return if already closed
        if self._stop_flag.is_set():
            return

        # Set stop flag first to signal any background threads
        self._stop_flag.set()

        # Close eventfds
        os.close(self._store_efd)
        os.close(self._lookup_efd)
        os.close(self._load_efd)


# Factory registration
def _create_pd_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a PdL2Adapter from config.

    Args:
        config: PdL2AdapterConfig instance.
        l1_memory_desc: Optional L1 memory descriptor.

    Returns:
        A new PdL2Adapter instance.
    """
    return PdL2Adapter(config, l1_memory_desc)  # type: ignore[arg-type]


register_l2_adapter_factory("pd", _create_pd_adapter)
