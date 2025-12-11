# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

# Third Party
import zmq.asyncio

# First Party
from lmcache.v1.cache_controller.locks import FastLockWithTimeout, RWLockWithTimeout


class KVChunkInfo(NamedTuple):
    """
    Represents the location information of a KV chunk in the cluster.
    This class is immutable and can be used as a dictionary key.
    """

    instance_id: str
    worker_id: int
    location: str


@dataclass
class WorkerInfo:
    instance_id: str
    worker_id: int
    ip: str
    port: int
    peer_init_url: Optional[str]
    registration_time: float
    last_heartbeat_time: float


@dataclass
class WorkerNode:
    """
    Represents a single worker with all its associated metadata.
    """

    worker_id: int
    ip: str
    port: int
    peer_init_url: Optional[str]
    socket: Optional[zmq.asyncio.Socket]
    registration_time: float
    last_heartbeat_time: float
    # Guarded by _lock
    seq_tracker: dict[str, int] = field(default_factory=dict)  # location -> seq_num
    kv_store: dict[str, set[int]] = field(
        default_factory=dict
    )  # location -> set[chunk_hash]

    def __post_init__(self):
        # Fast lock with timeout for WorkerNode operations
        self._lock = FastLockWithTimeout()

    def admit_kv(self, location: str, key: int) -> None:
        """Admit a KV chunk to this worker."""
        with self._lock:
            if location not in self.kv_store:
                self.kv_store[location] = set()
            self.kv_store[location].add(key)

    def evict_kv(self, location: str, key: int) -> bool:
        """Evict a KV chunk from this worker. Returns True if evicted."""
        with self._lock:
            if location not in self.kv_store or key not in self.kv_store[location]:
                return False
            self.kv_store[location].remove(key)
            if not self.kv_store[location]:
                del self.kv_store[location]
            return True

    def has_kv(self, location: str, key: int) -> bool:
        """Check if a KV chunk exists in this worker."""
        with self._lock:
            return location in self.kv_store and key in self.kv_store[location]

    def get_kv_keys(self, location: str) -> set[int]:
        """Get all keys for a location."""
        with self._lock:
            keys = self.kv_store.get(location)
            if keys is None:
                return set()
            # Return a shallow copy for thread safety
            return set(keys)

    def clear_kv_store(self) -> None:
        """Clear all KV data for this worker."""
        with self._lock:
            self.kv_store.clear()

    def get_kv_count(self) -> int:
        """Get total count of KV chunks."""
        with self._lock:
            return sum(len(keys) for keys in self.kv_store.values())

    def update_seq_num(self, location: str, seq_num: int) -> None:
        """Update sequence number for a location."""
        with self._lock:
            self.seq_tracker[location] = seq_num

    def get_seq_num(self, location: str) -> Optional[int]:
        """Get sequence number for a location."""
        with self._lock:
            return self.seq_tracker.get(location)

    def to_worker_info(self, instance_id: str) -> WorkerInfo:
        """Convert to WorkerInfo for backward compatibility."""
        with self._lock:
            return WorkerInfo(
                instance_id=instance_id,
                worker_id=self.worker_id,
                ip=self.ip,
                port=self.port,
                peer_init_url=self.peer_init_url,
                registration_time=self.registration_time,
                last_heartbeat_time=self.last_heartbeat_time,
            )


@dataclass
class InstanceNode:
    """
    Represents an instance with all its workers.
    Tree structure: InstanceNode -> WorkerNode
    Each InstanceNode has its own lock for thread-safe worker operations.
    """

    instance_id: str
    # Guarded by _rwlock
    workers: dict[int, WorkerNode] = field(
        default_factory=dict
    )  # worker_id -> WorkerNode

    def __post_init__(self):
        # RW lock for protecting workers dict access
        self._rwlock = RWLockWithTimeout()

    def add_worker(self, worker_node: WorkerNode) -> None:
        """Add a worker to this instance."""
        with self._rwlock.write_lock(timeout=0.1):
            self.workers[worker_node.worker_id] = worker_node

    def remove_worker(self, worker_id: int) -> Optional[WorkerNode]:
        """Remove and return a worker from this instance."""
        with self._rwlock.write_lock(timeout=0.1):
            return self.workers.pop(worker_id, None)

    def get_worker(self, worker_id: int) -> Optional[WorkerNode]:
        """Get a worker by worker_id."""
        with self._rwlock.read_lock(timeout=0.05):
            return self.workers.get(worker_id)

    def get_worker_ids(self) -> list[int]:
        """Get sorted list of worker IDs."""
        with self._rwlock.read_lock(timeout=0.05):
            return sorted(self.workers.keys())

    def has_workers(self) -> bool:
        """Check if instance has any workers."""
        with self._rwlock.read_lock(timeout=0.05):
            return len(self.workers) > 0

    def get_all_worker_infos(self) -> list[WorkerInfo]:
        """Get WorkerInfo for all workers in this instance."""
        with self._rwlock.read_lock(timeout=0.05):
            return [
                worker.to_worker_info(self.instance_id)
                for worker in self.workers.values()
            ]


class RegistryTree:
    """
    Central registry managing the tree structure of instances and workers.
    Structure: instance_id -> InstanceNode -> WorkerNode

    Lock hierarchy (from coarse to fine):
    1. RegistryTree._rwlock: protects instances dict access
    2. InstanceNode._rwlock: protects workers dict access
    3. WorkerNode._lock: protects kv_store and seq_tracker access

    This fine-grained locking allows concurrent operations on different
    instances/workers, improving throughput significantly.
    """

    def __init__(self):
        # Guarded by _rwlock
        # instance_id -> InstanceNode
        self.instances: dict[str, InstanceNode] = {}
        # RW lock only for protecting instances dict access
        self._rwlock = RWLockWithTimeout()

    def _get_or_create_instance(self, instance_id: str) -> InstanceNode:
        """Get or create an instance node. Internal use only."""
        # First try with read lock
        with self._rwlock.read_lock(timeout=0.05):
            instance_node = self.instances.get(instance_id)
            if instance_node is not None:
                return instance_node

        # Need to create, use write lock
        with self._rwlock.write_lock(timeout=0.1):
            # Double-check after acquiring write lock
            instance_node = self.instances.get(instance_id)
            if instance_node is None:
                instance_node = InstanceNode(instance_id=instance_id)
                self.instances[instance_id] = instance_node
            return instance_node

    def _get_instance(self, instance_id: str) -> Optional[InstanceNode]:
        """Get an instance node with read lock. Internal use only."""
        with self._rwlock.read_lock(timeout=0.05):
            return self.instances.get(instance_id)

    def register_worker(
        self,
        instance_id: str,
        worker_id: int,
        ip: str,
        port: int,
        peer_init_url: Optional[str],
        socket: zmq.asyncio.Socket,
        registration_time: float,
    ) -> WorkerNode:
        """Register a new worker, creating instance if needed."""
        # Get or create instance (locks instances dict)
        instance_node = self._get_or_create_instance(instance_id)

        # Create worker node
        worker_node = WorkerNode(
            worker_id=worker_id,
            ip=ip,
            port=port,
            peer_init_url=peer_init_url,
            socket=socket,
            registration_time=registration_time,
            last_heartbeat_time=registration_time,
        )
        # Add worker (locks workers dict in instance_node)
        instance_node.add_worker(worker_node)
        return worker_node

    def deregister_worker(
        self, instance_id: str, worker_id: int
    ) -> Optional[WorkerNode]:
        """Deregister a worker and clean up empty instances."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return None

        # Remove worker (locks workers dict in instance_node)
        worker_node = instance_node.remove_worker(worker_id)

        # Clean up empty instance (need write lock on instances)
        if not instance_node.has_workers():
            # TODO(baoloongmao): Move timeout values to configuration
            with self._rwlock.write_lock(timeout=100):
                # Double-check after acquiring write lock
                if not instance_node.has_workers():
                    del self.instances[instance_id]

        return worker_node

    def get_worker(self, instance_id: str, worker_id: int) -> Optional[WorkerNode]:
        """Get a specific worker."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return None
        return instance_node.get_worker(worker_id)

    def get_instance(self, instance_id: str) -> Optional[InstanceNode]:
        """Get an instance by instance_id."""
        return self._get_instance(instance_id)

    def get_instance_by_ip(self, ip: str) -> Optional[InstanceNode]:
        """Get an instance by IP address. Returns first instance if multiple exist."""
        with self._rwlock.read_lock(timeout=0.05):
            for instance_node in self.instances.values():
                # Check workers with instance's lock
                with instance_node._rwlock.read_lock(timeout=0.05):
                    for worker_node in instance_node.workers.values():
                        if worker_node.ip == ip:
                            return instance_node
            return None

    def get_instances_by_ip(self, ip: str) -> list[InstanceNode]:
        """Get all instances by IP address."""
        with self._rwlock.read_lock(timeout=0.05):
            result = []
            for instance_node in self.instances.values():
                # Check workers with instance's lock
                with instance_node._rwlock.read_lock(timeout=0.05):
                    for worker_node in instance_node.workers.values():
                        if worker_node.ip == ip:
                            result.append(instance_node)
                            break
            return result

    def get_worker_ids(self, instance_id: str) -> list[int]:
        """Get sorted list of worker IDs for an instance."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return []
        return instance_node.get_worker_ids()

    def get_all_worker_infos(self) -> list[WorkerInfo]:
        """Get WorkerInfo for all workers across all instances."""
        with self._rwlock.read_lock(timeout=0.05):
            result = []
            for instance_node in self.instances.values():
                result.extend(instance_node.get_all_worker_infos())
            return result

    def update_heartbeat(
        self, instance_id: str, worker_id: int, timestamp: float
    ) -> bool:
        """Update worker heartbeat timestamp. Returns True if successful."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return False
        worker_node = instance_node.get_worker(worker_id)
        if worker_node is None:
            return False
        worker_node.last_heartbeat_time = timestamp
        return True

    def update_seq_num(
        self, instance_id: str, worker_id: int, location: str, seq_num: int
    ) -> bool:
        """Update sequence number for a worker location. Returns True if successful."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return False
        worker_node = instance_node.get_worker(worker_id)
        if worker_node is None:
            return False
        # update_seq_num uses WorkerNode's internal lock
        worker_node.update_seq_num(location, seq_num)
        return True

    def get_seq_num(
        self, instance_id: str, worker_id: int, location: str
    ) -> Optional[int]:
        """Get sequence number for a worker location."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return None
        worker_node = instance_node.get_worker(worker_id)
        if worker_node is None:
            return None
        return worker_node.get_seq_num(location)

    def admit_kv(
        self, instance_id: str, worker_id: int, location: str, key: int
    ) -> bool:
        """Admit a KV chunk. Returns True if successful."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return False
        worker_node = instance_node.get_worker(worker_id)
        if worker_node is None:
            return False
        # admit_kv uses WorkerNode's internal lock
        worker_node.admit_kv(location, key)
        return True

    def evict_kv(
        self, instance_id: str, worker_id: int, location: str, key: int
    ) -> bool:
        """Evict a KV chunk. Returns True if successful."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return False
        worker_node = instance_node.get_worker(worker_id)
        if worker_node is None:
            return False
        return worker_node.evict_kv(location, key)

    def find_kv(
        self,
        key: int,
        exclude_instance_id: Optional[str] = None,
    ) -> Optional[KVChunkInfo]:
        """
        Find a KV chunk across all workers.

        Args:
            key: The KV chunk key to find.
            exclude_instance_id: Instance ID to exclude
            (all workers in this instance will be excluded).

        Returns: KVChunkInfo if found, None otherwise.
        """
        with self._rwlock.read_lock(timeout=0.1):
            for instance_id, instance_node in self.instances.items():
                # Exclude all workers in the specified instance
                if (
                    exclude_instance_id is not None
                    and instance_id == exclude_instance_id
                ):
                    continue
                for worker_id, worker_node in instance_node.workers.items():
                    for location, keys in worker_node.kv_store.items():
                        if key in keys:
                            return KVChunkInfo(instance_id, worker_id, location)
            return None

    def get_total_kv_count(self) -> int:
        """Get total count of KV chunks across all workers."""
        with self._rwlock.read_lock(timeout=0.1):
            total = 0
            for instance_node in self.instances.values():
                with instance_node._rwlock.read_lock(timeout=0.05):
                    for worker_node in instance_node.workers.values():
                        total += worker_node.get_kv_count()
            return total

    def get_worker_kv_keys(
        self, instance_id: str, worker_id: int, location: str
    ) -> set[int]:
        """Get all KV keys for a specific worker and location."""
        instance_node = self._get_instance(instance_id)
        if instance_node is None:
            return set()
        worker_node = instance_node.get_worker(worker_id)
        if worker_node is None:
            return set()
        return worker_node.get_kv_keys(location)
