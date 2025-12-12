# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

# Third Party
import zmq.asyncio


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
    seq_tracker: dict[str, int] = field(default_factory=dict)  # location -> seq_num
    kv_store: dict[str, set[int]] = field(
        default_factory=dict
    )  # location -> set[chunk_hash]

    def admit_kv(self, location: str, key: int) -> None:
        """Admit a KV chunk to this worker."""
        if location not in self.kv_store:
            self.kv_store[location] = set()
        self.kv_store[location].add(key)

    def evict_kv(self, location: str, key: int) -> bool:
        """Evict a KV chunk from this worker. Returns True if evicted."""
        if location not in self.kv_store or key not in self.kv_store[location]:
            return False
        self.kv_store[location].remove(key)
        if not self.kv_store[location]:
            del self.kv_store[location]
        return True

    def has_kv(self, location: str, key: int) -> bool:
        """Check if a KV chunk exists in this worker."""
        return location in self.kv_store and key in self.kv_store[location]

    def get_kv_keys(self, location: str) -> set[int]:
        """Get all keys for a location."""
        return self.kv_store.get(location, set())

    def clear_kv_store(self) -> None:
        """Clear all KV data for this worker."""
        self.kv_store.clear()

    def get_kv_count(self) -> int:
        """Get total count of KV chunks."""
        return sum(len(keys) for keys in self.kv_store.values())

    def to_worker_info(self, instance_id: str) -> WorkerInfo:
        """Convert to WorkerInfo for backward compatibility."""
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
    """

    instance_id: str
    workers: dict[int, WorkerNode] = field(
        default_factory=dict
    )  # worker_id -> WorkerNode

    def add_worker(self, worker_node: WorkerNode) -> None:
        """Add a worker to this instance."""
        self.workers[worker_node.worker_id] = worker_node

    def remove_worker(self, worker_id: int) -> Optional[WorkerNode]:
        """Remove and return a worker from this instance."""
        return self.workers.pop(worker_id, None)

    def get_worker(self, worker_id: int) -> Optional[WorkerNode]:
        """Get a worker by worker_id."""
        return self.workers.get(worker_id)

    def get_worker_ids(self) -> list[int]:
        """Get sorted list of worker IDs."""
        return sorted(self.workers.keys())

    def has_workers(self) -> bool:
        """Check if instance has any workers."""
        return len(self.workers) > 0

    def get_all_worker_infos(self) -> list[WorkerInfo]:
        """Get WorkerInfo for all workers in this instance."""
        return [
            worker.to_worker_info(self.instance_id) for worker in self.workers.values()
        ]


class RegistryTree:
    """
    Central registry managing the tree structure of instances and workers.
    Structure: instance_id -> InstanceNode -> WorkerNode
    """

    def __init__(self):
        # instance_id -> InstanceNode
        self.instances: dict[str, InstanceNode] = {}

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
        # Get or create instance
        if instance_id not in self.instances:
            instance_node = InstanceNode(instance_id=instance_id)
            self.instances[instance_id] = instance_node
        else:
            instance_node = self.instances[instance_id]

        # Create and add worker
        worker_node = WorkerNode(
            worker_id=worker_id,
            ip=ip,
            port=port,
            peer_init_url=peer_init_url,
            socket=socket,
            registration_time=registration_time,
            last_heartbeat_time=registration_time,
        )
        instance_node.add_worker(worker_node)
        return worker_node

    def deregister_worker(
        self, instance_id: str, worker_id: int
    ) -> Optional[WorkerNode]:
        """Deregister a worker and clean up empty instances."""
        instance_node = self.instances.get(instance_id)
        if instance_node is None:
            return None

        worker_node = instance_node.remove_worker(worker_id)

        # Clean up empty instance
        if not instance_node.has_workers():
            del self.instances[instance_id]

        return worker_node

    def get_worker(self, instance_id: str, worker_id: int) -> Optional[WorkerNode]:
        """Get a specific worker."""
        instance_node = self.instances.get(instance_id)
        if instance_node is None:
            return None
        return instance_node.get_worker(worker_id)

    def get_instance(self, instance_id: str) -> Optional[InstanceNode]:
        """Get an instance by instance_id."""
        return self.instances.get(instance_id)

    def get_instance_by_ip(self, ip: str) -> Optional[InstanceNode]:
        """Get an instance by IP address."""
        for instance_node in self.instances.values():
            for worker_node in instance_node.workers.values():
                if worker_node.ip == ip:
                    return instance_node
        return None

    def get_worker_ids(self, instance_id: str) -> list[int]:
        """Get sorted list of worker IDs for an instance."""
        instance_node = self.instances.get(instance_id)
        if instance_node is None:
            return []
        return instance_node.get_worker_ids()

    def get_all_worker_infos(self) -> list[WorkerInfo]:
        """Get WorkerInfo for all workers across all instances."""
        result = []
        for instance_node in self.instances.values():
            result.extend(instance_node.get_all_worker_infos())
        return result

    def update_heartbeat(
        self, instance_id: str, worker_id: int, timestamp: float
    ) -> bool:
        """Update worker heartbeat timestamp. Returns True if successful."""
        worker_node = self.get_worker(instance_id, worker_id)
        if worker_node is None:
            return False
        worker_node.last_heartbeat_time = timestamp
        return True

    def update_seq_num(
        self, instance_id: str, worker_id: int, location: str, seq_num: int
    ) -> bool:
        """Update sequence number for a worker location. Returns True if successful."""
        worker_node = self.get_worker(instance_id, worker_id)
        if worker_node is None:
            return False
        worker_node.seq_tracker[location] = seq_num
        return True

    def get_seq_num(
        self, instance_id: str, worker_id: int, location: str
    ) -> Optional[int]:
        """Get sequence number for a worker location."""
        worker_node = self.get_worker(instance_id, worker_id)
        if worker_node is None:
            return None
        return worker_node.seq_tracker.get(location)

    # KV store operations
    def admit_kv(
        self, instance_id: str, worker_id: int, location: str, key: int
    ) -> bool:
        """Admit a KV chunk. Returns True if successful."""
        worker_node = self.get_worker(instance_id, worker_id)
        if worker_node is None:
            return False
        worker_node.admit_kv(location, key)
        return True

    def evict_kv(
        self, instance_id: str, worker_id: int, location: str, key: int
    ) -> bool:
        """Evict a KV chunk. Returns True if successful."""
        worker_node = self.get_worker(instance_id, worker_id)
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
        for instance_id, instance_node in self.instances.items():
            # Exclude all workers in the specified instance
            if exclude_instance_id is not None and instance_id == exclude_instance_id:
                continue
            for worker_id, worker_node in instance_node.workers.items():
                for location, keys in worker_node.kv_store.items():
                    if key in keys:
                        return KVChunkInfo(instance_id, worker_id, location)
        return None

    def get_total_kv_count(self) -> int:
        """Get total count of KV chunks across all workers."""
        return sum(
            worker_node.get_kv_count()
            for instance_node in self.instances.values()
            for worker_node in instance_node.workers.values()
        )

    def get_worker_kv_keys(
        self, instance_id: str, worker_id: int, location: str
    ) -> set[int]:
        """Get all KV keys for a specific worker and location."""
        worker_node = self.get_worker(instance_id, worker_id)
        if worker_node is None:
            return set()
        return worker_node.get_kv_keys(location)
