# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Optional

# Third Party
import zmq.asyncio


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
    ip: str
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
    Root structure: ip -> instance_id -> InstanceNode -> WorkerNode
    """

    def __init__(self):
        # Root level: ip -> instance_id -> InstanceNode
        self.instances: dict[str, dict[str, InstanceNode]] = {}
        # Quick lookup: instance_id -> InstanceNode
        self.instance_id_index: dict[str, InstanceNode] = {}

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
        if ip not in self.instances:
            self.instances[ip] = {}

        if instance_id not in self.instances[ip]:
            instance_node = InstanceNode(instance_id=instance_id, ip=ip)
            self.instances[ip][instance_id] = instance_node
            self.instance_id_index[instance_id] = instance_node
        else:
            instance_node = self.instances[ip][instance_id]

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
        instance_node = self.instance_id_index.get(instance_id)
        if instance_node is None:
            return None

        worker_node = instance_node.remove_worker(worker_id)

        # Clean up empty instance
        if not instance_node.has_workers():
            ip = instance_node.ip
            if ip in self.instances:
                self.instances[ip].pop(instance_id, None)
                # Clean up empty ip entry
                if not self.instances[ip]:
                    del self.instances[ip]
            self.instance_id_index.pop(instance_id, None)

        return worker_node

    def get_worker(self, instance_id: str, worker_id: int) -> Optional[WorkerNode]:
        """Get a specific worker."""
        instance_node = self.instance_id_index.get(instance_id)
        if instance_node is None:
            return None
        return instance_node.get_worker(worker_id)

    def get_instance(self, instance_id: str) -> Optional[InstanceNode]:
        """Get an instance by instance_id."""
        return self.instance_id_index.get(instance_id)

    def get_instance_by_ip(self, ip: str) -> Optional[InstanceNode]:
        """Get an instance by IP address. Returns first instance if multiple exist."""
        ip_instances = self.instances.get(ip)
        if ip_instances:
            return next(iter(ip_instances.values()), None)
        return None

    def get_instances_by_ip(self, ip: str) -> list[InstanceNode]:
        """Get all instances by IP address."""
        ip_instances = self.instances.get(ip)
        if ip_instances:
            return list(ip_instances.values())
        return []

    def get_worker_ids(self, instance_id: str) -> list[int]:
        """Get sorted list of worker IDs for an instance."""
        instance_node = self.instance_id_index.get(instance_id)
        if instance_node is None:
            return []
        return instance_node.get_worker_ids()

    def get_all_worker_infos(self) -> list[WorkerInfo]:
        """Get WorkerInfo for all workers across all instances."""
        result = []
        for ip_instances in self.instances.values():
            for instance_node in ip_instances.values():
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
