# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for RegistryTree with fine-grained locking.

Tests verify that the object-based locking mechanism provides:
1. Thread-safe operations on instances, workers, and KV stores
2. Concurrent operations on different instances don't block each other
3. Data consistency under high concurrency
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# First Party
from lmcache.v1.cache_controller.message import BatchedKVOperationMsg, KVOpEvent, OpType
from lmcache.v1.cache_controller.utils import (
    InstanceNode,
    RegistryTree,
    WorkerNode,
)


class TestWorkerNodeLocking:
    """Test WorkerNode's internal locking for kv_store and seq_tracker."""

    def test_concurrent_admit_kv(self):
        """Test concurrent KV admission on the same worker."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        num_threads = 10
        keys_per_thread = 100

        def admit_keys(thread_id):
            for i in range(keys_per_thread):
                key = thread_id * keys_per_thread + i
                msg = BatchedKVOperationMsg(
                    instance_id="test_instance",
                    worker_id=worker.worker_id,
                    location=location,
                    operations=[
                        KVOpEvent(
                            op_type=OpType.ADMIT,
                            key=key,
                            seq_num=0,
                        )
                    ],
                )
                worker.handle_batched_kv_operations(msg)

        threads = [
            threading.Thread(target=admit_keys, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All keys should be admitted
        assert worker.get_kv_count() == num_threads * keys_per_thread

    def test_concurrent_admit_evict_kv(self):
        """Test concurrent admit and evict on the same worker."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        num_threads = 5
        operations_per_thread = 100
        errors = []

        def mixed_operations(thread_id):
            for i in range(operations_per_thread):
                key = i  # Same key range for all threads
                try:
                    if i % 2 == 0:
                        msg = BatchedKVOperationMsg(
                            instance_id="test_instance",
                            worker_id=worker.worker_id,
                            location=location,
                            operations=[
                                KVOpEvent(
                                    op_type=OpType.ADMIT,
                                    key=key,
                                    seq_num=0,
                                )
                            ],
                        )
                    else:
                        msg = BatchedKVOperationMsg(
                            instance_id="test_instance",
                            worker_id=worker.worker_id,
                            location=location,
                            operations=[
                                KVOpEvent(
                                    op_type=OpType.EVICT,
                                    key=key,
                                    seq_num=0,
                                )
                            ],
                        )
                    worker.handle_batched_kv_operations(msg)
                except Exception as e:
                    errors.append("Thread %d error: %s" % (thread_id, e))

        threads = [
            threading.Thread(target=mixed_operations, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_concurrent_seq_num_update(self):
        """Test concurrent sequence number updates."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        num_threads = 10
        updates_per_thread = 100

        def update_seq(thread_id):
            for i in range(updates_per_thread):
                worker.update_seq_num(location, thread_id * updates_per_thread + i)

        threads = [
            threading.Thread(target=update_seq, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final seq_num should be set (exact value depends on thread order)
        final_seq = worker.get_seq_num(location)
        assert final_seq is not None


class TestInstanceNodeLocking:
    """Test InstanceNode's internal locking for workers dict."""

    def test_concurrent_add_workers(self):
        """Test concurrent worker additions to same instance."""
        instance = InstanceNode(instance_id="test_instance")
        num_threads = 10

        def add_worker(worker_id):
            worker = WorkerNode(
                worker_id=worker_id,
                ip="127.0.0.1",
                port=8000 + worker_id,
                peer_init_url=None,
                socket=None,
                registration_time=time.time(),
                last_heartbeat_time=time.time(),
            )
            instance.add_worker(worker)

        threads = [
            threading.Thread(target=add_worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All workers should be added
        assert len(instance.get_worker_ids()) == num_threads

    def test_concurrent_add_remove_workers(self):
        """Test concurrent worker add/remove on same instance."""
        instance = InstanceNode(instance_id="test_instance")
        errors = []

        # Pre-add some workers
        for i in range(50):
            worker = WorkerNode(
                worker_id=i,
                ip="127.0.0.1",
                port=8000 + i,
                peer_init_url=None,
                socket=None,
                registration_time=time.time(),
                last_heartbeat_time=time.time(),
            )
            instance.add_worker(worker)

        def add_workers(start_id):
            for i in range(10):
                try:
                    worker = WorkerNode(
                        worker_id=start_id + i,
                        ip="127.0.0.1",
                        port=9000 + start_id + i,
                        peer_init_url=None,
                        socket=None,
                        registration_time=time.time(),
                        last_heartbeat_time=time.time(),
                    )
                    instance.add_worker(worker)
                except Exception as e:
                    errors.append("Add error: %s" % e)

        def remove_workers(start_id):
            for i in range(10):
                try:
                    instance.remove_worker(start_id + i)
                except Exception as e:
                    errors.append("Remove error: %s" % e)

        threads = []
        # Add workers 100-199
        for i in range(10):
            threads.append(threading.Thread(target=add_workers, args=(100 + i * 10,)))
        # Remove workers 0-49
        for i in range(5):
            threads.append(threading.Thread(target=remove_workers, args=(i * 10,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors


class TestRegistryTreeFineGrainedLocking:
    """Test RegistryTree's fine-grained object-based locking."""

    def test_concurrent_operations_different_instances(self):
        """
        Test that operations on different instances don't block each other.
        This is the key benefit of fine-grained locking.
        """
        registry = RegistryTree()
        num_instances = 5
        workers_per_instance = 20
        errors = []
        timing_results = []

        def operate_on_instance(instance_idx):
            instance_id = "instance_%d" % instance_idx
            start_time = time.time()

            for worker_id in range(workers_per_instance):
                try:
                    # Register worker
                    registry.register_worker(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        ip="192.168.%d.%d" % (instance_idx, worker_id),
                        port=8000 + worker_id,
                        peer_init_url=None,
                        socket=None,
                        registration_time=time.time(),
                    )

                    # KV operations
                    for kv_key in range(10):
                        registry.admit_kv(
                            instance_id, worker_id, "location_%d" % worker_id, kv_key
                        )

                    # Seq num operations
                    registry.update_seq_num(instance_id, worker_id, "loc1", worker_id)

                except Exception as e:
                    errors.append("Instance %d error: %s" % (instance_idx, e))

            elapsed = time.time() - start_time
            timing_results.append((instance_idx, elapsed))

        threads = [
            threading.Thread(target=operate_on_instance, args=(i,))
            for i in range(num_instances)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

        # Verify all data
        for i in range(num_instances):
            instance_id = "instance_%d" % i
            worker_ids = registry.get_worker_ids(instance_id)
            assert len(worker_ids) == workers_per_instance

    def test_concurrent_register_deregister_same_instance(self):
        """Test concurrent register/deregister on the same instance."""
        registry = RegistryTree()
        instance_id = "test_instance"
        num_threads = 10
        operations_per_thread = 50
        errors = []

        def register_deregister(thread_id):
            for i in range(operations_per_thread):
                worker_id = thread_id * 1000 + i
                try:
                    # Register
                    registry.register_worker(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        ip="192.168.1.%d" % thread_id,
                        port=8000 + i,
                        peer_init_url=None,
                        socket=None,
                        registration_time=time.time(),
                    )

                    # Verify registration
                    worker = registry.get_worker(instance_id, worker_id)
                    if worker is None:
                        errors.append(
                            "Thread %d: Worker %d not found after register"
                            % (thread_id, worker_id)
                        )
                        continue

                    # Deregister
                    result = registry.deregister_worker(instance_id, worker_id)
                    if result is None:
                        errors.append(
                            "Thread %d: Deregister failed for worker %d"
                            % (thread_id, worker_id)
                        )

                except Exception as e:
                    errors.append("Thread %d error: %s" % (thread_id, e))

        threads = [
            threading.Thread(target=register_deregister, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_concurrent_kv_operations_same_worker(self):
        """Test concurrent KV operations on the same worker."""
        registry = RegistryTree()
        instance_id = "test_instance"
        worker_id = 0

        registry.register_worker(
            instance_id=instance_id,
            worker_id=worker_id,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        location = "test_location"
        num_threads = 10
        keys_per_thread = 100
        errors = []

        def kv_operations(thread_id):
            for i in range(keys_per_thread):
                key = thread_id * keys_per_thread + i
                try:
                    # Admit
                    result = registry.admit_kv(instance_id, worker_id, location, key)
                    if not result:
                        errors.append(
                            "Thread %d: admit_kv failed for key %d" % (thread_id, key)
                        )
                except Exception as e:
                    errors.append("Thread %d error: %s" % (thread_id, e))

        threads = [
            threading.Thread(target=kv_operations, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

        # Verify total KV count
        total = registry.get_total_kv_count()
        assert total == num_threads * keys_per_thread

    def test_find_kv_concurrent_with_admit(self):
        """Test find_kv while concurrent admit operations are happening."""
        registry = RegistryTree()
        num_instances = 3
        workers_per_instance = 3

        # Setup: register workers
        for i in range(num_instances):
            for w in range(workers_per_instance):
                registry.register_worker(
                    instance_id="instance_%d" % i,
                    worker_id=w,
                    ip="192.168.%d.%d" % (i, w),
                    port=8000 + w,
                    peer_init_url=None,
                    socket=None,
                    registration_time=time.time(),
                )

        errors = []
        found_keys = []

        def admit_keys(instance_idx, worker_id):
            for key in range(100):
                try:
                    registry.admit_kv(
                        "instance_%d" % instance_idx,
                        worker_id,
                        "location_%d" % worker_id,
                        key + instance_idx * 1000,
                    )
                except Exception as e:
                    errors.append("Admit error: %s" % e)

        def find_keys():
            for _ in range(50):
                for key in range(100):
                    try:
                        result = registry.find_kv(key)
                        if result:
                            found_keys.append(key)
                    except Exception as e:
                        errors.append("Find error: %s" % e)
                time.sleep(0.001)

        threads = []
        # Admit threads
        for i in range(num_instances):
            for w in range(workers_per_instance):
                threads.append(threading.Thread(target=admit_keys, args=(i, w)))
        # Find threads
        threads.append(threading.Thread(target=find_keys))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_heartbeat_update_concurrent(self):
        """Test concurrent heartbeat updates."""
        registry = RegistryTree()
        instance_id = "test_instance"
        num_workers = 10

        # Register workers
        for w in range(num_workers):
            registry.register_worker(
                instance_id=instance_id,
                worker_id=w,
                ip="127.0.0.1",
                port=8000 + w,
                peer_init_url=None,
                socket=None,
                registration_time=time.time(),
            )

        errors = []
        num_threads = 20
        updates_per_thread = 100

        def update_heartbeats():
            for _ in range(updates_per_thread):
                for w in range(num_workers):
                    try:
                        result = registry.update_heartbeat(instance_id, w, time.time())
                        if not result:
                            errors.append("Heartbeat update failed for worker %d" % w)
                    except Exception as e:
                        errors.append("Heartbeat error: %s" % e)

        threads = [
            threading.Thread(target=update_heartbeats) for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_get_all_worker_infos_concurrent(self):
        """Test get_all_worker_infos during concurrent modifications."""
        registry = RegistryTree()
        errors = []

        def register_workers():
            for i in range(100):
                try:
                    registry.register_worker(
                        instance_id="instance_%d" % (i % 5),
                        worker_id=i,
                        ip="192.168.1.%d" % i,
                        port=8000 + i,
                        peer_init_url=None,
                        socket=None,
                        registration_time=time.time(),
                    )
                except Exception as e:
                    errors.append("Register error: %s" % e)

        def get_infos():
            for _ in range(50):
                try:
                    infos = registry.get_all_worker_infos()
                    # Just verify it returns a list without error
                    assert isinstance(infos, list)
                except Exception as e:
                    errors.append("Get infos error: %s" % e)
                time.sleep(0.001)

        threads = [
            threading.Thread(target=register_workers),
            threading.Thread(target=get_infos),
            threading.Thread(target=get_infos),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_instance_cleanup_on_last_worker_deregister(self):
        """Test that empty instances are cleaned up correctly."""
        registry = RegistryTree()
        instance_id = "test_instance"
        num_workers = 10

        # Register workers
        for w in range(num_workers):
            registry.register_worker(
                instance_id=instance_id,
                worker_id=w,
                ip="127.0.0.1",
                port=8000 + w,
                peer_init_url=None,
                socket=None,
                registration_time=time.time(),
            )

        # Deregister all workers concurrently
        errors = []

        def deregister(worker_id):
            try:
                registry.deregister_worker(instance_id, worker_id)
            except Exception as e:
                errors.append("Deregister error: %s" % e)

        threads = [
            threading.Thread(target=deregister, args=(w,)) for w in range(num_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

        # Instance should be cleaned up
        assert registry.get_instance(instance_id) is None

    def test_high_contention_stress(self):
        """Stress test with high contention on a single instance."""
        registry = RegistryTree()
        instance_id = "stress_instance"
        errors = []
        num_threads = 30
        operations_per_thread = 100

        def stress_operations(thread_id):
            for i in range(operations_per_thread):
                worker_id = thread_id * operations_per_thread + i
                try:
                    # Register
                    registry.register_worker(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        ip="10.0.0.%d" % (thread_id % 256),
                        port=8000 + (i % 1000),
                        peer_init_url=None,
                        socket=None,
                        registration_time=time.time(),
                    )

                    # KV operations
                    registry.admit_kv(instance_id, worker_id, "loc1", i)
                    registry.update_seq_num(instance_id, worker_id, "loc1", i)

                    # Read operations
                    registry.get_worker(instance_id, worker_id)
                    registry.get_seq_num(instance_id, worker_id, "loc1")

                    # Deregister
                    registry.deregister_worker(instance_id, worker_id)

                except Exception as e:
                    errors.append(
                        "Thread %d iteration %d error: %s" % (thread_id, i, e)
                    )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(stress_operations, i) for i in range(num_threads)
            ]
            for f in futures:
                f.result()

        assert not errors, "Errors occurred (first 10): %s" % errors[:10]

    def test_data_consistency_after_concurrent_ops(self):
        """Verify data consistency after heavy concurrent operations."""
        registry = RegistryTree()
        num_instances = 5
        workers_per_instance = 10
        kv_keys = 50

        # Register all workers
        for inst in range(num_instances):
            for w in range(workers_per_instance):
                registry.register_worker(
                    instance_id="inst_%d" % inst,
                    worker_id=w,
                    ip="10.%d.%d.1" % (inst, w),
                    port=8000,
                    peer_init_url=None,
                    socket=None,
                    registration_time=time.time(),
                )

        errors = []

        def concurrent_operations(inst_idx):
            instance_id = "inst_%d" % inst_idx
            for w in range(workers_per_instance):
                for key in range(kv_keys):
                    try:
                        registry.admit_kv(instance_id, w, "loc", key)
                    except Exception as e:
                        errors.append("Admit error: %s" % e)

        threads = [
            threading.Thread(target=concurrent_operations, args=(i,))
            for i in range(num_instances)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors during concurrent ops: %s" % errors

        # Verify final state
        total_kv = registry.get_total_kv_count()
        expected_kv = num_instances * workers_per_instance * kv_keys
        assert total_kv == expected_kv, "Expected %d KV entries, got %d" % (
            expected_kv,
            total_kv,
        )

        for inst in range(num_instances):
            worker_ids = registry.get_worker_ids("inst_%d" % inst)
            assert len(worker_ids) == workers_per_instance, (
                "Instance %d: expected %d workers, got %d"
                % (
                    inst,
                    workers_per_instance,
                    len(worker_ids),
                )
            )


class TestBatchOperations:
    """Test batch KV operations for performance optimization."""

    def test_batch_admit_kv_basic(self):
        """Test basic batch admit functionality."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        keys = list(range(100))

        worker.batch_admit_kv(location, keys)

        assert worker.get_kv_count() == 100
        for key in keys:
            assert worker.has_kv(location, key)

    def test_batch_evict_kv_basic(self):
        """Test basic batch evict functionality."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        keys = list(range(100))

        # First admit all keys
        worker.batch_admit_kv(location, keys)
        assert worker.get_kv_count() == 100

        # Evict half of them
        evict_keys = list(range(50))
        evicted_count = worker.batch_evict_kv(location, evict_keys)

        assert evicted_count == 50
        assert worker.get_kv_count() == 50

        # Check remaining keys
        for key in range(50, 100):
            assert worker.has_kv(location, key)

    def test_batch_evict_nonexistent_keys(self):
        """Test batch evict with some non-existent keys."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"

        # Admit only keys 0-49
        worker.batch_admit_kv(location, list(range(50)))

        # Try to evict keys 0-99 (half don't exist)
        evicted_count = worker.batch_evict_kv(location, list(range(100)))

        assert evicted_count == 50
        assert worker.get_kv_count() == 0

    def test_registry_batch_operations(self):
        """Test batch operations through RegistryTree."""
        registry = RegistryTree()

        registry.register_worker(
            instance_id="inst_0",
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        location = "test_location"
        keys = list(range(1000))

        # Batch admit
        result = registry.batch_admit_kv("inst_0", 0, location, keys)
        assert result is True
        assert registry.get_total_kv_count() == 1000

        # Batch evict
        evicted = registry.batch_evict_kv("inst_0", 0, location, keys[:500])
        assert evicted == 500
        assert registry.get_total_kv_count() == 500

    def test_batch_operations_nonexistent_worker(self):
        """Test batch operations on non-existent worker."""
        registry = RegistryTree()

        # Batch admit to non-existent worker should return False
        result = registry.batch_admit_kv("inst_0", 0, "loc", [1, 2, 3])
        assert result is False

        # Batch evict from non-existent worker should return 0
        evicted = registry.batch_evict_kv("inst_0", 0, "loc", [1, 2, 3])
        assert evicted == 0

    def test_concurrent_batch_operations(self):
        """Test concurrent batch operations on same worker."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        num_threads = 10
        keys_per_batch = 100
        errors = []

        def batch_admit(thread_id):
            keys = [thread_id * keys_per_batch + i for i in range(keys_per_batch)]
            try:
                worker.batch_admit_kv(location, keys)
            except Exception as e:
                errors.append("Thread %d error: %s" % (thread_id, e))

        threads = [
            threading.Thread(target=batch_admit, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors
        assert worker.get_kv_count() == num_threads * keys_per_batch

    def test_check_and_update_seq_first_batch(self):
        """Test check_and_update_seq for first batch (no existing seq)."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        # First batch: seq 0-99
        is_continuous, expected, gap = worker.check_and_update_seq(location, 0, 99)

        assert is_continuous is True
        assert expected == 0
        assert gap == 0
        assert worker.get_seq_num(location) == 99

    def test_check_and_update_seq_continuous(self):
        """Test check_and_update_seq with continuous sequence."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        # First batch: seq 0-99
        worker.check_and_update_seq(location, 0, 99)

        # Second batch: seq 100-199 (continuous)
        is_continuous, expected, gap = worker.check_and_update_seq(location, 100, 199)

        assert is_continuous is True
        assert expected == 100
        assert gap == 0
        assert worker.get_seq_num(location) == 199

    def test_check_and_update_seq_discontinuous(self):
        """Test check_and_update_seq with discontinuous sequence."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        # First batch: seq 0-99
        worker.check_and_update_seq(location, 0, 99)

        # Second batch: seq 200-299 (gap of 100)
        is_continuous, expected, gap = worker.check_and_update_seq(location, 200, 299)

        assert is_continuous is False
        assert expected == 100
        assert gap == 100
        assert worker.get_seq_num(location) == 299

    def test_registry_batch_with_seq_check(self):
        """Test batch operations with sequence check through RegistryTree."""
        registry = RegistryTree()

        registry.register_worker(
            instance_id="inst_0",
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        location = "test_location"

        # First batch admit with seq check
        operations = [([1, 2, 3], 0, 2)]
        success, seq_result = registry.batch_admit_kv_with_seq_check(
            "inst_0", 0, location, operations
        )
        assert success is True
        assert seq_result[0] is True  # is_continuous
        assert registry.get_total_kv_count() == 3

        # Second batch admit (continuous)
        operations = [([4, 5, 6], 3, 5)]
        success, seq_result = registry.batch_admit_kv_with_seq_check(
            "inst_0", 0, location, operations
        )
        assert success is True
        assert seq_result[0] is True
        assert registry.get_total_kv_count() == 6

        # Third batch with gap
        operations = [([10, 11], 10, 11)]
        success, seq_result = registry.batch_admit_kv_with_seq_check(
            "inst_0", 0, location, operations
        )
        assert success is True
        assert seq_result[0] is False  # discontinuous
        assert seq_result[2] == 4  # gap

    def test_registry_batch_evict_with_seq_check(self):
        """Test batch evict with sequence check through RegistryTree."""
        registry = RegistryTree()

        registry.register_worker(
            instance_id="inst_0",
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        location = "test_location"

        # First admit some keys
        registry.batch_admit_kv("inst_0", 0, location, list(range(100)))

        # Batch evict with seq check
        operations = [(list(range(50)), 0, 49)]
        evicted, seq_result = registry.batch_evict_kv_with_seq_check(
            "inst_0", 0, location, operations
        )
        assert evicted == 50
        assert seq_result[0] is True
        assert registry.get_total_kv_count() == 50
