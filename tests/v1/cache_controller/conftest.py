# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures and utilities for cache controller tests."""

# Standard
from unittest.mock import MagicMock, patch
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.controllers.full_sync_tracker import FullSyncTracker
from lmcache.v1.cache_controller.controllers.kv_controller import KVController
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStartRetMsg,
    FullSyncStatusMsg,
    FullSyncStatusRetMsg,
    KVOpEvent,
    RegisterMsg,
)
from lmcache.v1.cache_controller.utils import RegistryTree
from lmcache.v1.config import LMCacheEngineConfig

# Constants
LOCATION = "LocalCPUBackend"


# ============= Config & Key Creation =============


def create_test_config(
    batch_size: int = 100,
    batch_interval_ms: int = 0,
    startup_delay_s: float = 0.0,
    status_poll_interval_s: float = 0.01,
    max_retry_count: int = 3,
    retry_delay_s: float = 0.01,
):
    """Create a test configuration."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="test_instance",
    )
    config.extra_config = {
        "full_sync_batch_size": batch_size,
        "full_sync_batch_interval_ms": batch_interval_ms,
        "full_sync_startup_delay_s": startup_delay_s,
        "full_sync_status_poll_interval_s": status_poll_interval_s,
        "full_sync_max_retry_count": max_retry_count,
        "full_sync_retry_delay_s": retry_delay_s,
    }
    return config


def create_test_key(key_id: int) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey("vllm", "test_model", 3, 123, key_id, torch.bfloat16)


# ============= Mock Classes =============


class MockWorker:
    """Mock LMCacheWorker for testing."""

    def __init__(self):
        self.worker_id = 0
        self.messages = []
        self.req_responses = []
        self._response_index = 0

    def put_msg(self, msg):
        """Record pushed messages."""
        self.messages.append(msg)

    async def async_put_and_wait_msg(self, msg):
        """Return predefined responses for REQ-REP messages."""
        if self._response_index < len(self.req_responses):
            response = self.req_responses[self._response_index]
            self._response_index += 1
            return response
        # Default responses
        if isinstance(msg, FullSyncStartMsg):
            return FullSyncStartRetMsg(sync_id=msg.sync_id, accepted=True)
        elif isinstance(msg, FullSyncStatusMsg):
            return FullSyncStatusRetMsg(
                sync_id=msg.sync_id,
                is_complete=True,
                global_progress=1.0,
                can_exit_freeze=True,
            )
        return None

    def set_responses(self, responses):
        """Set predefined responses."""
        self.req_responses = responses
        self._response_index = 0


class MockLMCacheEngine:
    """Mock LMCacheEngine for testing."""

    def __init__(self):
        self._freeze = False
        self.freeze_calls = []

    def freeze(self, enabled: bool):
        """Record freeze mode changes."""
        self.freeze_calls.append(enabled)
        self._freeze = enabled


class MockLocalCPUBackend:
    """Mock LocalCPUBackend for testing."""

    def __init__(self, keys=None):
        self._keys = keys or []

    def get_keys(self):
        """Return predefined keys."""
        return self._keys

    def __str__(self):
        return LOCATION


# ============= Helper Class =============


class H:
    """Helper class for test utilities."""

    @staticmethod
    def create_tracker(threshold=0.8, timeout=300.0):
        """Create FullSyncTracker with registry."""
        reg = RegistryTree()
        tracker = FullSyncTracker(reg, threshold, timeout)
        return tracker, reg

    @staticmethod
    def create_controller(threshold=0.8, timeout=300.0):
        """Create KVController with test settings."""
        reg = RegistryTree()
        ctrl = KVController(reg, threshold, timeout)
        ctrl.cluster_executor = MagicMock()
        return ctrl

    @staticmethod
    def reg_worker(reg, inst, wid, ip="127.0.0.1", port=8000):
        """Register a worker."""
        reg.register_worker(inst, wid, ip, port, None, MagicMock(), time.time())

    @staticmethod
    async def reg_worker_async(ctrl, inst, wid, ip="127.0.0.1", port=8000):
        """Register worker via registration controller."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock:
            mock.return_value = MagicMock()
            mock.return_value.sent_messages = []
            return await ctrl.register(RegisterMsg(inst, wid, ip, port, None))

    @staticmethod
    def start_msg(inst, wid, sid, keys=100, batches=5):
        """Create FullSyncStartMsg."""
        return FullSyncStartMsg(inst, wid, LOCATION, sid, keys, batches)

    @staticmethod
    def batch_msg(inst, wid, sid, bid, keys):
        """Create FullSyncBatchMsg."""
        return FullSyncBatchMsg(inst, wid, LOCATION, sid, bid, keys)

    @staticmethod
    def end_msg(inst, wid, sid, total):
        """Create FullSyncEndMsg."""
        return FullSyncEndMsg(inst, wid, LOCATION, sid, total)

    @staticmethod
    def batched_op(inst, wid, op, key, seq=0):
        """Create BatchedKVOperationMsg."""
        return BatchedKVOperationMsg(inst, wid, LOCATION, [KVOpEvent(op, key, seq)])

    @staticmethod
    def start_ret(sync_id, accepted=True, error_msg=None):
        """Create FullSyncStartRetMsg."""
        return FullSyncStartRetMsg(
            sync_id=sync_id, accepted=accepted, error_msg=error_msg
        )

    @staticmethod
    def status_ret(sync_id, is_complete=True, progress=1.0, can_exit=True):
        """Create FullSyncStatusRetMsg."""
        return FullSyncStatusRetMsg(
            sync_id=sync_id,
            is_complete=is_complete,
            global_progress=progress,
            can_exit_freeze=can_exit,
        )
