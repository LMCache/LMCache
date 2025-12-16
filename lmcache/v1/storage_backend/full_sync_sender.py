# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, List, Optional
import asyncio
import random
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.message import (
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStartRetMsg,
    FullSyncStatusMsg,
    FullSyncStatusRetMsg,
)
from lmcache.v1.config import LMCacheEngineConfig

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class FullSyncSender:
    """
    Handles full sync of hot_cache keys to the Controller.

    This class manages the process of sending all keys from the local hot_cache
    to the Controller when a full sync is requested (e.g., after Controller restart).

    The sync process:
    1. Enter freeze mode (prevent allocations)
    2. Add random startup delay (avoid thundering herd)
    3. Send FullSyncStartMsg and wait for confirmation
    4. Send keys in batches via FullSyncBatchMsg
    5. Send FullSyncEndMsg
    6. Poll for completion status
    7. Exit freeze mode when threshold is reached
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        worker: "LMCacheWorker",
        lmcache_engine: "LMCacheEngine",
        local_cpu_backend: "LocalCPUBackend",
    ):
        # Configuration
        self.batch_size = config.get_extra_config_value("full_sync_batch_size", 2000)
        self.batch_interval_ms = config.get_extra_config_value(
            "full_sync_batch_interval_ms", 5
        )
        self.startup_delay_range_s = config.get_extra_config_value(
            "full_sync_startup_delay_s", 5.0
        )
        self.status_poll_interval_s = config.get_extra_config_value(
            "full_sync_status_poll_interval_s", 5.0
        )
        self.max_retry_count = config.get_extra_config_value(
            "full_sync_max_retry_count", 3
        )
        self.retry_delay_s = config.get_extra_config_value(
            "full_sync_retry_delay_s", 1.0
        )

        # Dependencies
        self.worker = worker
        self.lmcache_engine = lmcache_engine
        self.local_cpu_backend = local_cpu_backend
        self.config = config

        # State
        self._is_syncing = False
        self._current_sync_id: Optional[str] = None

    @property
    def instance_id(self) -> str:
        return self.config.lmcache_instance_id

    @property
    def worker_id(self) -> int:
        return self.worker.worker_id

    @property
    def location(self) -> str:
        return str(self.local_cpu_backend)

    def _generate_sync_id(self) -> str:
        """Generate a unique sync session ID"""
        return f"{self.instance_id}_{self.worker_id}_{uuid.uuid4().hex[:8]}"

    def _get_all_hot_cache_keys(self) -> List[int]:
        """Get all chunk hashes from the hot cache"""
        keys = self.local_cpu_backend.get_keys()
        return [key.chunk_hash for key in keys]

    async def _send_sync_start(
        self, sync_id: str, total_keys: int, batch_count: int
    ) -> Optional[FullSyncStartRetMsg]:
        """Send FullSyncStartMsg and wait for confirmation"""
        msg = FullSyncStartMsg(
            instance_id=self.instance_id,
            worker_id=self.worker_id,
            location=self.location,
            sync_id=sync_id,
            total_keys=total_keys,
            batch_count=batch_count,
        )

        try:
            ret_msg = await self.worker.async_put_and_wait_msg(msg)
            if isinstance(ret_msg, FullSyncStartRetMsg):
                return ret_msg
            else:
                logger.error(
                    "Unexpected response type for FullSyncStartMsg: %s", type(ret_msg)
                )
                return None
        except Exception as e:
            logger.error("Error sending FullSyncStartMsg: %s", e)
            return None

    def _send_sync_batch(self, sync_id: str, batch_id: int, keys: List[int]) -> None:
        """Send a batch of keys via PUSH mode"""
        msg = FullSyncBatchMsg(
            instance_id=self.instance_id,
            worker_id=self.worker_id,
            location=self.location,
            sync_id=sync_id,
            batch_id=batch_id,
            keys=keys,
        )
        self.worker.put_msg(msg)

    def _send_sync_end(self, sync_id: str, actual_total_keys: int) -> None:
        """Send FullSyncEndMsg via PUSH mode"""
        msg = FullSyncEndMsg(
            instance_id=self.instance_id,
            worker_id=self.worker_id,
            location=self.location,
            sync_id=sync_id,
            actual_total_keys=actual_total_keys,
        )
        self.worker.put_msg(msg)

    async def _query_sync_status(self, sync_id: str) -> Optional[FullSyncStatusRetMsg]:
        """Query sync status from controller"""
        msg = FullSyncStatusMsg(
            instance_id=self.instance_id,
            worker_id=self.worker_id,
            sync_id=sync_id,
        )

        try:
            ret_msg = await self.worker.async_put_and_wait_msg(msg)
            if isinstance(ret_msg, FullSyncStatusRetMsg):
                return ret_msg
            else:
                logger.error(
                    "Unexpected response type for FullSyncStatusMsg: %s", type(ret_msg)
                )
                return None
        except Exception as e:
            logger.error("Error querying sync status: %s", e)
            return None

    async def start_full_sync(self, reason: Optional[str] = None) -> bool:
        """
        Start the full sync process.

        Args:
            reason: The reason for full sync (e.g., "controller_restart")

        Returns:
            True if sync completed successfully, False otherwise
        """
        if self._is_syncing:
            logger.warning("Full sync already in progress, skipping")
            return False

        self._is_syncing = True
        self._current_sync_id = None

        logger.info(
            "Starting full sync for worker %s:%s, reason: %s",
            self.instance_id,
            self.worker_id,
            reason,
        )

        try:
            # Step 1: Random startup delay to avoid thundering herd
            delay = random.uniform(0, self.startup_delay_range_s)
            logger.info("Full sync startup delay: %.2fs", delay)
            await asyncio.sleep(delay)

            # Step 2: Enter freeze mode
            self.lmcache_engine.freeze(True)

            # Step 3: Get all keys from hot cache
            keys = self._get_all_hot_cache_keys()
            total_keys = len(keys)
            batch_count = (total_keys + self.batch_size - 1) // self.batch_size
            batch_count = max(batch_count, 1)  # At least 1 batch even if empty

            logger.info(
                "Full sync: total_keys=%d, batch_size=%d, batch_count=%d",
                total_keys,
                self.batch_size,
                batch_count,
            )

            # Step 4: Generate sync ID and send start message with retry
            sync_id = self._generate_sync_id()
            self._current_sync_id = sync_id

            start_accepted = False
            for attempt in range(self.max_retry_count):
                ret_msg = await self._send_sync_start(sync_id, total_keys, batch_count)
                if ret_msg is not None and ret_msg.accepted:
                    start_accepted = True
                    break
                logger.warning(
                    "FullSyncStart not accepted, attempt %d/%d, error: %s",
                    attempt + 1,
                    self.max_retry_count,
                    ret_msg.error_msg if ret_msg else "No response",
                )
                await asyncio.sleep(self.retry_delay_s)

            if not start_accepted:
                logger.error(
                    "Failed to start full sync after %d attempts", self.max_retry_count
                )
                return False

            # Step 5: Send keys in batches
            for batch_id in range(batch_count):
                start_idx = batch_id * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_keys)
                batch_keys = keys[start_idx:end_idx]

                self._send_sync_batch(sync_id, batch_id, batch_keys)

                logger.debug(
                    "Sent batch %d/%d with %d keys",
                    batch_id + 1,
                    batch_count,
                    len(batch_keys),
                )

                # Small delay between batches to avoid overwhelming controller
                if self.batch_interval_ms > 0 and batch_id < batch_count - 1:
                    await asyncio.sleep(self.batch_interval_ms / 1000.0)

            # Step 6: Send end message
            self._send_sync_end(sync_id, total_keys)
            logger.info("Full sync batches sent, total_keys=%d", total_keys)

            # Step 7: Poll for completion status
            # TODO(baoloongmao): Make this configurable
            max_poll_attempts = 60  # 5 minutes with 5s interval
            for poll_attempt in range(max_poll_attempts):
                await asyncio.sleep(self.status_poll_interval_s)

                status = await self._query_sync_status(sync_id)
                if status is None:
                    logger.warning(
                        "Failed to query sync status, attempt %d", poll_attempt + 1
                    )
                    continue

                logger.info(
                    "Sync status: is_complete=%s, global_progress=%.1f%%, "
                    "can_exit_freeze=%s",
                    status.is_complete,
                    status.global_progress * 100,
                    status.can_exit_freeze,
                )

                if status.can_exit_freeze:
                    break
            else:
                # TODO(baoloongmao): Use heartbeat to detect controller failure
                # and exit freeze mode if necessary
                logger.warning(
                    "Full sync status poll timeout, exiting freeze mode anyway"
                )

            logger.info("Full sync completed successfully")
            return True

        except Exception as e:
            logger.error("Error during full sync: %s", e)
            return False

        finally:
            # Always clean up state, regardless of success or failure
            self.lmcache_engine.freeze(False)
            self._is_syncing = False
            self._current_sync_id = None

    @property
    def is_syncing(self) -> bool:
        """Check if full sync is currently in progress"""
        return self._is_syncing
