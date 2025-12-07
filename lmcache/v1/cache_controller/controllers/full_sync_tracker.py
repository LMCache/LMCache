# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Tuple
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.utils import (
    FullSyncState,
    RegistryTree,
    WorkerSyncInfo,
)

logger = init_logger(__name__)


class FullSyncTracker:
    """
    Tracks full sync state for all workers.

    This class manages the state of full sync operations, including:
    - Tracking which workers need full sync
    - Monitoring sync progress
    - Handling sync timeout
    - Determining when freeze mode can be exited
    """

    def __init__(
        self,
        registry_tree: RegistryTree,
        completion_threshold: float = 0.8,
        sync_timeout_s: float = 300.0,
    ):
        """
        Initialize the FullSyncTracker.

        Args:
            registry_tree: The registry tree containing worker nodes
            completion_threshold: Percentage of workers that need to complete
                                  sync before others can exit freeze mode (default: 80%)
            sync_timeout_s: Timeout in seconds for a single worker's sync
                                  (default: 300s)
        """
        self.registry_tree = registry_tree
        self.completion_threshold = completion_threshold
        self.sync_timeout_s = sync_timeout_s

        # Flag to indicate if controller just restarted and needs full sync
        self._need_full_sync_all = True

    def _get_sync_info(
        self, instance_id: str, worker_id: int
    ) -> Optional[WorkerSyncInfo]:
        """Get sync info for a worker from the registry tree."""
        worker_node = self.registry_tree.get_worker(instance_id, worker_id)
        if worker_node is None:
            return None
        return worker_node.sync_info

    def _set_sync_info(
        self, instance_id: str, worker_id: int, sync_info: Optional[WorkerSyncInfo]
    ) -> bool:
        """Set sync info for a worker. Returns True if successful."""
        worker_node = self.registry_tree.get_worker(instance_id, worker_id)
        if worker_node is None:
            return False
        worker_node.sync_info = sync_info
        return True

    def set_need_full_sync_all(self, need: bool) -> None:
        """Set whether all workers need full sync (e.g., after controller restart)"""
        self._need_full_sync_all = need
        logger.info("Set need_full_sync_all to %s", need)

    def _get_all_workers(self) -> list[Tuple[str, int]]:
        """Get all registered workers from the registry tree."""
        return [
            (info.instance_id, info.worker_id)
            for info in self.registry_tree.get_all_worker_infos()
        ]

    def should_request_full_sync(
        self, instance_id: str, worker_id: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a worker should perform full sync.

        Returns:
            Tuple of (need_sync, reason)
        """
        # Case 1: Controller just restarted, all workers need sync
        if self._need_full_sync_all:
            sync_info = self._get_sync_info(instance_id, worker_id)
            if sync_info is None or sync_info.state not in (
                FullSyncState.SYNCING,
                FullSyncState.COMPLETED,
            ):
                return True, "controller_restart"

        # Case 2: Worker sync failed/timeout, needs retry
        sync_info = self._get_sync_info(instance_id, worker_id)
        if sync_info is not None and sync_info.state == FullSyncState.FAILED:
            return True, "sync_failed_retry"

        return False, None

    def is_worker_syncing(self, instance_id: str, worker_id: int) -> bool:
        """
        Check if a worker is currently in sync state.

        When a worker is syncing, incremental events should be discarded.
        """
        sync_info = self._get_sync_info(instance_id, worker_id)
        if sync_info is None:
            return False
        return sync_info.state == FullSyncState.SYNCING

    def start_sync(
        self,
        instance_id: str,
        worker_id: int,
        sync_id: str,
        total_keys: int,
        batch_count: int,
    ) -> bool:
        """
        Start sync for a worker.

        Returns:
            True if sync started successfully, False otherwise
        """
        report_id = (instance_id, worker_id)
        current_time = time.time()

        # Check if already syncing with different sync_id
        existing_sync = self._get_sync_info(instance_id, worker_id)
        if existing_sync is not None and existing_sync.state == FullSyncState.SYNCING:
            if existing_sync.sync_id != sync_id:
                logger.warning(
                    "Worker %s already syncing with different sync_id: "
                    "existing=%s, new=%s",
                    report_id,
                    existing_sync.sync_id,
                    sync_id,
                )
                return False

        new_sync_info = WorkerSyncInfo(
            sync_id=sync_id,
            state=FullSyncState.SYNCING,
            start_time=current_time,
            expected_total_keys=total_keys,
            expected_batch_count=batch_count,
            last_activity_time=current_time,
        )
        if not self._set_sync_info(instance_id, worker_id, new_sync_info):
            logger.warning(
                "Failed to start sync for worker %s: worker not found", report_id
            )
            return False

        logger.info(
            "Started full sync for worker %s: sync_id=%s, "
            "expected_keys=%d, expected_batches=%d",
            report_id,
            sync_id,
            total_keys,
            batch_count,
        )
        return True

    def receive_batch(
        self,
        instance_id: str,
        worker_id: int,
        sync_id: str,
        batch_id: int,
        keys_count: int,
    ) -> bool:
        """
        Record receipt of a sync batch.

        Returns:
            True if batch was recorded, False if invalid
        """
        report_id = (instance_id, worker_id)
        sync_info = self._get_sync_info(instance_id, worker_id)

        if sync_info is None:
            logger.warning(
                "Received batch for unknown sync session: worker=%s, sync_id=%s",
                report_id,
                sync_id,
            )
            return False

        if sync_info.sync_id != sync_id:
            logger.warning(
                "Sync ID mismatch: expected=%s, received=%s",
                sync_info.sync_id,
                sync_id,
            )
            return False

        if sync_info.state != FullSyncState.SYNCING:
            logger.warning(
                "Received batch for non-syncing worker: worker=%s, state=%s",
                report_id,
                sync_info.state,
            )
            return False

        sync_info.received_batches.add(batch_id)
        sync_info.received_keys_count += keys_count
        sync_info.last_activity_time = time.time()

        logger.debug(
            "Received batch %d for worker %s: keys=%d, total_received=%d",
            batch_id,
            report_id,
            keys_count,
            sync_info.received_keys_count,
        )
        return True

    def complete_sync(
        self,
        instance_id: str,
        worker_id: int,
        sync_id: str,
        actual_total_keys: int,
    ) -> bool:
        """
        Mark sync as completed for a worker.

        Returns:
            True if completion was successful, False otherwise
        """
        report_id = (instance_id, worker_id)
        sync_info = self._get_sync_info(instance_id, worker_id)

        if sync_info is None:
            logger.warning(
                "Received sync end for unknown session: worker=%s, sync_id=%s",
                report_id,
                sync_id,
            )
            return False

        if sync_info.sync_id != sync_id:
            logger.warning(
                "Sync ID mismatch on completion: expected=%s, received=%s",
                sync_info.sync_id,
                sync_id,
            )
            return False

        # Verify key count
        if sync_info.received_keys_count != actual_total_keys:
            logger.warning(
                "Key count mismatch on completion: received=%d, reported=%d",
                sync_info.received_keys_count,
                actual_total_keys,
            )
            # Still mark as completed but log the discrepancy

        sync_info.state = FullSyncState.COMPLETED
        sync_info.last_activity_time = time.time()

        logger.info(
            "Completed full sync for worker %s: sync_id=%s, "
            "received_keys=%d, batches=%d",
            report_id,
            sync_id,
            sync_info.received_keys_count,
            len(sync_info.received_batches),
        )
        return True

    def mark_failed(self, instance_id: str, worker_id: int, reason: str) -> None:
        """Mark a worker's sync as failed"""
        report_id = (instance_id, worker_id)
        sync_info = self._get_sync_info(instance_id, worker_id)

        if sync_info is not None:
            sync_info.state = FullSyncState.FAILED
            logger.warning(
                "Marked sync as failed for worker %s: reason=%s", report_id, reason
            )

    def check_sync_timeout(self) -> None:
        """
        Check for sync timeouts and mark failed workers.

        This should be called periodically (e.g., in health check loop).
        """
        current_time = time.time()
        for instance_id, worker_id in self._get_all_workers():
            sync_info = self._get_sync_info(instance_id, worker_id)
            if sync_info is not None and sync_info.state == FullSyncState.SYNCING:
                if current_time - sync_info.last_activity_time > self.sync_timeout_s:
                    self.mark_failed(
                        instance_id,
                        worker_id,
                        f"timeout after {self.sync_timeout_s}s",
                    )

    def get_global_progress(self) -> float:
        """
        Get the global sync progress.

        Returns:
            Progress as a float between 0.0 and 1.0
        """
        all_workers = self._get_all_workers()
        if not all_workers:
            return 0.0

        completed = sum(
            1
            for instance_id, worker_id in all_workers
            if (sync_info := self._get_sync_info(instance_id, worker_id)) is not None
            and sync_info.state == FullSyncState.COMPLETED
        )
        return completed / len(all_workers)

    def get_completed_count(self) -> int:
        """Get count of workers that have completed sync"""
        return sum(
            1
            for instance_id, worker_id in self._get_all_workers()
            if (sync_info := self._get_sync_info(instance_id, worker_id)) is not None
            and sync_info.state == FullSyncState.COMPLETED
        )

    def get_syncing_count(self) -> int:
        """Get count of workers currently syncing"""
        return sum(
            1
            for instance_id, worker_id in self._get_all_workers()
            if (sync_info := self._get_sync_info(instance_id, worker_id)) is not None
            and sync_info.state == FullSyncState.SYNCING
        )

    def can_exit_freeze(self) -> bool:
        """
        Check if the completion threshold is reached and freeze mode can be exited.

        Returns:
            True if enough workers have completed sync
        """
        progress = self.get_global_progress()
        can_exit = progress >= self.completion_threshold

        if can_exit and self._need_full_sync_all:
            # Once threshold is reached, disable the global full sync flag
            logger.info(
                "Full sync completion threshold reached (%.1f%%), "
                "disabling need_full_sync_all",
                progress * 100,
            )
            self._need_full_sync_all = False

        return can_exit

    def get_sync_status(
        self, instance_id: str, worker_id: int, sync_id: str
    ) -> Tuple[bool, float, bool]:
        """
        Get sync status for a specific worker.

        Returns:
            Tuple of (is_complete, global_progress, can_exit_freeze)
        """
        sync_info = self._get_sync_info(instance_id, worker_id)

        is_complete = (
            sync_info is not None
            and sync_info.sync_id == sync_id
            and sync_info.state == FullSyncState.COMPLETED
        )
        global_progress = self.get_global_progress()
        can_exit = self.can_exit_freeze()

        return is_complete, global_progress, can_exit
