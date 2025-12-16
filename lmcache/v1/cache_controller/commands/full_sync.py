# SPDX-License-Identifier: Apache-2.0
"""Full sync command implementation"""

# Standard
from typing import TYPE_CHECKING
import asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.commands.base import HeartbeatCommand

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class FullSyncCommand(HeartbeatCommand, tag="full_sync"):
    """Command to request full state synchronization

    Sent when the controller needs the worker to perform a full sync,
    e.g., after controller restart or worker re-registration.
    """

    # FullSync-specific fields can be added here, e.g.:
    # sync_scope: Optional[str] = None  # "all", "metadata", "data"
    # priority: int = 0  # Sync priority level

    def describe(self) -> str:
        return f"FullSyncCommand(reason={self.reason}, args={self.args})"

    def execute(self, worker: "LMCacheWorker") -> None:
        """Trigger full sync on the worker"""
        logger.info(
            "Received full sync command with reason: %s, args: %s",
            self.reason,
            self.args,
        )

        # Check if full sync is already in progress
        if worker._full_sync_in_progress:
            logger.warning("Full sync already in progress, skipping")
            return

        # Trigger full sync in background
        worker._full_sync_in_progress = True
        asyncio.create_task(self._do_full_sync(worker, self.reason))

    async def _do_full_sync(
        self, worker: "LMCacheWorker", reason: str | None = None
    ) -> None:
        """Perform full sync in background"""
        try:
            sender = worker._get_full_sync_sender()
            success = await sender.start_full_sync(reason)
            if success:
                logger.info("Full sync completed successfully")
            else:
                logger.error("Full sync failed")
        except Exception as e:
            logger.error("Error during full sync: %s", e)
        finally:
            worker._full_sync_in_progress = False
