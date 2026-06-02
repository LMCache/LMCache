# SPDX-License-Identifier: Apache-2.0
"""Entrypoint for the mp coordinator process.

Run with ``python -m lmcache.v1.mp_coordinator``. Configuration is read
from ``LMCACHE_MP_COORDINATOR_*`` environment variables (see
:class:`MPCoordinatorConfig`).
"""

# Standard
import asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.manager import MPCoordinatorManager

logger = init_logger(__name__)


def main() -> None:
    """Build the coordinator from the environment and run it until interrupted."""
    config = MPCoordinatorConfig.from_env()
    manager = MPCoordinatorManager(config)
    try:
        asyncio.run(manager.start_all())
    except KeyboardInterrupt:
        logger.info("MP coordinator interrupted; shutting down")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
