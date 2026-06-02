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
from lmcache.v1.mp_coordinator.zmq_transport import ZmqCoordinatorTransport
from lmcache.v1.rpc_utils import get_zmq_context

logger = init_logger(__name__)


def main() -> None:
    """Build the coordinator from the environment and run it until interrupted."""
    config = MPCoordinatorConfig.from_env()
    transport = ZmqCoordinatorTransport(
        get_zmq_context(), request_url=config.reply_url, push_url=config.pull_url
    )
    manager = MPCoordinatorManager(config, transport)
    try:
        asyncio.run(manager.start_all())
    except KeyboardInterrupt:
        logger.info("MP coordinator interrupted; shutting down")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
