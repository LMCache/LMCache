# SPDX-License-Identifier: Apache-2.0
"""Entrypoint for the mp coordinator process.

Run with ``python -m lmcache.v1.mp_coordinator``. Configuration is read from
``LMCACHE_MP_COORDINATOR_*`` environment variables (see
:class:`MPCoordinatorConfig`).
"""

# Third Party
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.observability import init_coordinator_metrics

logger = init_logger(__name__)


def main() -> None:
    """Build the coordinator app from the environment and serve it."""
    config = MPCoordinatorConfig.from_env()
    init_coordinator_metrics(config)
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        timeout_keep_alive=config.timeout_keep_alive,
    )


if __name__ == "__main__":
    main()
