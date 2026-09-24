# SPDX-License-Identifier: Apache-2.0
"""Run the standalone Memory Coordinator from environment configuration."""

# Third Party
import uvicorn

# First Party
from lmcache.v1.memory_coordinator.app import create_app
from lmcache.v1.memory_coordinator.config import MemoryCoordinatorConfig


def main() -> None:
    """Serve one in-process pool with exactly one Uvicorn worker."""
    config = MemoryCoordinatorConfig.from_env()
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
