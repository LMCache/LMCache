# SPDX-License-Identifier: Apache-2.0
"""Discovery for the coordinator's controllers.

Controllers are found by scanning this package rather than listed at the
call site, so dropping a new one in this directory is the whole of adding
it: startup builds it, hands it out by type, and routes whatever durable
state it advertises.
"""

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.controllers.base import Controller
from lmcache.v1.mp_coordinator.discovery import Registry, discover

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    from lmcache.v1.mp_coordinator.views.base import View

logger = init_logger(__name__)


def build_controllers(
    config: "MPCoordinatorConfig", views: "Registry[View]"
) -> Registry[Controller]:
    """Discover and construct every controller in this package.

    Args:
        config: Passed to each ``from_config``.
        views: The fleet's read models, which a controller may depend on.

    Returns:
        A registry with every discovered controller built.
    """
    registry: Registry[Controller] = Registry(
        list(discover(__path__, __name__, Controller)),
        build=lambda controller_type, controllers: controller_type.from_config(
            config, views, controllers
        ),
    )
    built = registry.all()
    logger.debug(
        "Discovered %d controller(s): %s",
        len(built),
        ", ".join(type(controller).__name__ for controller in built),
    )
    return registry


__all__ = ["Controller", "build_controllers"]
