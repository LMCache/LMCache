# SPDX-License-Identifier: Apache-2.0
"""Discovery for the coordinator's views.

Views are found by scanning this package rather than listed at the call
site, so dropping a new one in this directory is the whole of adding it:
startup builds it, subscribes it to the cache-event stream, hands it out
by type, and routes whatever durable state it advertises.
"""

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.discovery import Registry, discover
from lmcache.v1.mp_coordinator.views import base
from lmcache.v1.mp_coordinator.views.base import View

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig

logger = init_logger(__name__)


def build_views(config: "MPCoordinatorConfig") -> Registry[View]:
    """Discover and construct every view in this package.

    Args:
        config: Passed to each ``from_config``.

    Returns:
        A registry with every discovered view built.
    """
    registry: Registry[View] = Registry(
        list(discover(__path__, __name__, View)),
        build=lambda view_type, views: view_type.from_config(config, views),
    )
    built = registry.all()
    logger.debug(
        "Discovered %d view(s): %s",
        len(built),
        ", ".join(type(view).__name__ for view in built),
    )
    return registry


__all__ = ["View", "base", "build_views"]
