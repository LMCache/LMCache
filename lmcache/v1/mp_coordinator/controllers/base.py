# SPDX-License-Identifier: Apache-2.0
"""What makes a class in this package a controller.

A controller acts: it holds policy, drives loops, and answers requests
that change what the fleet does. It reads the fleet's state from views
rather than keeping its own copy.

Controllers may depend on views and on other controllers. Views cannot
depend on controllers -- their ``from_config`` is handed no way to reach
one -- so the direction cannot invert by accident.
"""

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    from lmcache.v1.mp_coordinator.discovery import Registry
    from lmcache.v1.mp_coordinator.views.base import View


class Controller:
    """A collaborator the coordinator builds once at startup.

    Subclass to have discovery construct it. Consuming the cache-event
    stream and holding durable state are separate protocols -- implement
    ``consume`` or ``get_durable_components`` and discovery picks that up
    too, so a controller that does neither declares neither.
    """

    @classmethod
    def from_config(
        cls,
        config: "MPCoordinatorConfig",
        views: "Registry[View]",
        controllers: "Registry[Controller]",
    ) -> "Controller":
        """Build this controller.

        Defaults to needing none of the arguments, so only a controller
        that reads configuration or depends on a peer writes this hook.
        Ask either registry and the peer is built on demand -- there is no
        construction order to get right.

        Args:
            config: The coordinator configuration.
            views: The fleet's read models.
            controllers: The registry being populated.
        """
        return cls()
