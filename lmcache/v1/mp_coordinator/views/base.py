# SPDX-License-Identifier: Apache-2.0
"""What makes a class in this package a view.

A view is a read model of the fleet: built by consuming the cache-event
stream, queried by request handlers, and shared by everything that needs
to know what the fleet has cached. It holds no policy and drives nothing.

Views may depend on other views, and nothing else -- ``from_config``
cannot reach a controller, so the direction cannot invert by accident.
"""

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    from lmcache.v1.mp_coordinator.discovery import Registry


class View:
    """A read model the coordinator builds once at startup.

    Subclass to have discovery construct it. Consuming the cache-event
    stream and holding durable state are separate protocols -- implement
    ``consume`` or ``get_durable_components`` and discovery picks that up
    too, so a view that does neither declares neither.
    """

    @classmethod
    def from_config(
        cls, config: "MPCoordinatorConfig", views: "Registry[View]"
    ) -> "View":
        """Build this view.

        Defaults to needing neither argument. Ask ``views`` for a peer and
        it is built on demand -- there is no construction order to get
        right.

        Args:
            config: The coordinator configuration.
            views: The registry being populated.
        """
        return cls()
