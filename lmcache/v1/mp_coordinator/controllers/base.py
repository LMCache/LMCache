# SPDX-License-Identifier: Apache-2.0
"""What makes a class in this package a controller.

A controller acts: it holds policy, drives loops, and answers requests
that change what the fleet does. It reads the fleet's state from views,
and depends on nothing else -- not on another controller, which would
break as soon as that one shipped elsewhere.
"""

# Standard
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Third Party
import httpx

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    from lmcache.v1.mp_coordinator.discovery import Registry
    from lmcache.v1.mp_coordinator.views.base import View


@dataclass(frozen=True)
class ControllerRuntime:
    """What a controller can only be handed once the app is running.

    One field today. It stays a struct so that a second loop-bound
    collaborator can be added without touching the signature of every
    controller that never asked for it.

    Attributes:
        http_client: Shared client for outbound calls to mp servers.
    """

    http_client: httpx.AsyncClient


class Controller:
    """A collaborator the coordinator builds once at startup.

    Subclass to have discovery construct it. Construction and lifetime
    are the whole interface, both defaulted. Consuming the cache-event
    stream, holding durable state and answering HTTP are protocols:
    implement ``consume``, ``get_durable_components`` or ``get_routers``
    and discovery picks that up too.
    """

    @classmethod
    def from_config(
        cls,
        config: "MPCoordinatorConfig",
        views: "Registry[View]",
    ) -> "Controller":
        """Build this controller.

        Views are built on demand, so there is no construction order to
        get right.

        Args:
            config: The coordinator configuration.
            views: The fleet's read models.
        """
        return cls()

    @asynccontextmanager
    async def run(self, runtime: ControllerRuntime) -> AsyncIterator[None]:
        """Run background work for as long as the app is serving.

        Start the work, ``yield`` exactly once -- the app serves for the
        duration of that yield -- then shut it down in a ``finally``, so
        teardown runs whether shutdown was clean or an exception unwound
        the stack::

            task = asyncio.create_task(self._loop())
            try:
                yield
            finally:
                task.cancel()

        Entered once inside the lifespan, so tasks may be created here. A
        context manager rather than a start/stop pair so a controller that
        fails to enter cannot leave the ones before it running, and one
        that never entered is never torn down.

        Yielding without starting anything is how configuration switches
        the work off. Defaults to no work at all.

        Args:
            runtime: What cannot come from :meth:`from_config`, because
                it binds to the running event loop.
        """
        yield
