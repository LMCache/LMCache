# SPDX-License-Identifier: Apache-2.0
"""What makes a class in this package a controller.

A controller acts: it holds policy, drives loops, and answers requests
that change what the fleet does. It reads the fleet's state from views,
and depends on nothing else -- not on another controller, which would
break as soon as that one shipped elsewhere.
"""

# Standard
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Third Party
import httpx

# First Party
from lmcache.v1.mp_coordinator.background_loops import BackgroundLoops
from lmcache.v1.mp_coordinator.metric_sources import register_metric_source

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    from lmcache.v1.mp_coordinator.discovery import Registry
    from lmcache.v1.mp_coordinator.views.base import View


@dataclass(frozen=True)
class ControllerRuntime:
    """What a controller can only be handed once the app is running.

    A struct, so a loop-bound collaborator can be added without touching
    the signature of every controller that never asked for it.

    Attributes:
        http_client: Shared client for outbound calls to mp servers.
        background_loops: Where a controller starts its periodic work, so a
            failed pass is counted and the loop keeps running, and the loop
            is reported with the coordinator's own. Defaults to a private
            runner, so a runtime built with only ``http_client`` still works.
    """

    http_client: httpx.AsyncClient
    background_loops: BackgroundLoops = field(default_factory=BackgroundLoops)


class ControllerStatus:
    """Which controllers started.

    Every controller starts out not started; the app marks each one whose
    ``run()`` it entered without an error.
    """

    def __init__(self, controller_names: Iterable[str]) -> None:
        """Track ``controller_names``, none of them started yet.

        Args:
            controller_names: The class name of every controller built.
        """
        self._up = {name: False for name in controller_names}
        register_metric_source(self)

    def mark_up(self, controller_name: str) -> None:
        """Record that the named controller started.

        Args:
            controller_name: The controller's class name.
        """
        self._up[controller_name] = True

    def stats(self) -> dict[str, bool]:
        """Return whether each controller, keyed by class name, started."""
        return dict(self._up)


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
