# SPDX-License-Identifier: Apache-2.0
"""Registering HTTP endpoints from a controller package.

The routers in ``http_apis`` resolve their collaborator with
``controllers.get(SomeClass)``, which only reaches a class the coordinator
already imports -- exactly what an out-of-tree controller is not. Such a
controller hands over routers built around itself instead.
"""

# Standard
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# Third Party
from fastapi import APIRouter


@runtime_checkable
class HttpRoutes(Protocol):
    """Something that answers HTTP requests about itself."""

    def get_routers(self) -> Sequence[APIRouter]:
        """Return the routers to mount, built around this object.

        Called once at startup, after the ``http_apis`` routers, so a path
        already claimed there wins. Bind handlers to ``self`` rather than
        resolving the owner per request -- that lookup is what this
        removes.
        """
        ...
