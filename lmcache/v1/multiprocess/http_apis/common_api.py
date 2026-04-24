# SPDX-License-Identifier: Apache-2.0
"""Aggregate HTTP routes exposed by ``lmcache.v1.internal_api_server.common``.

This module discovers every ``*_api`` sub-module under the
``internal_api_server/common`` package and merges their ``router``
attributes into a single :class:`~fastapi.APIRouter` named
``router`` so that :class:`HTTPAPIRegistry` picks it up through the
standard ``http_apis`` auto-discovery mechanism.

Adding a new API module under ``internal_api_server/common`` requires
no changes here: it is registered automatically.
"""

# Standard
from pathlib import Path

# Third Party
from fastapi import APIRouter

# First Party
from lmcache.v1 import internal_api_server
from lmcache.v1.utils.router_discovery import discover_api_routers

router = APIRouter()

_common_path = Path(internal_api_server.__file__).parent / "common"
_common_package = f"{internal_api_server.__name__}.common"

for _discovered in discover_api_routers(_common_path, _common_package):
    router.include_router(_discovered)
