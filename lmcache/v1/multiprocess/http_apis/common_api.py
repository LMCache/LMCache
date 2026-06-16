# SPDX-License-Identifier: Apache-2.0
"""Aggregate HTTP routes exposed by ``lmcache.v1.internal_api_server.common``.

This module merges the ``router`` from every API module under the
``internal_api_server/common`` package into a single
:class:`~fastapi.APIRouter` named ``router``, which :class:`HTTPAPIRegistry`
registers on the multiprocess HTTP server.

To expose a new common API on the multiprocess server, import its ``router``
below and add it to :data:`_COMMON_ROUTERS`. Routers are imported explicitly
rather than discovered from the filesystem so that registration does not depend
on the on-disk package layout, which differs between source checkouts and
editable/wheel installs.

Note:
    Some modules under ``internal_api_server/common`` target the vLLM-embedded
    API server and rely on ``app.state`` attributes that only exist there (e.g.
    ``lmcache_adapter``). Those are simply left out of :data:`_COMMON_ROUTERS`.
"""

# Third Party
from fastapi import APIRouter

# First Party
from lmcache.v1.internal_api_server.common.env_api import router as env_router
from lmcache.v1.internal_api_server.common.loglevel_api import router as loglevel_router
from lmcache.v1.internal_api_server.common.metrics_api import router as metrics_router
from lmcache.v1.internal_api_server.common.periodic_thread_api import (
    router as periodic_thread_router,
)
from lmcache.v1.internal_api_server.common.run_script_api import (
    router as run_script_router,
)
from lmcache.v1.internal_api_server.common.thread_api import router as thread_router

# Common API routers that are safe to serve from the multiprocess HTTP server.
_COMMON_ROUTERS: tuple[APIRouter, ...] = (
    env_router,
    loglevel_router,
    metrics_router,
    periodic_thread_router,
    run_script_router,
    thread_router,
)

router = APIRouter()
for _common_router in _COMMON_ROUTERS:
    router.include_router(_common_router)
