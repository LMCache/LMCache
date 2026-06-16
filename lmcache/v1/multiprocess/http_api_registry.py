# SPDX-License-Identifier: Apache-2.0
# Third Party
from fastapi import APIRouter, FastAPI

# First Party
from lmcache.v1.multiprocess.http_apis.cache_api import router as cache_router
from lmcache.v1.multiprocess.http_apis.common_api import router as common_router
from lmcache.v1.multiprocess.http_apis.conf_api import router as conf_router
from lmcache.v1.multiprocess.http_apis.healthcheck_api import (
    router as healthcheck_router,
)
from lmcache.v1.multiprocess.http_apis.quota_api import router as quota_router
from lmcache.v1.multiprocess.http_apis.reconfigure_api import (
    router as reconfigure_router,
)
from lmcache.v1.multiprocess.http_apis.root_api import router as root_router
from lmcache.v1.multiprocess.http_apis.status_api import router as status_router
from lmcache.v1.multiprocess.http_apis.version_api import router as version_router

# Every multiprocess HTTP API router, registered in import order. Add a new
# ``http_apis`` module's router here to expose it. Routers are imported
# explicitly rather than discovered from the filesystem so that registration
# does not depend on the on-disk package layout, which differs between source
# checkouts and editable/wheel installs.
_MP_HTTP_ROUTERS: tuple[APIRouter, ...] = (
    cache_router,
    common_router,
    conf_router,
    healthcheck_router,
    quota_router,
    reconfigure_router,
    root_router,
    status_router,
    version_router,
)


class HTTPAPIRegistry:
    """Registers the multiprocess HTTP API routers on a FastAPI app.

    The routers exposed on the multiprocess HTTP server are the ones listed
    in :data:`_MP_HTTP_ROUTERS`.
    """

    def __init__(self, app: FastAPI):
        self.app = app
        self.router = APIRouter()

    def register_all_apis(self) -> None:
        """Register every router in :data:`_MP_HTTP_ROUTERS` on the app."""
        for router in _MP_HTTP_ROUTERS:
            self.router.include_router(router)

        self.app.include_router(self.router)
