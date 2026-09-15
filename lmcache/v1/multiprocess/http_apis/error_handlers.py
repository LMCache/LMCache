# SPDX-License-Identifier: Apache-2.0
"""Map cache-control domain errors onto HTTP responses, centrally.

The ``cache_control`` services raise transport-agnostic
:class:`CacheControlError` subclasses; this module owns the single place where
those become HTTP status codes. Registering one handler for the base class
(Starlette matches by the exception's MRO) means routes never need a
``try/except`` -- a domain error raised anywhere in the call stack is converted
here into a ``{"detail": ...}`` body with the mapped status.
"""

# Standard
from http import HTTPStatus
from typing import cast

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.multiprocess.cache_control.errors import (
    CacheControlError,
    Conflict,
    Disabled,
    InvalidRequest,
    NotFound,
    TooLarge,
    Unavailable,
    Unsupported,
)

# Domain error type -> HTTP status. The only place the two vocabularies meet.
_STATUS_BY_ERROR: dict[type[CacheControlError], HTTPStatus] = {
    InvalidRequest: HTTPStatus.BAD_REQUEST,
    NotFound: HTTPStatus.NOT_FOUND,
    Unavailable: HTTPStatus.SERVICE_UNAVAILABLE,
    Conflict: HTTPStatus.CONFLICT,
    Unsupported: HTTPStatus.NOT_IMPLEMENTED,
    Disabled: HTTPStatus.FORBIDDEN,
    TooLarge: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
}


def register_error_handlers(app: FastAPI) -> None:
    """Register the cache-control domain-error handler on ``app``.

    Must be called on the ``FastAPI`` app (not a router); router auto-discovery
    does not pick up exception handlers.

    Args:
        app: The FastAPI application to attach the handler to.
    """

    async def _handle(request: Request, exc: Exception) -> JSONResponse:
        error = cast(CacheControlError, exc)
        status = _STATUS_BY_ERROR.get(type(error), HTTPStatus.INTERNAL_SERVER_ERROR)
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    app.add_exception_handler(CacheControlError, _handle)
