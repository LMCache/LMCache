# SPDX-License-Identifier: Apache-2.0
"""Standalone HTTP service for shared Device-DAX metadata."""

# Standard
from collections.abc import Callable
import hmac
import os

# Third Party
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_coordinator.api import (
    EpochResponse,
    InvalidReservationError,
    LookupRequest,
    LookupResponse,
    OutOfSpaceError,
    RegionContract,
    ReservationBatchRequest,
    ReservationRef,
    StaleEpochError,
    StatusResponse,
    WriteReserveRequest,
    WriteReserveResponse,
)
from lmcache.v1.memory_coordinator.config import (
    MemoryCoordinatorConfig,
    read_token_file,
)
from lmcache.v1.memory_coordinator.pool import MemoryPool

logger = init_logger(__name__)


def create_app(config: MemoryCoordinatorConfig) -> FastAPI:
    """Build one in-process coordinator for one immutable region.

    A durable startup latch refuses every restart, including clean shutdowns.
    Epoch checks cannot revoke GPU access to old extents, so an operator must
    stop all MP servers before manually removing the latch to reset the pool.
    The latch is not metadata recovery and must survive pod replacement.

    Args:
        config: Validated service and region settings.

    Returns:
        Ready-to-serve FastAPI application.

    Raises:
        ValueError: The token file or region contract is invalid.
        RuntimeError: The persistent startup latch already exists.
        OSError: The startup latch cannot be durably created.
    """
    token = read_token_file(config.token_file).encode()
    pool = MemoryPool(
        config.region_id,
        config.capacity_bytes,
        config.alignment_bytes,
        config.layout_id,
    )
    try:
        descriptor = os.open(
            config.state_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "memory coordinator state_file already exists; stop all MP servers "
            "before manually removing it for a coordinated pool reset"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="ascii") as marker:
        marker.write("LMCache shared L1: coordinated reset required before reuse.\n")
        marker.flush()
        os.fsync(marker.fileno())
    directory = os.open(
        os.path.dirname(config.state_file), os.O_RDONLY | os.O_DIRECTORY
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    app = FastAPI(
        title="LMCache Memory Coordinator",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def require_auth(request: Request) -> None:
        """Require the bearer token loaded from the configured file."""
        scheme, _, presented = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not presented:
            raise HTTPException(status_code=401, detail="bearer token required")
        if not hmac.compare_digest(presented.encode(), token):
            raise HTTPException(status_code=403, detail="invalid bearer token")

    def error(status: int, name: str, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={"error": name, "detail": str(exc)},
        )

    @app.exception_handler(StaleEpochError)
    async def stale_epoch(request: Request, exc: StaleEpochError) -> JSONResponse:
        return error(409, "stale_epoch", exc)

    @app.exception_handler(InvalidReservationError)
    async def invalid_reservation(
        request: Request,
        exc: InvalidReservationError,
    ) -> JSONResponse:
        return error(409, "invalid_reservation", exc)

    @app.exception_handler(OutOfSpaceError)
    async def out_of_space(request: Request, exc: OutOfSpaceError) -> JSONResponse:
        return error(507, "out_of_space", exc)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Return process liveness."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        """Return readiness after the pool has been constructed."""
        return {"status": "ready"}

    @app.get("/v1/region", dependencies=[Depends(require_auth)])
    async def region() -> RegionContract:
        """Return the fixed region contract and restart epoch."""
        return pool.region_contract()

    @app.get("/v1/status", dependencies=[Depends(require_auth)])
    async def status() -> StatusResponse:
        """Return constant-size allocation diagnostics."""
        return pool.status()

    @app.post("/v1/writes/reserve", dependencies=[Depends(require_auth)])
    async def reserve_writes(body: WriteReserveRequest) -> WriteReserveResponse:
        """Reserve absent keys in one capacity-atomic batch."""
        pool.check_epoch(body.region_epoch)
        try:
            grants = pool.reserve_writes(body.items)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return WriteReserveResponse(
            region_epoch=body.region_epoch,
            grants=grants,
        )

    def complete(
        body: ReservationBatchRequest,
        operation: Callable[[list[ReservationRef]], None],
    ) -> EpochResponse:
        pool.check_epoch(body.region_epoch)
        try:
            operation(body.reservations)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return EpochResponse(region_epoch=body.region_epoch)

    @app.post("/v1/writes/finish", dependencies=[Depends(require_auth)])
    async def finish_writes(body: ReservationBatchRequest) -> EpochResponse:
        """Make a complete write batch readable."""
        return complete(body, pool.finish_writes)

    @app.post("/v1/writes/abort", dependencies=[Depends(require_auth)])
    async def abort_writes(body: ReservationBatchRequest) -> EpochResponse:
        """Discard failed write metadata without reusing its extents."""
        return complete(body, pool.abort_writes)

    @app.post("/v1/lookup", dependencies=[Depends(require_auth)])
    async def lookup(body: LookupRequest) -> LookupResponse:
        """Return immutable VALID objects and partial cache misses."""
        pool.check_epoch(body.region_epoch)
        try:
            hits = pool.lookup(body.keys)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return LookupResponse(
            region_epoch=body.region_epoch,
            hits=hits,
        )

    logger.info(
        "Memory coordinator ready: region_id=%s capacity_bytes=%d",
        config.region_id,
        config.capacity_bytes,
    )
    return app
