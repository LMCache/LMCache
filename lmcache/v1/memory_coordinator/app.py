# SPDX-License-Identifier: Apache-2.0
"""Standalone HTTP service for shared Device-DAX metadata."""

# Standard
import hmac
import os

# Third Party
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.memory_coordinator.api import (
    EpochResponse,
    InvalidReservationError,
    LookupRequest,
    LookupResponse,
    MemoryCoordinatorError,
    OutOfSpaceError,
    RegionContract,
    ReservationBatchRequest,
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


def create_app(config: MemoryCoordinatorConfig) -> FastAPI:
    """Return a single-pool HTTP app using validated ``config`` settings.

    Invalid tokens/contracts raise ValueError; an existing durable startup
    latch raises RuntimeError. Other file errors propagate. The latch survives
    shutdown: stop every mapped worker before removing it to reset the pool.
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

    @app.exception_handler(MemoryCoordinatorError)
    async def coordinator_error(
        request: Request, exc: MemoryCoordinatorError
    ) -> JSONResponse:
        status, name = {
            StaleEpochError: (409, "stale_epoch"),
            InvalidReservationError: (409, "invalid_reservation"),
            OutOfSpaceError: (507, "out_of_space"),
        }[type(exc)]
        return JSONResponse(
            status_code=status,
            content={"error": name, "detail": str(exc)},
        )

    @app.exception_handler(ValueError)
    async def invalid_request(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

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
        return WriteReserveResponse(
            region_epoch=body.region_epoch,
            grants=pool.reserve_writes(body.items),
        )

    @app.post("/v1/writes/finish", dependencies=[Depends(require_auth)])
    async def finish_writes(body: ReservationBatchRequest) -> EpochResponse:
        """Make a complete write batch readable."""
        pool.check_epoch(body.region_epoch)
        pool.finish_writes(body.reservations)
        return EpochResponse(region_epoch=body.region_epoch)

    @app.post("/v1/writes/abort", dependencies=[Depends(require_auth)])
    async def abort_writes(body: ReservationBatchRequest) -> EpochResponse:
        """Discard failed write metadata without reusing its extents."""
        pool.check_epoch(body.region_epoch)
        pool.abort_writes(body.reservations)
        return EpochResponse(region_epoch=body.region_epoch)

    @app.post("/v1/lookup", dependencies=[Depends(require_auth)])
    async def lookup(body: LookupRequest) -> LookupResponse:
        """Return immutable VALID objects and partial cache misses."""
        pool.check_epoch(body.region_epoch)
        return LookupResponse(
            region_epoch=body.region_epoch,
            hits=pool.lookup(body.keys),
        )

    return app
