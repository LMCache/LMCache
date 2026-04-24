# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/api/status")
async def status(request: Request) -> Any:
    """
    Detailed status endpoint for inspecting internal state
    of all MP components (L1 cache, L2 adapters, controllers,
    sessions).
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return engine.report_status()
