# SPDX-License-Identifier: Apache-2.0
"""Endpoint that receives commands pushed by the mp coordinator.

The mp coordinator POSTs commands here (the connect-back / push channel). This
is a no-op stub for now: it accepts and acknowledges a command so the push path
works end to end. Future controllers (quota reconcile, KV-op fan-out) extend
this to route commands into the engine.
"""

# Standard
from typing import Any

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

router = APIRouter()


@router.post("/coordinator/command")
async def coordinator_command(request: Request) -> Any:
    """Accept a command pushed by the coordinator.

    Body: an arbitrary JSON object describing the command. Currently
    acknowledged without action.

    Returns:
        ``{"status": "ok"}`` on a well-formed body, or a 400 JSON error.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON body"})
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400, content={"error": "body must be an object"}
        )

    logger.debug("Received coordinator command: %s", body.get("type", "<unspecified>"))
    return {"status": "ok"}
