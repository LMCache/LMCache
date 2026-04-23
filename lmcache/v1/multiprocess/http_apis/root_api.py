# SPDX-License-Identifier: Apache-2.0
# Third Party
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root() -> dict[str, str]:
    """
    Basic liveness check endpoint.
    Returns:
        dict: A dictionary containing the status and service name.
    """
    return {"status": "ok", "service": "LMCache HTTP API"}
