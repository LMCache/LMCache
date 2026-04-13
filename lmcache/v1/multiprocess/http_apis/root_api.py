# SPDX-License-Identifier: Apache-2.0
# Third Party
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    return {"status": "ok", "service": "LMCache HTTP API"}
