# SPDX-License-Identifier: Apache-2.0
# Standard
import json
import os

# Third Party
from fastapi import APIRouter
from starlette.responses import PlainTextResponse

router = APIRouter()

_REDACTED_ENV_VARS = frozenset(
    {
        "LMCACHE_API_KEY",
        "LMCACHE_HTTP_API_KEY",
    }
)
_REDACTED_VALUE = "<redacted>"


@router.get("/env")
async def get_env():
    """
    Get all environment variables
    """
    env_dict = {
        name: _REDACTED_VALUE if name in _REDACTED_ENV_VARS else value
        for name, value in os.environ.items()
    }
    return PlainTextResponse(
        content=json.dumps(env_dict, indent=2, sort_keys=True),
        media_type="text/plain",
    )
