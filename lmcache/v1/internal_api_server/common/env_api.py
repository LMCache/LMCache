# SPDX-License-Identifier: Apache-2.0
# Standard
import json
import os

# Third Party
from fastapi import APIRouter
from starlette.responses import PlainTextResponse

router = APIRouter()


@router.get("/env")
async def get_env():
    """
    Get all environment variables, with credential values redacted.

    A variable is redacted when its upper-cased name contains KEY, TOKEN,
    SECRET, PASSWORD or CREDENTIAL; its name stays visible so operators can
    still confirm it is set.
    """
    env_dict = {
        name: "<redacted>" if _looks_secret(name) else value
        for name, value in os.environ.items()
    }
    return PlainTextResponse(
        content=json.dumps(env_dict, indent=2, sort_keys=True),
        media_type="text/plain",
    )


def _looks_secret(name: str) -> bool:
    """Return whether the variable ``name`` conventionally holds a credential.

    Args:
        name: The environment variable name.

    Returns:
        ``True`` if the upper-cased name contains a credential name fragment.
    """
    # Matches e.g. AWS_SECRET_ACCESS_KEY, HF_TOKEN, LMCACHE_HTTP_API_KEY.
    upper = name.upper()
    return any(
        fragment in upper
        for fragment in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    )
