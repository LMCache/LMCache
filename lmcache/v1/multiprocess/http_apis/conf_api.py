# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import is_dataclass
from typing import Any
import json

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.utils.json_utils import make_json_safe, safe_asdict

router = APIRouter()


class _IndentedJSONResponse(JSONResponse):
    """JSONResponse with indented output for readability."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ).encode("utf-8")


@router.get("/conf")
async def conf(request: Request) -> Any:
    """
    Return all server configurations (mp, storage_manager,
    observability) as a single JSON object.

    Args:
        request (Request): The incoming HTTP request; its
            ``app.state.configs`` mapping is serialized.

    Returns:
        Any: A JSON response whose body is a dict keyed by
        config name. Returns HTTP 503 if ``configs`` is not
        initialized yet.

    Exceptions:
        None.
    """
    configs = getattr(request.app.state, "configs", None)
    if configs is None:
        return JSONResponse(
            status_code=503,
            content={"error": "configs not initialized"},
        )
    result = {}
    for name, cfg in configs.items():
        if is_dataclass(cfg) and not isinstance(cfg, type):
            result[name] = safe_asdict(cfg)
        else:
            result[name] = make_json_safe(cfg)
    return _IndentedJSONResponse(content=result)
