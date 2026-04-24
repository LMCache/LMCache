# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict
from typing import Any
import json

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


class _IndentedJSONResponse(JSONResponse):
    """JSONResponse with indented output for readability."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")


def _make_json_safe(obj: Any) -> Any:
    """Recursively ensure all values are JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


@router.get("/conf")
async def conf(request: Request) -> Any:
    """
    Return all server configurations (mp, storage_manager,
    observability) as a single JSON object.
    """
    configs = getattr(request.app.state, "configs", None)
    if configs is None:
        return JSONResponse(
            status_code=503,
            content={"error": "configs not initialized"},
        )
    result = {}
    for name, cfg in configs.items():
        if hasattr(cfg, "__dataclass_fields__"):
            result[name] = _make_json_safe(asdict(cfg))
        else:
            result[name] = _make_json_safe(cfg)
    return _IndentedJSONResponse(content=result)
