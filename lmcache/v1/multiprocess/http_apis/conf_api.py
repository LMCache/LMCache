# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict, is_dataclass
from typing import Any
import json

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.utils.json_utils import make_json_safe

router = APIRouter()

_SENSITIVE_FIELD_MARKERS = ("password", "secret", "token", "credential", "api_key")


class _IndentedJSONResponse(JSONResponse):
    """JSONResponse with indented output for readability."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ).encode("utf-8")


def _serialize_config_value(name: str, value: Any) -> Any:
    if _is_sensitive_field(name) and value not in (None, ""):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _serialize_config_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_config_value("", item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_config_value(name, asdict(value))
    if hasattr(value, "__dict__") and not isinstance(value, type):
        public_attrs = {
            key: _serialize_config_value(key, attr_value)
            for key, attr_value in vars(value).items()
            if not key.startswith("_") and not callable(attr_value)
        }
        if public_attrs:
            return {"__class__": type(value).__name__, **public_attrs}
    return make_json_safe(value)


def _is_sensitive_field(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _SENSITIVE_FIELD_MARKERS)


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
        result[name] = _serialize_config_value(name, cfg)
    return _IndentedJSONResponse(content=result)
