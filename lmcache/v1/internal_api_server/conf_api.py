# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, Optional, Union
import json

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse
import torch

# First Party
from lmcache.v1.config import _CONFIG_DEFINITIONS, LMCacheEngineConfig

router = APIRouter()


def _get_config_dict(
    config: LMCacheEngineConfig,
    keys: Union[list[str], Optional[dict[str, dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Get the config dict filtered by keys"""
    if keys is None:
        keys = _CONFIG_DEFINITIONS
    return {key: getattr(config, key) for key in keys if hasattr(config, key)}


@router.get("/conf")
async def get_config(request: Request, names: Optional[str] = None):
    config = request.app.state.lmcache_adapter.config
    # Parse query parameter names (comma-separated list of config names)
    keys = names.split(",") if names else None
    config_dict = _get_config_dict(config, keys)
    return PlainTextResponse(
        content=json.dumps(config_dict, indent=2), media_type="text/plain"
    )


@router.get("/meta")
async def get_metadata(request: Request, names: Optional[str] = None):
    """
    Get metadata of the cache engine
    """
    lmcache_engine = request.app.state.lmcache_adapter.lmcache_engine
    if not lmcache_engine:
        return PlainTextResponse(
            content="/meta api only work for lmcache_engine", status_code=500
        )
    metadata = lmcache_engine.metadata

    if names:
        attr_list = names.split(",")
        metadata_dict = {}
        for attr in attr_list:
            if (
                hasattr(metadata, attr)
                and not attr.startswith("__")
                and not callable(getattr(metadata, attr))
            ):
                value = getattr(metadata, attr)
                if isinstance(value, (torch.dtype, torch.Size)):
                    value = str(value)
                metadata_dict[attr] = value
    else:
        metadata_dict = {
            attr: getattr(metadata, attr)
            for attr in dir(metadata)
            if not attr.startswith("__") and not callable(getattr(metadata, attr))
        }
        for key, value in metadata_dict.items():
            if isinstance(value, (torch.dtype, torch.Size)):
                metadata_dict[key] = str(value)

    return PlainTextResponse(
        content=json.dumps(metadata_dict, indent=2, default=str),
        media_type="text/plain",
    )
