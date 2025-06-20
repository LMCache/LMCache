# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Union
import copy
import os

# First Party
from lmcache.config import LMCacheEngineConfig as Config  # type: ignore[assignment]
from lmcache.logging import init_logger
from lmcache.v1.config import (
    LMCacheEngineConfig as V1Config,  # type: ignore[assignment]
)

if TYPE_CHECKING:
    # Third Party
    from vllm.multimodal.inputs import PlaceholderRange

logger = init_logger(__name__)
ENGINE_NAME = "vllm-instance"


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def lmcache_get_config() -> Union[Config, V1Config]:
    """Get the LMCache configuration from the environment variable
    `LMCACHE_CONFIG_FILE`. If the environment variable is not set, this
    function will return the default configuration.
    """

    if is_false(os.getenv("LMCACHE_USE_EXPERIMENTAL", "True")):
        logger.warning(
            "Detected LMCACHE_USE_EXPERIMENTAL is set to False. "
            "Using legacy configuration is deprecated and will "
            "be remove soon! Please set LMCACHE_USE_EXPERIMENTAL "
            "to True."
        )
        LMCacheEngineConfig = Config  # type: ignore[assignment]
    else:
        LMCacheEngineConfig = V1Config  # type: ignore[assignment]

    if "LMCACHE_CONFIG_FILE" not in os.environ:
        logger.warn(
            "No LMCache configuration file is set. Trying to read"
            " configurations from the environment variables."
        )
        logger.warn(
            "You can set the configuration file through "
            "the environment variable: LMCACHE_CONFIG_FILE"
        )
        config = LMCacheEngineConfig.from_env()
    else:
        config_file = os.environ["LMCACHE_CONFIG_FILE"]
        logger.info(f"Loading LMCache config file {config_file}")
        config = LMCacheEngineConfig.from_file(config_file)

    return config


def hex_hash_to_int16(s: str) -> int:
    """
    Convert a hex hash string to a 16-bit integer.
    """
    return int(s, 16) & 0xFFFF


def apply_mm_hashes_to_token_ids(
    token_ids: list[int], mm_hashes: list[str], mm_positions: list[PlaceholderRange]
) -> None:
    """
    Overwrite token_ids in-place for multimodal placeholders.

    Args:
        token_ids: The list of token IDs to modify.
        mm_hashes: Hexadecimal hash strings for each placeholder.
        mm_positions: Corresponding placeholder ranges.
    """
    for hash_str, placeholder in zip(mm_hashes, mm_positions, strict=False):
        start = placeholder.offset
        end = start + placeholder.length
        hash_int = hex_hash_to_int16(hash_str)
        for idx in range(start, end):
            if idx < len(token_ids):
                token_ids[idx] = hash_int


def mask_mm_hashes_in_request(request) -> list[int]:
    # No multimodal hashes in the request, return the original request
    if request.mm_hashes is None or len(request.mm_hashes) == 0:
        return request

    # Mask the multimodal hashes in the request's prompt token ids
    cloned = copy.deepcopy(request)
    apply_mm_hashes_to_token_ids(
        cloned.prompt_token_ids, cloned.mm_hashes, cloned.mm_positions
    )
    return cloned
