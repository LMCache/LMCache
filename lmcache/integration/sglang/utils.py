# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for in-process SGLang configuration helpers."""

# First Party
from lmcache.inprocess.integration.sglang.utils import (
    ENGINE_NAME,
    is_false,
    lmcache_get_config,
)

__all__ = ["ENGINE_NAME", "is_false", "lmcache_get_config"]
