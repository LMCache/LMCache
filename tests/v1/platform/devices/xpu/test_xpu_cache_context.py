# SPDX-License-Identifier: Apache-2.0
"""Tests for the XPU cache context's BaseCacheContext contract."""

# Standard
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import torch

# First Party
from lmcache.v1.platform.devices.xpu.cache_context import XPUCacheContext


def test_implements_every_abstract_method() -> None:
    """A base-class method left unimplemented makes the context uninstantiable."""
    assert not XPUCacheContext.__abstractmethods__


def test_temp_staging_buffer_is_prefix_of_staging_buffer() -> None:
    """The region starts at the buffer's first byte; too large a request is None."""
    ctx = object.__new__(XPUCacheContext)
    staging = torch.arange(16, dtype=torch.uint8)
    ctx._temp_buffer = cast(Any, SimpleNamespace(buffer=staging))

    region = ctx.get_temp_staging_buffer(10)
    assert region is not None
    assert region.data_ptr() == staging.data_ptr()
    assert region.dtype == torch.uint8
    assert region.numel() == 10

    whole = ctx.get_temp_staging_buffer(16)
    assert whole is not None and whole.numel() == 16
    assert ctx.get_temp_staging_buffer(17) is None
