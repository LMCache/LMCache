# SPDX-License-Identifier: Apache-2.0
"""Shared data models, descriptors, and foundational types for lmcache.v1.common."""

# First Party
import lmcache.lmcache_native as _native

PageBufferShapeDesc = _native.PageBufferShapeDesc

__all__ = [
    "PageBufferShapeDesc",
]
