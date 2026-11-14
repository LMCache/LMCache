# SPDX-License-Identifier: Apache-2.0
"""Destination for data types and utilities needed by both client and server.

This package must remain independent of client and server implementations.
"""

# First Party
from lmcache.v1.common.types import PageBufferShapeDesc

__all__ = [
    "PageBufferShapeDesc",
]
