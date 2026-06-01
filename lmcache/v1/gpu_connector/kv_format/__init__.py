# SPDX-License-Identifier: Apache-2.0
"""Format-dispatched geometry for GPU KV caches.

Public surface:

- :class:`KVFormatSpec` -- per-format geometry interface.
- :func:`get_spec` / :func:`get_spec_class` -- look up the spec for a
  ``EngineKVFormat`` (instance for geometry, class for static facts).
- :func:`detect_format` -- normalize a raw ``kv_caches`` and discover
  its ``EngineKVFormat``.
"""

# First Party
from lmcache.v1.gpu_connector.kv_format.base import KVFormatSpec
from lmcache.v1.gpu_connector.kv_format.detection import detect_format
from lmcache.v1.gpu_connector.kv_format.specs import get_spec, get_spec_class

__all__ = [
    "KVFormatSpec",
    "detect_format",
    "get_spec",
    "get_spec_class",
]
