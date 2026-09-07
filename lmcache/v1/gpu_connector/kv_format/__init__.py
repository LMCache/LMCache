# SPDX-License-Identifier: Apache-2.0
"""Format-dispatched geometry for GPU KV caches.

Public surface:

- :class:`KVFormatSpec` -- per-format geometry interface.
- :func:`get_spec` / :func:`get_spec_class` -- look up the spec for a format.
- :func:`detect_format` -- normalize a raw ``kv_caches`` and discover its format.
- :func:`is_indexer_cache` / :func:`drop_indexer_caches` -- keep sparse-attention
  indexer caches (spec fact ``is_indexer``) away from attention-KV connectors.
- :func:`describe_shape` / :func:`concrete_shape` -- render a format's symbolic /
  numeric shape string.
"""

# First Party
from lmcache.v1.gpu_connector.kv_format.detection import (
    detect_format,
    drop_indexer_caches,
    extract_kv_cache_shapes,
    is_indexer_cache,
)
from lmcache.v1.gpu_connector.kv_format.specs import (
    KVFormatSpec,
    concrete_shape,
    describe_shape,
    get_spec,
    get_spec_class,
)

__all__ = [
    "KVFormatSpec",
    "concrete_shape",
    "describe_shape",
    "detect_format",
    "drop_indexer_caches",
    "extract_kv_cache_shapes",
    "get_spec",
    "get_spec_class",
    "is_indexer_cache",
]
