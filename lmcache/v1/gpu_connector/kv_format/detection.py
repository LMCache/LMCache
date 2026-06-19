# SPDX-License-Identifier: Apache-2.0
"""Format detection orchestration for raw engine KV caches.

``detect_format`` applies engine-agnostic contiguous-view recovery, then hands
off to the engine's :class:`EngineDetector` to reshape and identify the format.
"""

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.contiguity import (
    attempt_permute_to_contiguous_view,
)
from lmcache.v1.gpu_connector.kv_format.detectors import get_detector
from lmcache.v1.gpu_connector.kv_format.specs import describe_shape
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


def detect_format(
    kv_caches: DiscoverableKVCache,
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> "tuple[lmc_ops.EngineKVFormat, DiscoverableKVCache]":
    """Recover a contiguous view, then discover the format + canonical kv_caches.

    Returns ``(engine_kv_format, normalized_kv_caches)``. Callers must use the
    returned structure -- it shares storage with the input but may be a
    permuted/reshaped view.

    Raises:
        ValueError: If no detector exists for *serving_engine*, or the structure
            matches no known format.
    """
    kv_caches = attempt_permute_to_contiguous_view(kv_caches)
    detector = get_detector(serving_engine)
    if detector is None:
        raise ValueError(f"no KV cache detector for serving engine {serving_engine}")
    engine_kv_format, kv_caches = detector.discover(kv_caches, layout_hints or {})
    if engine_kv_format is None:
        raise ValueError(f"unsupported kv_caches structure for {serving_engine}")
    logger.info(
        "Engine KV Format: %s %s", engine_kv_format, describe_shape(engine_kv_format)
    )
    return engine_kv_format, kv_caches
