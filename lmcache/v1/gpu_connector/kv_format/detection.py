# SPDX-License-Identifier: Apache-2.0
"""Format detection orchestration for raw engine KV caches.

``detect_format`` applies engine-agnostic contiguous-view recovery, then hands
off to the engine's :class:`EngineDetector` to reshape and identify the format.
"""

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.contiguity import (
    attempt_permute_to_contiguous_view,
)
from lmcache.v1.gpu_connector.kv_format.detectors import get_detector
from lmcache.v1.gpu_connector.kv_format.specs import describe_shape, get_spec_class
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def extract_kv_cache_shapes(
    kv_caches: DiscoverableKVCache,
) -> set[torch.Size]:
    """Extract the tensor shapes that is in the kv_caches structure.

    Args:
        kv_caches (DiscoverableKVCache): The kv_caches structure to extract shapes
        from.

    Returns:
        set[torch.Size]: A set of tensor shapes found in the kv_caches structure.
    """
    if isinstance(kv_caches, torch.Tensor):
        return {kv_caches.shape}

    shapes = set()
    for t in kv_caches:
        shapes.update(extract_kv_cache_shapes(t))

    return shapes


def detect_format(
    kv_caches: DiscoverableKVCache,
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> "tuple[lmcache_native.EngineKVFormat, DiscoverableKVCache]":
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


def find_indexer_caches(
    kv_caches: dict[str, torch.Tensor], serving_engine: EngineType
) -> list[str]:
    """Return the names in *kv_caches* whose format spec declares ``is_indexer``.

    DSA models (DeepSeek-V3.2, GLM-5.3) register a sparse-attention indexer
    k-cache per sparse layer beside the attention KV. Each tensor is detected on
    its own; layouts detection rejects are not indexers.
    """
    detector = get_detector(serving_engine)
    if detector is None:
        return []

    def is_indexer(kv_cache: torch.Tensor) -> bool:
        try:
            view = attempt_permute_to_contiguous_view([kv_cache])
            fmt, _ = detector.discover(view, {})
        except ValueError:
            return False
        return fmt is not None and get_spec_class(fmt).is_indexer

    return [name for name, t in kv_caches.items() if is_indexer(t)]
