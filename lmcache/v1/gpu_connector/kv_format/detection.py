# SPDX-License-Identifier: Apache-2.0
"""Format detection orchestration for raw engine KV caches.

``detect_format`` applies engine-agnostic contiguous-view recovery, then hands
off to the engine's :class:`EngineDetector` to reshape and identify the format.
"""

# Standard
from collections.abc import Mapping

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


def is_indexer_cache(
    kv_cache: torch.Tensor,
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> bool:
    """Return whether one per-layer tensor is a sparse-attention indexer cache.

    Detects the tensor's format as a single-layer registration and reads the
    ``is_indexer`` fact off its spec, so the answer follows the format table
    rather than any engine layer class. Unrecognized layouts are not indexers.
    """
    kv_caches = attempt_permute_to_contiguous_view([kv_cache])
    detector = get_detector(serving_engine)
    if detector is None:
        return False
    engine_kv_format, _ = detector.discover(kv_caches, layout_hints or {})
    return engine_kv_format is not None and get_spec_class(engine_kv_format).is_indexer


def drop_indexer_caches(
    kv_caches: Mapping[str, torch.Tensor],
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> dict[str, torch.Tensor]:
    """Return *kv_caches* without its indexer caches, in registration order.

    DSA models (DeepSeek-V3.2, GLM-5.3) register an indexer k-cache per sparse
    layer beside the attention KV caches. The engine manages those itself, and
    connectors that size themselves to the attention layer count cannot carry
    them, so registration paths call this before handing the caches on. Tensors
    are returned as-is (no copies); a summary is logged only when something was
    dropped.
    """
    kept = {
        name: kv_cache
        for name, kv_cache in kv_caches.items()
        if not is_indexer_cache(kv_cache, serving_engine, layout_hints)
    }
    dropped = len(kv_caches) - len(kept)
    if dropped:
        logger.info(
            "Skipping %d indexer KV caches, registering %d attention KV caches",
            dropped,
            len(kept),
        )
    return kept
