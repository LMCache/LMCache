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
from lmcache.v1.gpu_connector.kv_format.declared import (
    resolve_declared_tensor,
    tensor_axes,
)
from lmcache.v1.gpu_connector.kv_format.detectors import get_detector
from lmcache.v1.gpu_connector.kv_format.specs import describe_shape
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
    if layout_hints is not None and "tensor_layouts" in layout_hints:
        if serving_engine != EngineType.VLLM or not isinstance(kv_caches, list):
            raise ValueError("Declared tensor layouts require per-layer vLLM caches")
        axes_per_layer = tensor_axes(layout_hints, len(kv_caches))
        normalized: list[DiscoverableKVCache] = []
        formats = []
        legacy_hints = layout_hints.copy()
        del legacy_hints["tensor_layouts"]
        for cache, axes in zip(kv_caches, axes_per_layer, strict=True):
            if axes is None:
                fmt, views = detect_format([cache], serving_engine, legacy_hints)
                normalized.append(views[0])
            else:
                if not isinstance(cache, torch.Tensor):
                    raise ValueError(
                        "Declared layout requires a single tensor per layer"
                    )
                fmt, view = resolve_declared_tensor(cache, axes)
                normalized.append(view)
            formats.append(fmt)
        if not formats or len(set(formats)) != 1:
            raise ValueError(
                "Declared layers have different formats; use "
                "normalize_and_discover_per_layer_formats"
            )
        return formats[0], normalized
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
