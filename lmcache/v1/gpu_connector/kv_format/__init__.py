# SPDX-License-Identifier: Apache-2.0
"""Strategy-based GPU KV format dispatch.

Public surface:

* :class:`KVFormatSpec`, :class:`AxisLayout` and :func:`get_spec` for
  shape access.
* :func:`detect_format` for engine-driven format discovery.
* :data:`DiscoverableKVCache`, :class:`LayoutHints` types.
* :class:`EngineDetector` and registry helpers for diagnostics /
  test-side tear-down.

Specs and detectors live under :mod:`specs` and :mod:`detectors`
respectively. They are discovered lazily on first lookup — adding a
new format or engine means dropping a new file in those directories,
no other source needs to change.
"""

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.base import AxisLayout, KVFormatSpec
from lmcache.v1.gpu_connector.kv_format.detection_base import (
    EngineDetector,
    descend_to_tensor,
    list_depth_tensor_dim,
)
from lmcache.v1.gpu_connector.kv_format.registry import (
    all_format_ids,
    all_gpu_kv_formats,
    ensure_loaded,
    failed_module_reports,
    get_detector,
    get_spec,
    get_spec_class,
    get_spec_class_by_id,
    supported_engines,
    unregister_spec,
)
from lmcache.v1.gpu_connector.kv_format.types import (
    DiscoverableKVCache,
    LayoutHints,
)

if TYPE_CHECKING:
    # Third Party
    import lmc_ops

logger = init_logger(__name__)


def detect_format(
    kv_caches: DiscoverableKVCache,
    serving_engine: EngineType,
    layout_hints: LayoutHints | None = None,
) -> tuple["lmc_ops.GPUKVFormat", DiscoverableKVCache]:
    """Normalize ``kv_caches`` and discover its ``GPUKVFormat``.

    Dispatch goes through the per-engine :class:`EngineDetector`
    registered for ``serving_engine``. Returns
    ``(gpu_kv_format, normalized_kv_caches)``.
    """
    ensure_loaded()
    if layout_hints is None:
        layout_hints = {}  # type: ignore[assignment]
    detector = get_detector(serving_engine)
    if detector is None:
        reports = failed_module_reports()
        raise ValueError(
            "No KV format detector registered for engine "
            f"{serving_engine}. Loaded engines: {supported_engines()}. "
            f"Failed detector modules: {reports['detectors']}"
        )

    kv_caches = detector.normalize(kv_caches, layout_hints)  # type: ignore[arg-type]
    list_depth, tensor_dim = list_depth_tensor_dim(kv_caches)

    # Audit log mirroring the original helper so existing log scrapers
    # keep working.
    probe = kv_caches
    list_dims: list[int] = []
    for _ in range(list_depth):
        list_dims.append(len(probe))  # type: ignore[arg-type]
        probe = probe[0]  # type: ignore[index]
    tensor_dims = list(probe.shape)  # type: ignore[union-attr]
    dims_str = (
        "".join(f"[{d}]" for d in list_dims) + f"[{', '.join(map(str, tensor_dims))}]"
    )
    logger.info("list_depth: %d, tensor_dim: %d", list_depth, tensor_dim)
    logger.info("GPU KV Cache Dimensions: %s", dims_str)

    detected = detector.detect(kv_caches, layout_hints)  # type: ignore[arg-type]
    if detected is None:
        reports = failed_module_reports()
        raise ValueError(
            "currently unsupported kv_caches format with list depth "
            f"{list_depth} and tensor dimension {tensor_dim}. "
            f"Loaded format ids: {all_format_ids()}. "
            f"Failed spec modules: {reports['specs']}"
        )
    spec_cls = get_spec_class(detected)
    if spec_cls is not None:
        logger.info("GPU KV Format: %s", spec_cls.shape_desc)
        logger.info("Currently used by:\n  - %s", spec_cls.backend_label)
    return detected, kv_caches


__all__ = [
    "AxisLayout",
    "DiscoverableKVCache",
    "EngineDetector",
    "KVFormatSpec",
    "LayoutHints",
    "all_format_ids",
    "all_gpu_kv_formats",
    "descend_to_tensor",
    "detect_format",
    "ensure_loaded",
    "failed_module_reports",
    "get_detector",
    "get_spec",
    "get_spec_class",
    "get_spec_class_by_id",
    "list_depth_tensor_dim",
    "supported_engines",
    "unregister_spec",
]
