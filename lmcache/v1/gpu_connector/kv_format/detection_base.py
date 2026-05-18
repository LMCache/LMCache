# SPDX-License-Identifier: Apache-2.0
"""Engine-level KV cache detection: base class + shared helpers.

Each :class:`EngineDetector` subclass knows two things and nothing
else:

1. How to *normalize* a serving engine's raw KV cache into the
   canonical :data:`DiscoverableKVCache` shape (e.g. TRT-LLM's 4-D
   pool tensor needs a reshape into the 6-D cross-layer form).
2. How to *detect* the corresponding ``GPUKVFormat`` from the
   normalized shape and any engine-specific layout hints.

Adding a new engine = drop one new file under ``detectors/``;
registration is automatic via :meth:`__init_subclass__`.
"""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Optional

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.types import (
    DiscoverableKVCache,
    LayoutHints,
)

if TYPE_CHECKING:
    # First Party
    import lmcache.c_ops as lmc_ops


def list_depth_tensor_dim(
    kv_caches: DiscoverableKVCache,
) -> tuple[int, int]:
    """Measure the structural shape of a :data:`DiscoverableKVCache`.

    Returns ``(list_depth, tensor_ndim)``: the number of list-wrapping
    layers (0 for a bare tensor, 1 for a flat list, 2 for nested lists)
    and the ``ndim`` of the innermost tensor.
    """
    depth = 0
    probe: DiscoverableKVCache = kv_caches
    while isinstance(probe, list):
        depth += 1
        if not probe:
            raise ValueError("encountered an empty list")
        probe = probe[0]
    return depth, probe.ndim


def descend_to_tensor(kv_caches: DiscoverableKVCache, depth: int) -> torch.Tensor:
    probe: DiscoverableKVCache = kv_caches
    for _ in range(depth):
        probe = probe[0]  # type: ignore[index]
    assert isinstance(probe, torch.Tensor)
    return probe


class EngineDetector(ABC):
    """Strategy: normalize + detect for one ``EngineType``."""

    engine: ClassVar[EngineType]
    # Set on intermediate bases to opt out of auto-registration.
    abstract: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("abstract", False):
            # First Party
            from lmcache.v1.gpu_connector.kv_format.registry import (
                register_detector_class,
            )

            register_detector_class(cls)

    def normalize(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> DiscoverableKVCache:
        """Return the canonical form. Default: pass-through."""
        return kv_caches

    @abstractmethod
    def detect(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> Optional["lmc_ops.GPUKVFormat"]:
        """Identify the GPU KV format. Return ``None`` if unrecognized."""
