# SPDX-License-Identifier: Apache-2.0
"""Platform-agnostic cache-context factory.

The concrete implementations live in their respective sub-packages:

* :class:`~lmcache.v1.platform.cuda.cache_context.GPUCacheContext` --
  CUDA-backed.
* :class:`~lmcache.v1.platform.cpu.cache_context.CPUCacheContext` --
  CPU-only fallback (POSIX-SHM-backed KV tensors).

:func:`create_cache_context` keeps the dispatch out of the call site
in :mod:`lmcache.v1.multiprocess.server`. Selection is data-driven:
each backend sub-package ships its own
:class:`~lmcache.v1.platform.base.cache_context.BaseCacheContext`
subclass under ``platform/<backend>/cache_context.py`` and declares
the ``torch.device.type`` it handles via the
:attr:`BaseCacheContext.device_type` ClassVar.  The universal registry
in :mod:`lmcache.v1.platform._registry` discovers those subclasses on
first use.  Adding a new accelerator therefore requires *zero* edits to
this module -- just drop a new ``platform/<backend>/cache_context.py``
whose subclass sets ``device_type``.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.platform._registry import get_all_impls
from lmcache.v1.platform.base.cache_context import BaseCacheContext

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)

# ``device_type -> BaseCacheContext`` subclass.  Populated lazily on
# the first :func:`create_cache_context` call via the universal registry.
# Tests substitute entries via :func:`snapshot_backends` /
# :func:`restore_backends`.
#
# The value type is the loose ``type`` (rather than
# ``type[BaseCacheContext]``) so callers can instantiate it with the
# concrete subclass' positional ``__init__`` signature without mypy
# resolving the abstract-base ``__init__`` instead.
_BACKENDS: dict[str, type] = {}
_BACKENDS_DISCOVERED: bool = False


def _discover_backends_once() -> None:
    """Populate :data:`_BACKENDS` on first use.

    Delegates to the universal registry in
    :mod:`lmcache.v1.platform._registry` to discover all concrete
    :class:`BaseCacheContext` subclasses indexed by their ``device_type``
    ClassVar.
    """
    global _BACKENDS_DISCOVERED
    if _BACKENDS_DISCOVERED:
        return

    _BACKENDS.update(get_all_impls(BaseCacheContext))  # type: ignore[type-abstract]
    _BACKENDS_DISCOVERED = True


def _resolve_backend(device_type: str) -> type:
    _discover_backends_once()
    cls = _BACKENDS.get(device_type)
    if cls is None:
        raise ValueError(
            "No cache-context class registered for device type %r. "
            "Make sure ``lmcache.v1.platform.<backend>.cache_context`` "
            "ships a BaseCacheContext subclass with the matching "
            "``device_type`` ClassVar." % device_type
        )
    return cls


def snapshot_backends() -> dict[str, type]:
    """Return a shallow copy of the backend table.

    Pair with :func:`restore_backends` in test fixtures so installing
    fakes for one test does not leak into the next.
    """
    _discover_backends_once()
    return dict(_BACKENDS)


def restore_backends(state: dict[str, type]) -> None:
    """Replace the backend table with *state*.

    Marks the table as already-discovered so further calls do not
    re-trigger filesystem scanning and overwrite the test's fakes.
    """
    global _BACKENDS_DISCOVERED
    _BACKENDS.clear()
    _BACKENDS.update(state)
    _BACKENDS_DISCOVERED = True


def _detect_device_type(kv_caches: KVCache) -> str:
    """Return the ``torch.device.type`` describing *kv_caches*.

    All wrappers in *kv_caches* must materialize tensors on the same
    device type; mixed-device batches are not supported by any
    downstream cache-context implementation.
    """
    device_types = {w.to_tensor().device.type for w in kv_caches}
    if len(device_types) != 1:
        raise ValueError(
            "create_cache_context requires all kv_caches to share one "
            "device type, got %r" % sorted(device_types)
        )
    return next(iter(device_types))


def create_cache_context(
    kv_caches: KVCache,
    lmcache_tokens_per_chunk: int = 256,
    layout_hints: LayoutHints | None = None,
    engine_group_infos: "Sequence[EngineGroupInfo]" = (),
    engine_type: EngineType = EngineType.VLLM,
    separate_object_groups: bool = True,
) -> BaseCacheContext:
    """Create the appropriate cache context for *kv_caches*.

    The signature mirrors :class:`GPUCacheContext` so callers can
    forward their kwargs verbatim and stay agnostic of the active
    backend.

    Selection is driven by ``tensor.device.type`` of *kv_caches*:
    on first use the platform package is scanned for
    ``BaseCacheContext`` subclasses and the one whose
    ``device_type`` ClassVar matches is instantiated.  ``"cuda"``,
    ``"cpu"``, future ``"xpu"`` ... all resolve through the same
    code path -- no ``isinstance`` / ``if-elif`` chain.

    Args:
        kv_caches: KV cache tensor wrappers from the serving engine.
            Must be non-empty.
        lmcache_tokens_per_chunk: Number of tokens per LMCache chunk.
        layout_hints: Optional hints for KV format detection.
            Forwarded verbatim to the concrete context constructor.
        engine_group_infos: Engine-neutral KV cache group metadata.
        engine_type: Which serving engine produced the caches.
        separate_object_groups: When True (default), split kernel groups into
            one object group per sliding-window size; when False, a single
            full-attention object group.

    Returns:
        A concrete cache context instance.

    Raises:
        ValueError: If *kv_caches* is empty, mixes device types, or
            targets a device type with no registered backend.
    """
    if not kv_caches:
        raise ValueError("create_cache_context requires a non-empty kv_caches list")

    device_type = _detect_device_type(kv_caches)
    cls = _resolve_backend(device_type)
    return cls(
        kv_caches,
        lmcache_tokens_per_chunk,
        layout_hints,
        engine_group_infos,
        engine_type,
        separate_object_groups,
    )
