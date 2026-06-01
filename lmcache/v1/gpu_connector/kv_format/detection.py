# SPDX-License-Identifier: Apache-2.0
"""Normalization and format detection for raw engine KV caches.

``detect_format`` is the single entry point: it normalizes a raw
``kv_caches`` into the canonical structure and discovers its
``EngineKVFormat``. Detection is inherently engine-specific (each serving
engine lays its KV cache out differently), so the per-engine logic lives
in dedicated ``_normalize_*`` / ``_detect_*`` helpers rather than one
monolithic function.

This module is the only place engine identity (``EngineType``) is
consulted; the spec layer is engine-agnostic.
"""

# mypy: disable-error-code="union-attr"
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.specs import get_spec_class
from lmcache.v1.gpu_connector.types import DiscoverableKVCache, LayoutHints
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


def attempt_permute_to_contiguous_view(
    kv_caches: DiscoverableKVCache,
) -> DiscoverableKVCache:
    """Return a contiguous view of *kv_caches*, metadata-only (no copy).

    For a tensor leaf: reorders the dims by stride magnitude so shape
    lines up with a contiguous layout. For a list: recurses into each
    element. Tensor leaves alias the input's storage; list nodes are
    freshly allocated but hold the same tensor objects (or their
    permuted views).

    Recovers the vLLM HND case: tensors allocated physically as
    ``[2, NB, NH, BS, HS]`` but exposed logically as
    ``[2, NB, BS, NH, HS]`` via a dim permute. Sorting dims by stride
    undoes the permute without touching storage.

    For tensors that remain non-contiguous even after dim-permute
    recovery (e.g. vLLM unified KV pool views where dim-0 has an
    inflated periodic stride because every block slot is padded to a
    model-wide maximum), this function returns the tensor unchanged.
    Rationale: :class:`CudaIPCWrapper` transports ``(shape, stride,
    storage_offset)`` verbatim and the receiver rebuilds the view via
    ``torch.Tensor.set_(storage, offset, shape, stride)``, which
    supports arbitrary strided views (including periodic-dim-0 and
    ``as_strided``-produced layouts) and yields a bit-identical view.
    Downstream consumers that rely on ``shape`` alone to infer the
    physical layout must therefore also consult ``stride``.

    We deliberately never fall back to ``.contiguous()`` (which would
    allocate and copy), so the caller's zero-copy invariant is
    preserved.
    """
    if isinstance(kv_caches, torch.Tensor):
        if kv_caches.is_contiguous():
            return kv_caches
        strides = kv_caches.stride()
        perm = sorted(range(kv_caches.ndim), key=lambda i: strides[i], reverse=True)
        result = kv_caches.permute(perm)
        if result.is_contiguous():
            return result
        padding_per_block = _validate_dim0_padded_layout(result)
        logger.debug(
            "attempt_permute_to_contiguous_view: accepting dim-0-padded "
            "view; downstream kernels must honour block_stride_elems. "
            "shape=%s, stride=%s, padding_per_block_elems=%d, "
            "storage_nbytes=%s, dtype=%s",
            tuple(result.shape),
            tuple(result.stride()),
            padding_per_block,
            int(result.untyped_storage().nbytes()),
            result.dtype,
        )
        return result
    return [attempt_permute_to_contiguous_view(sub) for sub in kv_caches]


def _validate_dim0_padded_layout(tensor: torch.Tensor) -> int:
    """Validate that *tensor* matches the dim-0-padding-only strided layout.

    Mainly used for DeepSeek V4 integration, where compressor / indexer
    KV groups share a pool with larger attn groups and end up with
    per-block dim-0 padding. The downstream KV transfer kernels only
    honour this single non-contiguous shape (via
    :class:`PageBufferShapeDesc.block_stride_elems`); any other strided
    view would cause wrong reads/writes and is rejected here.

    The accepted layout requires:

    * ``stride[-1] == 1`` and ``stride[-2] == shape[-1]`` -- each block
      row is internally tightly packed.
    * Every interior dim ``i`` satisfies
      ``stride[i] == prod(shape[i+1:])`` -- only dim-0 may carry
      padding, with ``stride[0] >= prod(shape[1:])``.
    * ``storage_offset == 0`` -- no slice/narrow base shift.

    Callers must pass the stride-sorted permuted view (not the original
    tensor): for tensors that are both permuted and dim-0-padded, the
    original's unsorted inner strides would falsely trip the tight-
    packing check. ``permute`` shares storage and preserves
    ``storage_offset``/``numel``/storage bytes, so those checks are
    equivalent on either view.

    Returns:
        ``padding_per_block_elems`` (= ``stride[0] - prod(shape[1:])``).

    Raises:
        ValueError: *tensor* violates any of the invariants above.
    """
    shape = tuple(tensor.shape)
    stride = tuple(tensor.stride())
    ndim = tensor.ndim
    storage_offset = int(tensor.storage_offset())

    def _fail(reason: str) -> None:
        raise ValueError(
            "attempt_permute_to_contiguous_view: tensor is non-contiguous "
            f"and not a supported (dim-0 padding only) layout -- {reason}. "
            f"shape={shape}, stride={stride}, "
            f"storage_offset={storage_offset}, numel={int(tensor.numel())}, "
            f"storage_nbytes={int(tensor.untyped_storage().nbytes())}, "
            f"dtype={tensor.dtype}. "
            "Downstream KV transfer kernels only understand dim-0 "
            "block-row padding; other strided views would produce "
            "wrong reads/writes and are rejected."
        )

    if ndim < 2:
        _fail("ndim < 2")
    if stride[-1] != 1:
        _fail("stride[-1] != 1 (inner dim not contiguous)")
    if stride[-2] != shape[-1]:
        _fail("stride[-2] != shape[-1] (last-two dims not tightly packed)")
    if storage_offset != 0:
        _fail("storage_offset != 0 (slice/narrow view, base address shifted)")
    inner_tight = 1
    for i in range(ndim - 1, 0, -1):
        if i < ndim - 1 and stride[i] != inner_tight:
            _fail(
                f"dim {i} stride {stride[i]} != tight {inner_tight} "
                "(interior-dim padding is not supported)"
            )
        inner_tight *= shape[i]
    if stride[0] < inner_tight:
        _fail(
            f"dim-0 stride {stride[0]} < prod(shape[1:])={inner_tight} "
            "(overlapping blocks)"
        )
    return stride[0] - inner_tight


def _list_depth_tensor_dim(kv_caches: DiscoverableKVCache) -> tuple[int, int]:
    """Measure the structural shape of a :data:`DiscoverableKVCache`.

    Descends the first element of each list until a tensor is reached,
    counting list-wrapping layers along the way.

    Args:
        kv_caches: A :data:`DiscoverableKVCache` value.

    Returns:
        ``(list_depth, tensor_ndim)`` -- the number of list-wrapping
        layers (0 for a bare tensor, 1 for a flat list, 2 for nested
        lists) and the ``ndim`` of the innermost tensor.

    Raises:
        ValueError: If an empty list is encountered during descent.
    """
    depth = 0
    probe: DiscoverableKVCache = kv_caches
    while isinstance(probe, list):
        depth += 1
        if not probe:
            raise ValueError("encountered an empty list")
        probe = probe[0]
    return depth, probe.ndim


def _normalize_sglang(
    kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
) -> DiscoverableKVCache:
    """Restore SGLang MP's flat per-layer list to the canonical depth-2 form.

    SGLang MP hands us a flat ``list[Tensor]`` of length ``2 * num_layers``
    (first half K layers, second half V layers) so the wire payload fits
    ``KVCache = list[CudaIPCWrapper]``. Restore the canonical depth-2
    ``[K_layers, V_layers]`` shape, and reshape each per-layer tensor
    from ``(page_buffer_size, num_heads, head_size)`` to ``(num_blocks,
    block_size, num_heads, head_size)`` using the engine-supplied
    ``tokens_per_block``. After this, detection lands on the dedicated
    ``TWO_X_NL_X_NB_BS_NH_HS`` enum and num_blocks/block_size become
    readable as ``shape[0]``/``shape[1]``.

    Triggers structurally on a depth-1 list of an even number of 3-D
    Tensors with ``shape[1] > 1`` plus a ``tokens_per_block`` hint;
    returns *kv_caches* unchanged otherwise.

    Raises:
        ValueError: If a per-layer ``page_buffer_size`` is not divisible
            by ``tokens_per_block``.
    """
    if not (
        isinstance(kv_caches, list)
        and len(kv_caches) > 0
        and len(kv_caches) % 2 == 0
        and isinstance(kv_caches[0], torch.Tensor)
        and kv_caches[0].dim() == 3
        and kv_caches[0].shape[1] > 1
        and "tokens_per_block" in layout_hints
    ):
        return kv_caches

    block_size = layout_hints["tokens_per_block"]
    half = len(kv_caches) // 2
    reshaped: list[DiscoverableKVCache] = []
    for layers in (kv_caches[:half], kv_caches[half:]):
        inner: list[DiscoverableKVCache] = []
        for t in layers:
            pbs = t.shape[0]
            if pbs % block_size != 0:
                raise ValueError(
                    f"SGLang KV page_buffer_size {pbs} not divisible by "
                    f"tokens_per_block {block_size}"
                )
            inner.append(t.view(pbs // block_size, block_size, *t.shape[1:]))
        reshaped.append(inner)
    return reshaped


def _normalize_trtllm(
    kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
) -> DiscoverableKVCache:
    """Reshape TRT-LLM's 4-D pool tensor into the canonical 6-D cross-layer form.

    TRT-LLM hands us a 4-D pool tensor (possibly wrapped in a 1-element
    list for adapter-side ergonomics). Reshape to the canonical 6-D
    cross-layer form so detection lands on the standard path. Returns
    *kv_caches* unchanged if it is not the 4-D pool shape.

    Raises:
        ValueError: If the required layout hints are missing, or the
            flat dim does not match ``num_kv_heads * tokens_per_block *
            head_dim``.
    """
    if isinstance(kv_caches, list) and len(kv_caches) == 1:
        kv_caches = kv_caches[0]
    if isinstance(kv_caches, torch.Tensor) and kv_caches.dim() == 4:
        num_kv_heads = layout_hints.get("num_kv_heads")
        tokens_per_block = layout_hints.get("tokens_per_block")
        head_dim = layout_hints.get("head_dim")
        if num_kv_heads is None or tokens_per_block is None or head_dim is None:
            raise ValueError(
                "TRT-LLM normalize requires layout_hints with "
                "num_kv_heads, tokens_per_block, head_dim"
            )
        nb, nl, kv, flat = kv_caches.shape
        if flat != num_kv_heads * tokens_per_block * head_dim:
            raise ValueError(
                f"TRT-LLM 4D tensor flat dim {flat} does not match "
                f"num_kv_heads ({num_kv_heads}) * tokens_per_block "
                f"({tokens_per_block}) * head_dim ({head_dim})"
            )
        kv_caches = kv_caches.view(nb, nl, kv, num_kv_heads, tokens_per_block, head_dim)
    return kv_caches


def _detect_trtllm(list_depth: int, tensor_dim: int) -> "Optional[lmc_ops.EngineKVFormat]":
    """Detect the TRT-LLM cross-layer format from structural shape."""
    if list_depth == 0 and tensor_dim == 6:
        return lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS
    return None


def _detect_vllm(
    probe: DiscoverableKVCache,
    list_depth: int,
    tensor_dim: int,
    layout_hints: LayoutHints,
) -> "Optional[lmc_ops.EngineKVFormat]":
    """Detect the vLLM format from structure plus the ``kv_layout`` hint."""
    kv_layout = layout_hints.get("kv_layout")
    # NOTE: vLLM's CPU attention backend stores KV cache in HND layout.
    # however, get_kv_cache_layout from vllm.v1.attention.backends.utils
    # does not return the right layout for CPU attention.
    # Right fix should come from vllm side, but hardcode here as safeguard.
    if torch_device_type == "cpu":
        kv_layout = "HND"
        logger.info("CPU backend detected, using HND KV cache layout")
    elif kv_layout is None:
        logger.warning(
            "No KV Cache Layout hint provided when using vLLM, defaulting to NHD"
        )
        kv_layout = "NHD"
    logger.info("vLLM KV cache layout: %s", kv_layout)
    is_hnd = kv_layout == "HND"
    if list_depth == 0:
        return lmc_ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS
    elif list_depth == 1:
        if tensor_dim == 5:
            if probe.shape[0] == 2:
                if is_hnd:
                    return lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
                return lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
            elif probe.shape[1] == 2:
                if is_hnd:
                    return lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS
                return lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
        elif tensor_dim == 3:
            return lmc_ops.EngineKVFormat.NL_X_NB_BS_HS
    return None


def _detect_sglang(
    probe: DiscoverableKVCache, list_depth: int, tensor_dim: int
) -> "Optional[lmc_ops.EngineKVFormat]":
    """Detect the SGLang format from structural shape."""
    if list_depth == 1:
        if probe.shape[1] == 1:
            return lmc_ops.EngineKVFormat.NL_X_NBBS_ONE_HS
    elif list_depth == 2:
        if tensor_dim == 4:
            # MP path: reshaped per-layer tensor exposes block_size as
            # ``shape[1]``; ``num_blocks`` as ``shape[0]``.
            return lmc_ops.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS
        return lmc_ops.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS
    return None


def _log_detected_format(engine_kv_format: "lmc_ops.EngineKVFormat") -> None:
    """Log the detected format and its symbolic (geometry-only) shape."""
    spec_cls = get_spec_class(engine_kv_format)
    logger.info("GPU KV Format: %s %s", engine_kv_format, spec_cls.shape_desc)


def detect_format(
    kv_caches: DiscoverableKVCache,
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> tuple["lmc_ops.EngineKVFormat", DiscoverableKVCache]:
    """Normalize *kv_caches* and discover its ``EngineKVFormat``.

    Performs, in order: a stride-based dim permutation so ``.shape``
    reflects the physical layout (critical for HND vs NHD), an
    engine-specific normalization (SGLang un-flatten / TRT-LLM reshape),
    then structural format detection.

    Args:
        kv_caches: The KV cache tensors (possibly nested lists).
        serving_engine: Which serving engine produced the caches.
        layout_hints: Optional engine hints (see :class:`LayoutHints`).

    Returns:
        ``(engine_kv_format, normalized_kv_caches)``. Callers must use the
        returned tensor structure for subsequent operations -- it shares
        storage with the input but may be a permuted view.

    Raises:
        ValueError: If the structure does not match any known format.

    See ``csrc/mem_kernels.cuh`` for the ``EngineKVFormat`` naming schema.
    """
    kv_caches = attempt_permute_to_contiguous_view(kv_caches)

    if layout_hints is None:
        layout_hints = {}

    if serving_engine == EngineType.SGLANG:
        kv_caches = _normalize_sglang(kv_caches, layout_hints)
    elif serving_engine == EngineType.TRTLLM:
        kv_caches = _normalize_trtllm(kv_caches, layout_hints)

    list_depth, tensor_dim = _list_depth_tensor_dim(kv_caches)
    logger.info("list_depth: %d, tensor_dim: %d", list_depth, tensor_dim)
    probe: DiscoverableKVCache = kv_caches
    list_dims = []
    for _ in range(list_depth):
        list_dims.append(len(probe))
        probe = probe[0]
    tensor_dims = list(probe.shape)
    dims_str = (
        "".join(f"[{d}]" for d in list_dims) + f"[{', '.join(map(str, tensor_dims))}]"
    )
    logger.info("GPU KV Cache Dimensions: %s", dims_str)

    if serving_engine == EngineType.TRTLLM:
        detected_format = _detect_trtllm(list_depth, tensor_dim)
    elif serving_engine == EngineType.VLLM:
        detected_format = _detect_vllm(probe, list_depth, tensor_dim, layout_hints)
    elif serving_engine == EngineType.SGLANG:
        detected_format = _detect_sglang(probe, list_depth, tensor_dim)
    else:
        detected_format = None

    if detected_format is not None:
        _log_detected_format(detected_format)
        return detected_format, kv_caches
    raise ValueError(
        "currently unsupported kv_caches format "
        f"with list depth {list_depth} and tensor dimension {tensor_dim}"
    )
