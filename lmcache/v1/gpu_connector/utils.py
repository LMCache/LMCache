# SPDX-License-Identifier: Apache-2.0
# This module is now a *facade* layer over
# :mod:`lmcache.v1.gpu_connector.kv_format`. The format-aware shape
# accessors (``get_num_blocks``, ``get_num_heads``, ...) were
# refactored from a giant if/elif ladder into per-format
# :class:`KVFormatSpec` strategy classes. The module-level functions
# below are kept as a thin compatibility surface so the 17+ existing
# call sites do not need to be touched in one go; new code should
# prefer :func:`kv_format.get_spec` and reuse the spec instance.
# Silence union-attr errors only for this file because the legacy
# helpers still take ``DiscoverableKVCache`` directly without
# narrowing.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import TYPE_CHECKING, Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.kv_format import (
    DiscoverableKVCache,
    LayoutHints,
    all_gpu_kv_formats,
    detect_format,
    get_spec,
    get_spec_class,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface

# First Party
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

# Re-export so ``from lmcache.v1.gpu_connector.utils import
# DiscoverableKVCache, LayoutHints`` keeps working.
__all__ = [
    "DiscoverableKVCache",
    "LayoutHints",
]

# Error message for accessing non-existent attributes in GPU KV Cache.
# Parenthesized so Python actually concatenates the three string literals —
# adjacent literals on *separate lines* at module scope do NOT concatenate
# implicitly; without the parens, only the first fragment survives and the
# {format} placeholder is lost.
_ATTRIBUTE_NOT_EXIST_ERROR = (
    "trying to access an attribute of the GPU KV Cache "
    "that does not exist for the format detected {format}. "
    "A misalignment with the GPUKVFormat must be resolved"
)


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
        # Non-permute non-contiguity: only the strict dim-0-padding pattern
        # is recoverable downstream. Delegate validation + diagnostics to
        # the helper; on success we keep the stride-sorted view as-is and
        # rely on ``PageBufferShapeDesc.block_stride_elems`` to honour the
        # padding.
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

    * ``stride[-1] == 1`` and ``stride[-2] == shape[-1]`` — each block
      row is internally tightly packed.
    * Every interior dim ``i`` satisfies
      ``stride[i] == prod(shape[i+1:])`` — only dim-0 may carry
      padding, with ``stride[0] >= prod(shape[1:])``.
    * ``storage_offset == 0`` — no slice/narrow base shift.

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
            f"and not a supported (dim-0 padding only) layout — {reason}. "
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
    # Interior dims (1 .. ndim-2 exclusive) must be tightly packed with
    # respect to the dims to their right. Only dim-0's stride is allowed
    # to exceed the tight value.
    inner_tight = 1
    for i in range(ndim - 1, 0, -1):
        if i < ndim - 1 and stride[i] != inner_tight:
            _fail(
                f"dim {i} stride {stride[i]} != tight {inner_tight} "
                "(interior-dim padding is not supported)"
            )
        inner_tight *= shape[i]
    # Now ``inner_tight == prod(shape[1:])``; dim-0 must be >= that.
    if stride[0] < inner_tight:
        _fail(
            f"dim-0 stride {stride[0]} < prod(shape[1:])={inner_tight} "
            "(overlapping blocks)"
        )
    return stride[0] - inner_tight


def assert_contiguous(tensor: torch.Tensor) -> None:
    """Assert that *tensor* has a contiguous physical layout with zero offset.

    LMCache transfer kernels assume logical and physical views match for
    coalesced memory accesses. Used at boundaries where we receive a
    tensor we can't or shouldn't permute (e.g. raw CUDA-IPC reconstruction
    in :class:`~lmcache.v1.multiprocess.custom_types.RawCudaIPCWrapper`).

    Raises:
        ValueError: If *tensor* has a nonzero storage offset, or is
            non-contiguous.
    """
    if tensor.storage_offset() != 0:
        raise ValueError(f"expected storage_offset 0, got {tensor.storage_offset()}")
    if not tensor.is_contiguous():
        raise ValueError("tensor is not contiguous")


def is_cross_layer_format(gpu_kv_format: "lmc_ops.GPUKVFormat") -> bool:
    """Return ``True`` if *gpu_kv_format* stores all layers in one tensor.

    Cross-layer formats — ``NB_NL_TWO_BS_NH_HS`` (vLLM, NHD) and
    ``NB_NL_TWO_NH_BS_HS`` (TRT-LLM, HND) — are represented as a single
    bare :class:`torch.Tensor` rather than a list-of-tensors keyed by
    layer index.
    """
    cls = get_spec_class(gpu_kv_format)
    return cls is not None and cls.is_cross_layer


def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig):
    """
    Check if the GPU Connector needs to create an intermediate
    buffer on the GPU
    """
    if lmcache_config.enable_pd:
        return False
    else:
        return True


def assert_layerwise_gpu_connector(gpu_connector: "GPUConnectorInterface"):
    """
    Assert that a GPU Connector is a layerwise connector.
    """
    # Import at runtime to avoid circular dependency
    # First Party
    from lmcache.v1.gpu_connector import gpu_connectors, xpu_connectors

    valid_connectors = (
        gpu_connectors.VLLMPagedMemLayerwiseGPUConnector,
        gpu_connectors.VLLMBufferLayerwiseGPUConnector,
        gpu_connectors.SGLangLayerwiseGPUConnector,
        xpu_connectors.VLLMPagedMemLayerwiseXPUConnector,
        xpu_connectors.VLLMBufferLayerwiseXPUConnector,
    )

    assert isinstance(gpu_connector, valid_connectors)


def get_gpu_kv_shape_description(gpu_kv_format: "lmc_ops.GPUKVFormat") -> str:
    """Return a human-readable shape description for the GPU KV format.

    Uses short names matching the ``GPUKVFormat`` enum convention:
    NB=num_blocks, NL=num_layers, BS=block_size, NH=num_heads,
    HS=head_size, PBS=page_buffer_size (NB*BS).
    """
    cls = get_spec_class(gpu_kv_format)
    if cls is None:
        return f"Unknown ({gpu_kv_format})"
    return cls.shape_desc


def get_attention_backend(gpu_kv_format: "lmc_ops.GPUKVFormat") -> str:
    """Return the attention backend name for the GPU KV format."""
    cls = get_spec_class(gpu_kv_format)
    if cls is None:
        return f"Unknown ({gpu_kv_format})"
    return cls.backend_label


def get_concrete_gpu_kv_shape(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> str:
    """Return the shape with actual numeric values substituted.

    For example, instead of ``NL x [2, NB, BS, NH, HS]``
    this returns ``80 x [2, 2048, 128, 8, 128]``.
    """
    cls = get_spec_class(gpu_kv_format)
    if cls is None:
        return f"Unknown ({gpu_kv_format})"
    return cls(kv_caches).concrete_shape_str()


def legible_print_gpu_kv_format(gpu_kv_format: "lmc_ops.GPUKVFormat"):
    """
    Print the GPU KV Format in a legible way
    """
    cls = get_spec_class(gpu_kv_format)
    if cls is None:
        logger.warning(f"Unknown GPU KV Format: {gpu_kv_format}")
    else:
        logger.info("GPU KV Format: %s", cls.shape_desc)
        logger.info("Currently used by:\n  - %s", cls.backend_label)


def normalize_kv_and_discover_format(
    kv_caches: DiscoverableKVCache,
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> tuple["lmc_ops.GPUKVFormat", DiscoverableKVCache]:
    """Compatibility shim — first runs the stride-based permute fixup
    (so HND vs NHD detection lands on physical-shape order), then
    delegates to
    :func:`lmcache.v1.gpu_connector.kv_format.detect_format`.
    """
    kv_caches = attempt_permute_to_contiguous_view(kv_caches)
    return detect_format(kv_caches, serving_engine, layout_hints)


def get_num_layers(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> int:
    """Get the number of layers from the kv_caches."""
    return get_spec(kv_caches, gpu_kv_format).num_layers()


def get_num_blocks(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> int:
    """Get the number of blocks from the kv_caches."""
    return get_spec(kv_caches, gpu_kv_format).num_blocks()


def get_block_size(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int = 0,
) -> int:
    """Return the block size (tokens per block) for layer ``layer_idx``.

    ``layer_idx`` is honoured only for per-layer formats where BS may
    differ across layers (e.g. mixed-compression MLA pools). For
    cross-layer formats BS is shared across layers and ``layer_idx``
    is ignored. Raises ``ValueError`` for NBBS-fused formats, which
    have no separate BS dim.
    """
    return get_spec(kv_caches, gpu_kv_format).block_size(layer_idx)


def get_page_buffer_size(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> int:
    """Get page buffer size (num_blocks * block_size) from the kv_caches."""
    return get_spec(kv_caches, gpu_kv_format).page_buffer_size()


def get_num_heads(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int = 0,
) -> int:
    """Get the number of heads for a layer (defaults to layer 0)."""
    return get_spec(kv_caches, gpu_kv_format).num_heads(layer_idx)


def get_hidden_dim_size(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int = 0,
) -> int:
    """Get the hidden dimension for a layer (defaults to layer 0)."""
    return get_spec(kv_caches, gpu_kv_format).hidden_dim(layer_idx)


def get_head_size(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int = 0,
) -> int:
    """Get the head size for a layer (defaults to layer 0)."""
    return get_spec(kv_caches, gpu_kv_format).head_size(layer_idx)


def get_tokens_per_layer(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> int:
    """Get the number of tokens per layer (== page_buffer_size)."""
    return get_spec(kv_caches, gpu_kv_format).tokens_per_layer()


def get_elements_per_layer(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> int:
    """Get the number of elements per layer (K + V for non-MLA)."""
    return get_spec(kv_caches, gpu_kv_format).elements_per_layer()


def assert_is_vllm_flash_attn_or_flash_infer(gpu_kv_format: "lmc_ops.GPUKVFormat"):
    """
    Ensure that we have a GPU KV Cache Format
    that is either vLLM's flash attention or flash infer.

    Resolved dynamically from the registered spec metadata so adding a
    new vLLM non-MLA backend is purely additive (drop a new file under
    :mod:`lmcache.v1.gpu_connector.kv_format.specs`).
    """
    cls = get_spec_class(gpu_kv_format)
    assert (
        cls is not None
        and getattr(cls, "engine", None) == "vllm"
        and not cls.is_mla
        and not cls.is_cross_layer
    ), f"expected a vLLM non-MLA flash attention/infer format, got {gpu_kv_format!r}"


def is_hnd(gpu_kv_format: "lmc_ops.GPUKVFormat") -> bool:
    """Check if the GPU KV Format uses HND physical layout."""
    cls = get_spec_class(gpu_kv_format)
    return cls is not None and cls.is_hnd


def assert_is_vllm_mla_or_flash_attn_or_flash_infer(
    gpu_kv_format: "lmc_ops.GPUKVFormat",
) -> None:
    """
    Ensure that we have a GPU KV Cache Format that is either
    vLLM's MLA, flash attention, or flash infer.

    Resolved dynamically from the registered spec metadata: any
    vLLM-engine spec that is not cross-layer qualifies. Adding a new
    vLLM per-layer format requires no edits here.
    """
    cls = get_spec_class(gpu_kv_format)
    assert (
        cls is not None
        and getattr(cls, "engine", None) == "vllm"
        and not cls.is_cross_layer
    ), (
        f"expected a vLLM MLA / flash attention / flash infer format, got "
        f"{gpu_kv_format!r}"
    )


def is_mla(gpu_kv_format: "lmc_ops.GPUKVFormat") -> bool:
    """Check if the GPU KV Format is MLA."""
    cls = get_spec_class(gpu_kv_format)
    return cls is not None and cls.is_mla


def get_dtype(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int = 0,
) -> torch.dtype:
    """Get the dtype for a layer (defaults to layer 0)."""
    return get_spec(kv_caches, gpu_kv_format).dtype(layer_idx)


def get_group_data_ptrs(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_indices: list[int],
) -> list[int]:
    """Return device pointers for a group of layers in the order the
    transfer kernels expect for *gpu_kv_format*.

    Three buckets, mirroring the kernel dispatch in
    ``csrc/mp_mem_kernels.cu:160-169``:

    - Per-layer list formats: ``[p_{i0}, p_{i1}, ..., p_{iN}]`` — one
      pointer per requested layer, in the given order.
    - ``TWO_X_NL_X_NBBS_NH_HS`` (SGLang MHA): K's grouped first,
      then V's: ``[K_{i0}, ..., K_{iN}, V_{i0}, ..., V_{iN}]``.
    - Cross-layer formats (``NB_NL_TWO_BS_NH_HS`` /
      ``NB_NL_TWO_NH_BS_HS``): a single base pointer ``[base]``;
      ``layer_indices`` is unused.

    Raises:
        ValueError: If *gpu_kv_format* is not recognized.
    """
    return get_spec(kv_caches, gpu_kv_format).data_ptrs(layer_indices)


def get_device(kv_caches: DiscoverableKVCache) -> torch.device:
    """Return the device of the KV cache tensors.

    Descends into any list nesting until a tensor is found; assumes all
    tensors in *kv_caches* live on the same device (true for every
    current :class:`GPUKVFormat`).
    """
    probe: DiscoverableKVCache = kv_caches
    while isinstance(probe, list):
        probe = probe[0]
    return probe.device


# Formats whose per-layer tensor dim-0 is the *block* axis AND for
# which we currently support dim-0 padding (e.g. DeepSeek V4
# compressor / indexer caches sharing a KV pool with larger attn
# groups). Today only the MLA layout (``NL_X_NB_BS_HS``, kv_size==1)
# is exercised by real mixed-compression workloads.
#
# Sourced from :class:`KVFormatSpec.is_block_axis_dim0` so adding a
# new block-axis format only requires flipping that flag in the spec.
#
# ``NL_X_NB_TWO_BS_NH_HS`` *could* in principle also be the block
# axis on dim-0, but no real serving engine emits a padded layout of
# that format yet, and supporting it would require: (a) deciding
# (without a ground-truth example) which axis carries the padding
# — NB boundary vs K↔V offset — and (b) a coordinated change in
# ``attempt_permute_to_contiguous_view`` to let interior-dim padding
# through for that one format. Rather than ship an unverifiable code
# path, we keep ``NL_X_NB_TWO_BS_NH_HS`` out of this set, which means
# any padded tensor of that format will fail loudly via the
# non-block-axis dim-0-padding check below. Revisit and add a
# properly-tested branch when a concrete use case lands.
def _block_axis_formats() -> frozenset:
    return frozenset(
        fmt
        for fmt in all_gpu_kv_formats()
        if (cls := get_spec_class(fmt)) is not None and cls.is_block_axis_dim0
    )


_BLOCK_AXIS_FORMATS: frozenset = _block_axis_formats()


def resolve_block_stride_and_log_layout(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int,
    group_idx: int,
) -> Optional[int]:
    """Resolve the per-block stride for a KV layer group and log its layout.

    Single entry point for :class:`KVLayerGroupsManager` to obtain the
    ``block_stride_elems`` value for :class:`PageBufferShapeDesc` and emit
    a one-shot layout audit line. All ``GPUKVFormat``-aware reasoning is
    kept here so callers never touch a "representative KV cache" tensor.

    * Block-axis formats (:data:`_BLOCK_AXIS_FORMATS`): ``stride(0)`` is
      the per-block step and is returned as-is. A value larger than the
      tight stride indicates dim-0 padding (e.g. DeepSeek V4 compressor
      caches sharing a KV pool with larger attn groups).
    * Other formats: dim-0 is not the block axis, so ``None`` is
      returned and ``shape_desc`` falls back to the tight stride. Any
      dim-0 padding in such formats is rejected with ``ValueError``
      since downstream kernels cannot honour it.

    Args:
        kv_caches: Full KV cache structure (already normalised).
        gpu_kv_format: Format of ``kv_caches``.
        layer_idx: 0-based layer index used as the layout probe.
        group_idx: 0-based group index, used only for logging.

    Returns:
        ``stride(0)`` for block-axis formats; ``None`` otherwise.

    Raises:
        ValueError: Non-block-axis format carries dim-0 padding.
    """
    spec = get_spec(kv_caches, gpu_kv_format)
    rep = spec.layout_probe_tensor(layer_idx)

    block_stride_elems: Optional[int]
    if gpu_kv_format in _BLOCK_AXIS_FORMATS and rep.ndim > 0:
        block_stride_elems = int(rep.stride(0))
    else:
        # Non-block-axis format: detect forbidden dim-0 padding.
        if rep.ndim >= 2:
            tight_dim0 = 1
            for d in range(1, rep.ndim):
                tight_dim0 *= int(rep.shape[d])
            padding = int(rep.stride(0)) - tight_dim0
            if padding > 0:
                raise ValueError(
                    "resolve_block_stride_and_log_layout: group's probe "
                    f"tensor has dim-0 padding ({padding} elements per "
                    f"block) but gpu_kv_format={gpu_kv_format!r} is not "
                    "a supported dim-0-padded format (only "
                    "NL_X_NB_BS_HS is); downstream transfer kernels "
                    "cannot honour this padding and would read/write "
                    "wrong bytes. "
                    f"layer_idx={layer_idx}, shape={tuple(rep.shape)}, "
                    f"stride={tuple(rep.stride())}, "
                    f"tight_stride0={tight_dim0}, "
                    f"storage_offset={int(rep.storage_offset())}, "
                    f"dtype={rep.dtype}."
                )
        block_stride_elems = None

    # Best-effort layout audit log; the log line itself must not raise.
    shape = tuple(rep.shape)
    stride = tuple(rep.stride())
    try:
        inner = 1
        for s in shape[1:]:
            inner *= int(s)
        padding_per_block = stride[0] - inner if stride else 0
    except Exception:
        padding_per_block = -1
    try:
        storage_nbytes = rep.untyped_storage().nbytes()
    except Exception:
        storage_nbytes = -1
    logger.info(
        "Group %d first-layer tensor: layer_idx=%d shape=%s "
        "stride=%s is_contiguous=%s dtype=%s device=%s "
        "storage_offset=%d numel=%d storage_nbytes=%d "
        "padding_per_block=%d",
        group_idx,
        layer_idx,
        shape,
        stride,
        rep.is_contiguous(),
        rep.dtype,
        rep.device,
        rep.storage_offset(),
        rep.numel(),
        storage_nbytes,
        padding_per_block,
    )

    return block_stride_elems


def make_page_buffer_shape_desc(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int,
    num_layers_in_group: int,
    num_blocks: int,
    block_size: int,
    block_stride_elems: Optional[int] = None,
) -> "lmc_ops.PageBufferShapeDesc":
    """Build a :class:`PageBufferShapeDesc` from a representative layer.

    Args:
        kv_caches: Full kv_caches structure.
        gpu_kv_format: Format returned by :func:`normalize_kv_and_discover_format`.
        layer_idx: 0-based index of the representative layer.
        num_layers_in_group: Number of layers in the group (``nl``).
        num_blocks: Number of paged blocks (``nb``).
        block_size: Tokens per block (``bs``).
        block_stride_elems: Physical per-block stride in *elements*
            (= ``tensor.stride(0)`` of the representative layer). Pass
            the real value whenever the group's KV pool may be
            dim-0-padded (e.g. DeepSeek V4 compressor/indexer caches
            sharing a row width with a larger group in the same pool);
            otherwise downstream transfer kernels will skip into
            padding and corrupt data. Leave as ``None`` for unpadded
            pools — the kernel's ``per_block_stride()`` fallback
            (block_stride_elems <= 0) will reconstruct the tight
            stride from ``kv_size`` and ``scalars_per_block`` itself,
            so we don't duplicate that arithmetic on the Python side.

    Returns:
        A populated ``PageBufferShapeDesc``.
    """
    desc = lmc_ops.PageBufferShapeDesc()
    desc.kv_size = 1 if is_mla(gpu_kv_format) else 2
    desc.nl = num_layers_in_group
    desc.nb = num_blocks
    desc.bs = block_size
    desc.nh = (
        1
        if is_mla(gpu_kv_format)
        else get_num_heads(kv_caches, gpu_kv_format, layer_idx)
    )
    desc.hs = get_head_size(kv_caches, gpu_kv_format, layer_idx)
    desc.element_size = get_dtype(kv_caches, gpu_kv_format, layer_idx).itemsize

    resolved_stride = int(block_stride_elems) if block_stride_elems else 0
    desc.block_stride_elems = resolved_stride
    return desc


def _split_token2d_kv(token2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accepts either:
      - [2, T, D]
      - [T, 2, D]
    Returns:
      - k_tok: [T, D]
      - v_tok: [T, D]
    """
    if token2d.dim() != 3:
        raise ValueError(f"Expected token2d dim=3, got {token2d.shape}")
    if token2d.shape[0] == 2:  # [2, T, D]
        return token2d[0], token2d[1]
    if token2d.shape[1] == 2:  # [T, 2, D]
        return token2d[:, 0, :], token2d[:, 1, :]
    raise ValueError(f"Unrecognized token2d layout: {token2d.shape}")


def _get_head_size_view(
    kv_cache_layer: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    *,
    use_mla: bool,
    gpu_kv_format: Optional["lmc_ops.GPUKVFormat"] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns flattened views for index_copy/index_select.

    If gpu_kv_format is provided, use it to interpret tensor layout explicitly.
    If not provided, fall back to current structural behavior:
      - MLA: expects Tensor [P, B, HS]
      - Non-MLA: expects either
          * Tensor [2, P, B, NH, HS]  OR
          * (k, v) tuple each [P, B, NH, HS]
        (and also supports [P, 2, B, NH, HS] as a safe extension)
    """
    # -------------------------
    # MLA
    # -------------------------
    if use_mla:
        if not isinstance(kv_cache_layer, torch.Tensor):
            raise ValueError("MLA expects kv_cache_layer as Tensor")
        if kv_cache_layer.dim() != 3:
            raise ValueError(f"MLA expects 3D [P,B,HS], got {kv_cache_layer.shape}")
        p, b, hs = kv_cache_layer.shape
        return kv_cache_layer.view(p * b, hs)

    # -------------------------
    # non-MLA (K/V)
    # -------------------------
    # If already provided (k, v) in canonical per-layer form, no format needed.
    if not isinstance(kv_cache_layer, torch.Tensor):
        k, v = kv_cache_layer
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError(f"Expected (k,v) 4D [P,B,NH,HS], got {k.shape}, {v.shape}")
        p, b, nh, hs = k.shape
        if v.shape != (p, b, nh, hs):
            raise ValueError(f"k/v shape mismatch: {k.shape} vs {v.shape}")
        return k.view(p * b, nh * hs), v.view(p * b, nh * hs)

    t = kv_cache_layer
    if t.dim() != 5:
        raise ValueError(f"Expected 5D tensor for non-MLA, got {t.shape}")

    # If we have the format enum, decode explicitly.
    if gpu_kv_format is not None:
        if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
            # per-layer: [2, NB, BS, NH, HS]
            if t.shape[0] != 2:
                raise ValueError(
                    f"{gpu_kv_format} expects [2,NB,BS,NH,HS], got {t.shape}"
                )
            k, v = t[0], t[1]  # [NB,BS,NH,HS]

        elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
            # per-layer: [NB, 2, BS, NH, HS]
            if t.shape[1] != 2:
                raise ValueError(
                    f"{gpu_kv_format} expects [NB,2,BS,NH,HS], got {t.shape}"
                )
            k, v = t[:, 0], t[:, 1]  # [NB,BS,NH,HS]

        else:
            # Other formats are either MLA-only or require upstream normalization.
            raise NotImplementedError(
                f"gpu_kv_format={gpu_kv_format} not supported in non-MLA path here. "
                "Normalize to (k,v) tuple [NB,BS,NH,HS] per-layer before calling."
            )

    else:
        # No enum available: Assumed [2,P,B,H,D] (or [2,NB,BS,NH,HS] per-layer).
        # Also accept [P,2,B,H,D] (or [NB,2,BS,NH,HS]) to be more robust.
        if t.shape[0] == 2:
            k, v = t[0], t[1]
        elif t.shape[1] == 2:
            k, v = t[:, 0], t[:, 1]
        else:
            raise ValueError(
                f"gpu_kv_format is None and tensor does not look like stacked KV. "
                f"Expected axis0==2 or axis1==2, got {t.shape}"
            )

    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"Expected k/v 4D [NB,BS,NH,HS], got {k.shape}, {v.shape}")

    nb, bs, nh, hs = k.shape
    if v.shape != (nb, bs, nh, hs):
        raise ValueError(f"k/v shape mismatch after decode: {k.shape} vs {v.shape}")

    return k.view(nb * bs, nh * hs), v.view(nb * bs, nh * hs)
