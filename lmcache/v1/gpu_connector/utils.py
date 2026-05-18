# SPDX-License-Identifier: Apache-2.0
# This module is the single layer that performs format-dispatched raw
# indexing on DiscoverableKVCache values (kv_caches.shape[i],
# kv_caches[0][j]); the gpu_kv_format argument is the proof the
# indexing is well-defined. Silence union-attr errors only for this
# file so the accessors can take DiscoverableKVCache without 50+
# per-line type: ignore comments.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    TypedDict,
    Union,
    cast,
)

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface

# First Party
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

# Canonical recursive type consumed by :func:`normalize_kv_and_discover_format`
# and the downstream format-aware helpers. A value is either a single
# :class:`torch.Tensor` (e.g. vLLM cross-layer, TRT-LLM) or a list of
# nested ``DiscoverableKVCache`` values (per-layer lists, SGLang's two-list
# MHA, deeper nesting). Engine adapters that hand us other containers
# (e.g. vLLM's ``dict[str, torch.Tensor]``) are responsible for unwrapping
# to this form before calling the helpers.
DiscoverableKVCache = Union[torch.Tensor, list["DiscoverableKVCache"]]

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


class LayoutHints(TypedDict, total=False):
    """Hints passed from a serving engine to LMCache during KV cache
    registration (``REGISTER_KV_CACHE``).

    Serving engines may pass a plain ``dict`` that satisfies this
    schema — importing this type is optional.

    Keys:
        kv_layout: Physical ordering of the KV cache dimensions.
            ``"NHD"`` — heads after block-size (default for most
            vLLM builds).
            ``"HND"`` — heads before block-size (``VLLM_KV_CACHE_LAYOUT=HND``).
        num_kv_heads: Number of KV heads per layer. Used by TRT-LLM to
            reshape its 4-D pool tensor into the canonical 6-D form.
        tokens_per_block: Tokens per paged block. Used by TRT-LLM (same).
        head_dim: Per-head dimension. Used by TRT-LLM (same).
        inference_engine_logical_block_size: Inference-engine-side block
            size (logical tokens per engine block; for vLLM this is
            ``cache_config.block_size``). Carried inside
            ``LayoutHints`` (instead of as a standalone
            ``REGISTER_KV_CACHE`` argument) so that engines without a
            logical block-size concept can simply omit it. The server
            uses it to derive per-group compression ratios when some
            KV layer groups compress multiple logical tokens into a
            single physical slot
            (``shape_desc.bs < inference_engine_logical_block_size``).
    """

    kv_layout: Literal["NHD", "HND"]
    num_kv_heads: int
    tokens_per_block: int
    head_dim: int
    inference_engine_logical_block_size: int


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
    return gpu_kv_format in (
        lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS,
    )


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
    _SHAPE_DESCRIPTIONS: dict["lmc_ops.GPUKVFormat", str] = {
        lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS: "[NB, NL, 2, BS, NH, HS]",
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS: "NL x [2, NB, BS, NH, HS]",
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS: "NL x [NB, 2, BS, NH, HS]",
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS: "NL x [NB, BS, HS]",
        lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS: "2 x NL x [PBS, NH, HS]",
        lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS: "NL x [PBS, 1, HS]",
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS: "NL x [2, NB, NH, BS, HS]",
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS: "NL x [NB, 2, NH, BS, HS]",
        lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS: "[NB, NL, 2, NH, BS, HS]",
    }
    return _SHAPE_DESCRIPTIONS.get(gpu_kv_format, f"Unknown ({gpu_kv_format})")


def get_attention_backend(gpu_kv_format: "lmc_ops.GPUKVFormat") -> str:
    """Return the attention backend name for the GPU KV format."""
    _ATTENTION_BACKENDS: dict["lmc_ops.GPUKVFormat", str] = {
        lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS: "vLLM CROSS_LAYER",
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS: "vLLM non-MLA flash attention",
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS: "vLLM non-MLA flash infer",
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS: "vLLM MLA",
        lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS: (
            "SGLang MHA (flash attention and flash infer)"
        ),
        lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS: "SGLang MLA",
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS: (
            "vLLM non-MLA flash attention (HND layout)"
        ),
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS: (
            "vLLM non-MLA flash infer (HND layout)"
        ),
        lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS: "TRT-LLM cross-layer (HND layout)",
    }
    return _ATTENTION_BACKENDS.get(gpu_kv_format, f"Unknown ({gpu_kv_format})")


def get_concrete_gpu_kv_shape(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> str:
    """Return the shape with actual numeric values substituted.

    For example, instead of ``NL x [2, NB, BS, NH, HS]``
    this returns ``80 x [2, 2048, 128, 8, 128]``.
    """
    nl = get_num_layers(kv_caches, gpu_kv_format)
    hs = get_head_size(kv_caches, gpu_kv_format)

    fmt = gpu_kv_format
    F = lmc_ops.GPUKVFormat

    if fmt == F.NB_NL_TWO_BS_NH_HS:
        nb = get_num_blocks(kv_caches, fmt)
        bs = get_block_size(kv_caches, fmt)
        nh = get_num_heads(kv_caches, fmt)
        return f"[{nb}, {nl}, 2, {bs}, {nh}, {hs}]"

    if fmt == F.NL_X_TWO_NB_BS_NH_HS:
        nb = get_num_blocks(kv_caches, fmt)
        bs = get_block_size(kv_caches, fmt)
        nh = get_num_heads(kv_caches, fmt)
        return f"{nl} x [2, {nb}, {bs}, {nh}, {hs}]"

    if fmt == F.NL_X_NB_TWO_BS_NH_HS:
        nb = get_num_blocks(kv_caches, fmt)
        bs = get_block_size(kv_caches, fmt)
        nh = get_num_heads(kv_caches, fmt)
        return f"{nl} x [{nb}, 2, {bs}, {nh}, {hs}]"

    if fmt == F.NL_X_NB_BS_HS:
        nb = get_num_blocks(kv_caches, fmt)
        bs = get_block_size(kv_caches, fmt)
        return f"{nl} x [{nb}, {bs}, {hs}]"

    if fmt == F.TWO_X_NL_X_NBBS_NH_HS:
        pbs = get_page_buffer_size(kv_caches, fmt)
        nh = get_num_heads(kv_caches, fmt)
        return f"2 x {nl} x [{pbs}, {nh}, {hs}]"

    if fmt == F.NL_X_NBBS_ONE_HS:
        pbs = get_page_buffer_size(kv_caches, fmt)
        return f"{nl} x [{pbs}, 1, {hs}]"

    if fmt == F.NL_X_TWO_NB_NH_BS_HS:
        nb = get_num_blocks(kv_caches, fmt)
        nh = get_num_heads(kv_caches, fmt)
        bs = get_block_size(kv_caches, fmt)
        return f"{nl} x [2, {nb}, {nh}, {bs}, {hs}]"

    if fmt == F.NL_X_NB_TWO_NH_BS_HS:
        nb = get_num_blocks(kv_caches, fmt)
        nh = get_num_heads(kv_caches, fmt)
        bs = get_block_size(kv_caches, fmt)
        return f"{nl} x [{nb}, 2, {nh}, {bs}, {hs}]"

    if fmt == F.NB_NL_TWO_NH_BS_HS:
        nb = get_num_blocks(kv_caches, fmt)
        nh = get_num_heads(kv_caches, fmt)
        bs = get_block_size(kv_caches, fmt)
        return f"[{nb}, {nl}, 2, {nh}, {bs}, {hs}]"

    return f"Unknown ({gpu_kv_format})"


def legible_print_gpu_kv_format(gpu_kv_format: "lmc_ops.GPUKVFormat"):
    """
    Print the GPU KV Format in a legible way
    """
    shape = get_gpu_kv_shape_description(gpu_kv_format)
    backend = get_attention_backend(gpu_kv_format)
    if shape.startswith("Unknown"):
        logger.warning(f"Unknown GPU KV Format: {gpu_kv_format}")
    else:
        logger.info("GPU KV Format: %s", shape)
        logger.info("Currently used by:\n  - %s", backend)


def _list_depth_tensor_dim(kv_caches: DiscoverableKVCache) -> tuple[int, int]:
    """Measure the structural shape of a :data:`DiscoverableKVCache`.

    Descends the first element of each list until a tensor is reached,
    counting list-wrapping layers along the way.

    Args:
        kv_caches: A :data:`DiscoverableKVCache` value.

    Returns:
        ``(list_depth, tensor_ndim)`` — the number of list-wrapping
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


def normalize_kv_and_discover_format(
    kv_caches: DiscoverableKVCache,
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> tuple["lmc_ops.GPUKVFormat", DiscoverableKVCache]:
    """
    Normalize ``kv_caches`` into the canonical form and discover its GPU KV format.

    Performs (in order):
      1. ``attempt_permute_to_contiguous_view``: stride-based dim
         permutation so ``.shape`` reflects the physical (not
         permuted-logical) layout — critical for HND vs NHD detection.
         No-op if already contiguous; raises for non-permutation sources
         of non-contiguity (slicing, ``as_strided``).
      2. Format detection by descending list-wrapping and inspecting the
         innermost tensor's shape.

    The logic is that "external" layers are lists and there is one tensor
    internally. We "unwrap" layers until we find the tensor.

    Args:
        kv_caches: The KV cache tensors (possibly nested lists of tensors).
        serving_engine: Which serving engine produced the caches.
        layout_hints: See :class:`~lmcache.v1.multiprocess.custom_types.LayoutHints`.

    Returns:
        ``(gpu_kv_format, normalized_kv_caches)``. Callers must use the
        returned tensor structure for subsequent operations — it shares
        storage with the input but may be a permuted view.

    Please see csrc/mem_kernels.cuh for the naming schema of the GPUKVFormat.
    """
    kv_caches = attempt_permute_to_contiguous_view(kv_caches)

    if layout_hints is None:
        layout_hints = {}

    # TRT-LLM hands us a 4-D pool tensor (possibly wrapped in a 1-element
    # list for adapter-side ergonomics). Reshape to the canonical 6-D
    # cross-layer form here so detection lands on the standard path.
    if serving_engine == EngineType.TRTLLM:
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
            kv_caches = kv_caches.view(
                nb, nl, kv, num_kv_heads, tokens_per_block, head_dim
            )

    # list_depth: number of external wrapping lists
    # tensor_dim: number of dimensions of the internal tensor
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

    detected_format = None

    if serving_engine == EngineType.TRTLLM:
        if list_depth == 0 and tensor_dim == 6:
            detected_format = lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS
    elif serving_engine == EngineType.VLLM:
        kv_layout = layout_hints.get("kv_layout")
        if kv_layout is None:
            logger.warning(
                "No KV Cache Layout hint provided when using vLLM, defaulting to NHD"
            )
            kv_layout = "NHD"
        logger.info("vLLM KV cache layout: %s", kv_layout)
        is_hnd = kv_layout == "HND"
        if list_depth == 0:
            # vllm cross layer
            detected_format = lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS
        elif list_depth == 1:
            if tensor_dim == 5:
                if probe.shape[0] == 2:
                    # vllm non-MLA flash attention
                    if is_hnd:
                        detected_format = lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS
                    else:
                        detected_format = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
                elif probe.shape[1] == 2:
                    # vllm non-MLA flash infer
                    if is_hnd:
                        detected_format = lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS
                    else:
                        detected_format = lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS
            elif tensor_dim == 3:
                # vllm MLA
                detected_format = lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
    elif serving_engine == EngineType.SGLANG:
        if list_depth == 1:
            if probe.shape[1] == 1:
                # sglang MLA
                detected_format = lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS
        elif list_depth == 2:
            # sglang MHA (flash attention and flash infer)
            detected_format = lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS

    if detected_format is not None:
        legible_print_gpu_kv_format(detected_format)
        return detected_format, kv_caches
    else:
        raise ValueError(
            "currently unsupported kv_caches format "
            f"with list depth {list_depth} and tensor dimension {tensor_dim}"
        )


def get_num_layers(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> int:
    """
    Get the number of layers from the kv_caches
    """
    if gpu_kv_format in (
        lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS,
    ):
        return kv_caches.shape[1]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        return len(kv_caches)
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return len(kv_caches[0])
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return len(kv_caches)
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_num_blocks(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> int:
    """
    Get the number of blocks from the kv_caches
    """
    if gpu_kv_format in (
        lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS,
    ):
        return kv_caches.shape[0]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        # [2, num_blocks, ...] — shape[1] is num_blocks
        return kv_caches[0].shape[1]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # [num_blocks, 2, ...] — shape[0] is num_blocks
        return kv_caches[0].shape[0]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[0].shape[0]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


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
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS:
        # HND cross-layer: [NB, NL, 2, NH, BS, HS] — block_size at shape[4]
        return kv_caches.shape[4]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ):
        # NHD: [..., BS, NH, HS] — block_size at shape[2]
        return kv_caches[layer_idx].shape[2]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # HND: [..., NH, BS, HS] — block_size at shape[3]
        return kv_caches[layer_idx].shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[layer_idx].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_page_buffer_size(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> int:
    """
    Get page buffer size (num_blocks * block_size) from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        # [num_blocks, num_layers, 2, block_size, num_heads, head_size]
        return kv_caches.shape[0] * kv_caches.shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS:
        # [num_blocks, num_layers, 2, num_heads, block_size, head_size]
        return kv_caches.shape[0] * kv_caches.shape[4]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        # list[num_layers] of [2, num_blocks, block_size, num_heads, head_size]
        return kv_caches[0].shape[1] * kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
        # list[num_layers] of [2, num_blocks, num_heads, block_size, head_size]
        # num_blocks=shape[1], block_size=shape[3]
        return kv_caches[0].shape[1] * kv_caches[0].shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        # list[num_layers] of [num_blocks, 2, block_size, num_heads, head_size]
        return kv_caches[0].shape[0] * kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS:
        # list[num_layers] of [num_blocks, 2, num_heads, block_size, head_size]
        # num_blocks=shape[0], block_size=shape[3]
        return kv_caches[0].shape[0] * kv_caches[0].shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # list[num_layers] of [num_blocks, block_size, head_size]
        return kv_caches[0].shape[0] * kv_caches[0].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        # list[2] -> list[num_layers] of [page_buffer_size, num_heads, head_size]
        return kv_caches[0][0].shape[0]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        # list[num_layers] of [page_buffer_size, 1, head_size]
        return kv_caches[0].shape[0]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_num_heads(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int = 0,
) -> int:
    """
    Get the number of heads for a layer (defaults to layer 0).
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[4]  # global for cross-layer
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS:
        # HND cross-layer: [NB, NL, 2, NH, BS, HS] — num_heads at shape[3]
        return kv_caches.shape[3]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ):
        # NHD: [..., BS, NH, HS] — num_heads at shape[3]
        return kv_caches[layer_idx].shape[3]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # HND: [..., NH, BS, HS] — num_heads at shape[2]
        return kv_caches[layer_idx].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # MLA: heads are absorbed into hidden dim, so num_heads = 1
        return 1
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][layer_idx].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return kv_caches[layer_idx].shape[1]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_hidden_dim_size(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int = 0,
) -> int:
    """
    Get the hidden dimension for a layer (defaults to layer 0).
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[4] * kv_caches.shape[5]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS:
        # HND cross-layer: [NB, NL, 2, NH, BS, HS] — hidden = NH * HS
        return kv_caches.shape[3] * kv_caches.shape[5]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ):
        # NHD: [..., NH, HS] — hidden_dim = shape[3] * shape[4]
        return kv_caches[layer_idx].shape[3] * kv_caches[layer_idx].shape[4]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # HND: [..., NH, BS, HS] — hidden_dim = NH * HS = shape[2] * shape[4]
        return kv_caches[layer_idx].shape[2] * kv_caches[layer_idx].shape[4]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[layer_idx].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][layer_idx].shape[1] * kv_caches[0][layer_idx].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return kv_caches[layer_idx].shape[2]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_head_size(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int = 0,
) -> int:
    """
    Get the head size for a layer (defaults to layer 0).
    """
    if gpu_kv_format in (
        lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS,
    ):
        return kv_caches.shape[5]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # Both NHD [..., NH, HS] and HND [..., BS, HS] have head_size last
        return kv_caches[layer_idx].shape[4]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[layer_idx].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][layer_idx].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return kv_caches[layer_idx].shape[2]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_tokens_per_layer(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> int:
    """
    Get the number of tokens per layer from the kv_caches
    (num_blocks * block_size or page_buffer_size)
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        # [num_blocks, num_layers, 2, block_size, num_heads, head_size]
        return kv_caches.shape[0] * kv_caches.shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS:
        # [num_blocks, num_layers, 2, num_heads, block_size, head_size]
        return kv_caches.shape[0] * kv_caches.shape[4]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        # list[num_layers] of [2, num_blocks, block_size, num_heads, head_size]
        k_cache_shape = kv_caches[0][0].shape
        return k_cache_shape[0] * k_cache_shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
        # list[num_layers] of [2, num_blocks, num_heads, block_size, head_size]
        # k_cache = kv_caches[0][0] → (NB, NH, BS, HS); tokens = NB * BS
        k_cache_shape = kv_caches[0][0].shape
        return k_cache_shape[0] * k_cache_shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        # list[num_layers] of [num_blocks, 2, block_size, num_heads, head_size]
        k_cache_shape = kv_caches[0][:, 0].shape
        return k_cache_shape[0] * k_cache_shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS:
        # list[num_layers] of [num_blocks, 2, num_heads, block_size, head_size]
        # k_cache = kv_caches[0][:, 0] → (NB, NH, BS, HS); tokens = NB * BS
        k_cache_shape = kv_caches[0][:, 0].shape
        return k_cache_shape[0] * k_cache_shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # list[num_layers] of [num_blocks, block_size, head_size]
        return kv_caches[0].shape[0] * kv_caches[0].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        # list[2] -> list[num_layers] of [page_buffer_size, num_heads, head_size]
        return kv_caches[0][0].shape[0]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        # list[num_layers] of [page_buffer_size, 1, head_size]
        return kv_caches[0].shape[0]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_elements_per_layer(
    kv_caches: DiscoverableKVCache, gpu_kv_format: "lmc_ops.GPUKVFormat"
) -> int:
    """
    Get the number of elements per layer from the kv_caches
    (including both K and V for non-MLA)
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        # [num_blocks, num_layers, 2, block_size, num_heads, head_size]
        # For one layer: [num_blocks, 2, block_size, num_heads, head_size]
        num_blocks = kv_caches.shape[0]
        block_size = kv_caches.shape[3]
        num_heads = kv_caches.shape[4]
        head_size = kv_caches.shape[5]
        return num_blocks * 2 * block_size * num_heads * head_size
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS:
        # [num_blocks, num_layers, 2, num_heads, block_size, head_size]
        num_blocks = kv_caches.shape[0]
        num_heads = kv_caches.shape[3]
        block_size = kv_caches.shape[4]
        head_size = kv_caches.shape[5]
        return num_blocks * 2 * num_heads * block_size * head_size
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        # [2, num_blocks, ...] — k_cache is kv_caches[0][0]
        k_cache_shape = kv_caches[0][0].shape
        return k_cache_shape.numel() * 2
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # [num_blocks, 2, ...] — k_cache is kv_caches[0][:, 0]
        k_cache_shape = kv_caches[0][:, 0].shape
        return k_cache_shape.numel() * 2
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # list[num_layers] of [num_blocks, block_size, head_size] (MLA)
        return kv_caches[0].numel()
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        # list[2] -> list[num_layers] of
        # [page_buffer_size, num_heads, head_size] (separate K and V)
        return kv_caches[0][0].numel() * 2
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        # list[num_layers] of [page_buffer_size, 1, head_size] (MLA)
        return kv_caches[0].numel()
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def assert_is_vllm_flash_attn_or_flash_infer(gpu_kv_format: "lmc_ops.GPUKVFormat"):
    """
    Ensure that we have a GPU KV Cache Format
    that is either vLLM's flash attention or flash infer.
    """
    assert gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    )


def is_hnd(gpu_kv_format: "lmc_ops.GPUKVFormat") -> bool:
    """
    Check if the GPU KV Format uses HND physical layout
    """
    return gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
        lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS,
    )


def assert_is_vllm_mla_or_flash_attn_or_flash_infer(
    gpu_kv_format: "lmc_ops.GPUKVFormat",
) -> None:
    """
    Ensure that we have a GPU KV Cache Format that is either
    vLLM's MLA, flash attention, or flash infer.

    Accepted formats:
        - ``NL_X_TWO_NB_BS_NH_HS`` (flash attention, NHD)
        - ``NL_X_NB_TWO_BS_NH_HS`` (flash infer, NHD)
        - ``NL_X_TWO_NB_NH_BS_HS`` (flash attention, HND)
        - ``NL_X_NB_TWO_NH_BS_HS`` (flash infer, HND)
        - ``NL_X_NB_BS_HS`` (MLA)

    Raises:
        AssertionError: If *gpu_kv_format* is not one of the accepted formats.
    """
    assert gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
    )


def is_mla(gpu_kv_format: "lmc_ops.GPUKVFormat") -> bool:
    """
    Check if the GPU KV Format is MLA
    """
    return (
        gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS  # vllm MLA
        or gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS  # sglang MLA
    )


def get_dtype(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_idx: int = 0,
) -> torch.dtype:
    """
    Get the dtype for a layer (defaults to layer 0).
    """
    if gpu_kv_format in (
        lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS,
    ):
        return kv_caches.dtype
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
    ):
        return kv_caches[layer_idx].dtype
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][layer_idx].dtype
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_group_data_ptrs(
    kv_caches: DiscoverableKVCache,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    layer_indices: list[int],
) -> list[int]:
    """Return device pointers for a group of layers in the order the transfer
    kernels expect for *gpu_kv_format*.

    The pointer array's *shape* is a property of the format, not of the
    caller. Three buckets, mirroring the kernel dispatch in
    ``csrc/mp_mem_kernels.cu:160-169``:

    - Per-layer list formats: ``[p_{i0}, p_{i1}, ..., p_{iN}]`` — one
      pointer per requested layer, in the given order.
    - ``TWO_X_NL_X_NBBS_NH_HS`` (SGLang MHA): K's grouped first,
      then V's: ``[K_{i0}, ..., K_{iN}, V_{i0}, ..., V_{iN}]``.
    - ``NB_NL_TWO_BS_NH_HS`` / ``NB_NL_TWO_NH_BS_HS`` (cross-layer): a
      single base pointer, ``[base]``. The kernel walks layers by
      computing offsets from ``shape_desc.nl`` internally;
      ``layer_indices`` is unused.

    Args:
        kv_caches: Full kv_caches structure.
        gpu_kv_format: Format returned by :func:`normalize_kv_and_discover_format`.
        layer_indices: 0-based layer indices in the group, in the order
            the kernel should iterate them. Ignored for cross-layer.

    Returns:
        Device pointers (int), in kernel-expected order.

    Raises:
        ValueError: If *gpu_kv_format* is not recognized.
    """
    F = lmc_ops.GPUKVFormat
    if gpu_kv_format in (F.NB_NL_TWO_BS_NH_HS, F.NB_NL_TWO_NH_BS_HS):
        tensor = cast(torch.Tensor, kv_caches)
        return [tensor.data_ptr()]
    if gpu_kv_format == F.TWO_X_NL_X_NBBS_NH_HS:
        k, v = cast(list[list[torch.Tensor]], kv_caches)
        return [k[i].data_ptr() for i in layer_indices] + [
            v[i].data_ptr() for i in layer_indices
        ]
    if gpu_kv_format in (
        F.NL_X_TWO_NB_BS_NH_HS,
        F.NL_X_NB_TWO_BS_NH_HS,
        F.NL_X_TWO_NB_NH_BS_HS,
        F.NL_X_NB_TWO_NH_BS_HS,
        F.NL_X_NB_BS_HS,
        F.NL_X_NBBS_ONE_HS,
    ):
        layers = cast(list[torch.Tensor], kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]
    raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


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
_BLOCK_AXIS_FORMATS: frozenset = frozenset(
    {
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
    }
)


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

    def _pick_layout_probe_tensor() -> torch.Tensor:
        # Layout probe only (shape/stride/storage_offset/dtype); not
        # the K/V slice fed to the transfer kernel.
        # - Cross-layer formats: ``kv_caches`` is the single backing
        #   tensor packing all layers along dim-1; indexing dim-0
        #   (= NB) would yield a per-block slice, so return the whole
        #   tensor (its ``stride(0)`` is the authoritative per-NB step).
        # - SGL MHA: outer list is K/V (length 2), inner list is
        #   per-layer; K & V share shape/stride by construction.
        # - Other formats: ``kv_caches`` is already a per-layer list.
        if gpu_kv_format in (
            lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS,
            lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS,
        ):
            if not isinstance(kv_caches, torch.Tensor):
                raise TypeError(
                    "Cross-layer GPUKVFormat expects a single backing "
                    f"torch.Tensor, got {type(kv_caches).__name__}."
                )
            return kv_caches
        if gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
            return kv_caches[0][layer_idx]  # type: ignore[index]
        return kv_caches[layer_idx]  # type: ignore[index]

    rep = _pick_layout_probe_tensor()

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
