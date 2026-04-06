# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    overload,
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

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops
else:
    # First Party
    import lmcache.non_cuda_equivalents as lmc_ops

logger = init_logger(__name__)

# Error message for accessing non-existent attributes in GPU KV Cache
_ATTRIBUTE_NOT_EXIST_ERROR = "trying to access an attribute of the GPU KV Cache "
"that does not exist for the format detected {format}. "
"A misalignment with the GPUKVFormat must be resolved"


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
    """

    kv_layout: Literal["NHD", "HND"]


def permute_to_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Permute a tensor back to contiguous state (metadata-only, no copy).

    Assumption: the tensor is only non-contiguous because of a previous
    permutation.  Raises if this assumption is not met.

    The only known case is HND: vLLM allocates physically as HND
    (e.g. [2, NB, NH, BS, HS]) but exposes an NHD logical view via
    permute/transpose.  This function recovers the underlying HND
    physical shape so that ``discover_gpu_kv_format`` (with the
    ``kv_layout="HND"`` hint) sees the true memory layout and selects
    the correct HND format enum.  The reverse — physically NHD but
    logically permuted to HND — does not occur in practice.

    Returns the tensor unchanged if already contiguous.
    """
    if tensor.is_contiguous():
        return tensor

    strides = tensor.stride()
    perm = sorted(range(tensor.ndim), key=lambda i: strides[i], reverse=True)
    result = tensor.permute(perm)

    if not result.is_contiguous():
        raise ValueError(
            "tensor is non-contiguous for reasons other than permutation "
            "(e.g., slicing or as_strided). Cannot recover contiguous view."
        )
    return result


def permute_kv_caches_to_contiguous(
    kv_caches: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Apply :func:`permute_to_contiguous` to each tensor in *kv_caches*.

    The returned list shares the same underlying storage as the input.
    """
    return [permute_to_contiguous(t) for t in kv_caches]


def assert_contiguous(tensor: torch.Tensor) -> None:
    """Assert that a tensor has a contiguous physical layout with zero offset.

    LMCache transfer kernels assume logical and physical views match
    for coalesced memory accesses. Do NOT blindly call ``.contiguous()``
    or ``.permute()`` to fix failures here — identify the root cause.
    """
    assert tensor.storage_offset() == 0, (
        f"expected storage_offset 0, got {tensor.storage_offset()}"
    )
    assert tensor.is_contiguous(), "tensor is not contiguous"


def any_non_contiguous(kv_caches: dict[str, torch.Tensor] | List[torch.Tensor]) -> bool:
    """Return True if any tensor in *kv_caches* is non-contiguous."""
    tensors = kv_caches.values() if isinstance(kv_caches, dict) else kv_caches
    return not all(t.is_contiguous() for t in tensors)


@overload
def ensure_contiguous_kv_caches(
    kv_caches: dict[str, torch.Tensor],
    kv_layout: str | None = None,
) -> dict[str, torch.Tensor]: ...


@overload
def ensure_contiguous_kv_caches(
    kv_caches: List[torch.Tensor],
    kv_layout: str | None = None,
) -> List[torch.Tensor]: ...


def ensure_contiguous_kv_caches(
    kv_caches: dict[str, torch.Tensor] | List[torch.Tensor],
    kv_layout: str | None = None,
) -> dict[str, torch.Tensor] | List[torch.Tensor]:
    """Permute non-contiguous KV caches to contiguous physical shape.

    LMCache assumes tensors have matching logical and physical views.
    Known reasons for non-contiguity: HND format from vLLM.

    Accepts both ``dict`` and ``list`` forms.
    Returns *kv_caches* unchanged if already contiguous.
    """
    if not any_non_contiguous(kv_caches):
        return kv_caches

    if isinstance(kv_caches, dict):
        result: dict[str, torch.Tensor] | List[torch.Tensor] = dict(
            zip(
                kv_caches.keys(),
                permute_kv_caches_to_contiguous(list(kv_caches.values())),
                strict=False,
            )
        )
    else:
        result = permute_kv_caches_to_contiguous(kv_caches)

    if kv_layout == "HND":
        logger.info("Permuted HND tensors to contiguous physical shape")
    else:
        logger.warning(
            "Non-contiguous KV tensors detected with layout=%s; "
            "permuted to contiguous. Please identify the underlying reason.",
            kv_layout,
        )

    return result


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
    from lmcache.v1.gpu_connector.gpu_connectors import (
        SGLangLayerwiseGPUConnector,
        VLLMBufferLayerwiseGPUConnector,
        VLLMPagedMemLayerwiseGPUConnector,
    )
    from lmcache.v1.gpu_connector.xpu_connectors import (
        VLLMPagedMemLayerwiseXPUConnector,
    )

    assert isinstance(
        gpu_connector,
        (
            VLLMPagedMemLayerwiseGPUConnector,
            VLLMBufferLayerwiseGPUConnector,
            SGLangLayerwiseGPUConnector,
            VLLMPagedMemLayerwiseXPUConnector,
        ),
    )


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
    }
    return _ATTENTION_BACKENDS.get(gpu_kv_format, f"Unknown ({gpu_kv_format})")


def get_concrete_gpu_kv_shape(
    kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat"
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


def _list_depth_tensor_dim(kv_caches: Any) -> Tuple[int, int]:
    """
    Get the number of external wrapping lists in the kv_caches.

    Assumption: kv_caches is of the form
    List[List[...List[torch.Tensor]]]
    """
    depth = 0
    while isinstance(kv_caches, list):
        depth += 1
        if not kv_caches:
            raise ValueError("encountered an empty list")
        kv_caches = kv_caches[0]
    if not isinstance(kv_caches, torch.Tensor):
        raise ValueError("encountered a non-tensor inside")
    return depth, kv_caches.ndim


def discover_gpu_kv_format(
    kv_caches: Any,
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> "lmc_ops.GPUKVFormat":
    """
    Discover the GPU KV Cache Format from the kv_caches.

    KEY: the logical view and physical views of the kv_caches should be made consistent
    BEFORE format discovery

    The logic is that "external" layers are lists and there is one tensor internally.
    We "unwrap" layers until we find the tensor.

    Args:
        kv_caches: The KV cache tensors (possibly nested lists of tensors).
        serving_engine: Which serving engine produced the caches.
        layout_hints: See :class:`~lmcache.v1.multiprocess.custom_types.LayoutHints`.

    Please see csrc/mem_kernels.cuh for the naming schema of the GPUKVFormat.
    """
    # list_depth: number of external wrapping lists
    # tensor_dim: number of dimensions of the internal tensor
    list_depth, tensor_dim = _list_depth_tensor_dim(kv_caches)
    logger.info("list_depth: %d, tensor_dim: %d", list_depth, tensor_dim)
    list_dims = []
    ptr = kv_caches
    for _ in range(list_depth):
        list_dims.append(len(ptr))
        ptr = ptr[0]
    # ptr is now the tensor
    assert_contiguous(ptr)

    tensor_dims = list(ptr.shape)
    dims_str = (
        "".join(f"[{d}]" for d in list_dims) + f"[{', '.join(map(str, tensor_dims))}]"
    )
    logger.info("GPU KV Cache Dimensions: %s", dims_str)

    if layout_hints is None:
        layout_hints = {}

    detected_format = None

    if serving_engine == EngineType.VLLM:
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
                if kv_caches[0].shape[0] == 2:
                    # vllm non-MLA flash attention
                    if is_hnd:
                        detected_format = lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS
                    else:
                        detected_format = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
                elif kv_caches[0].shape[1] == 2:
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
            if kv_caches[0].shape[1] == 1:
                # sglang MLA
                detected_format = lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS
        elif list_depth == 2:
            # sglang MHA (flash attention and flash infer)
            detected_format = lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS

    if detected_format is not None:
        legible_print_gpu_kv_format(detected_format)
        return detected_format
    else:
        raise ValueError(
            "currently unsupported kv_caches format "
            f"with list depth {list_depth} and tensor dimension {tensor_dim}"
        )


def get_num_layers(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the number of layers from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
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


def get_num_blocks(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the number of blocks from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
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


def get_block_size(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the block size from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[3]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ):
        # NHD: [..., BS, NH, HS] — block_size at shape[2]
        return kv_caches[0].shape[2]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # HND: [..., NH, BS, HS] — block_size at shape[3]
        return kv_caches[0].shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[0].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_page_buffer_size(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get page buffer size (num_blocks * block_size) from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        # [num_blocks, num_layers, 2, block_size, num_heads, head_size]
        return kv_caches.shape[0] * kv_caches.shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        # List[num_layers] of [2, num_blocks, block_size, num_heads, head_size]
        return kv_caches[0].shape[1] * kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
        # List[num_layers] of [2, num_blocks, num_heads, block_size, head_size]
        # num_blocks=shape[1], block_size=shape[3]
        return kv_caches[0].shape[1] * kv_caches[0].shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        # List[num_layers] of [num_blocks, 2, block_size, num_heads, head_size]
        return kv_caches[0].shape[0] * kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS:
        # List[num_layers] of [num_blocks, 2, num_heads, block_size, head_size]
        # num_blocks=shape[0], block_size=shape[3]
        return kv_caches[0].shape[0] * kv_caches[0].shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # List[num_layers] of [num_blocks, block_size, head_size]
        return kv_caches[0].shape[0] * kv_caches[0].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        # List[2] -> List[num_layers] of [page_buffer_size, num_heads, head_size]
        return kv_caches[0][0].shape[0]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        # List[num_layers] of [page_buffer_size, 1, head_size]
        return kv_caches[0].shape[0]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_num_heads(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the number of heads from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[4]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ):
        # NHD: [..., BS, NH, HS] — num_heads at shape[3]
        return kv_caches[0].shape[3]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # HND: [..., NH, BS, HS] — num_heads at shape[2]
        return kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # MLA: heads are absorbed into hidden dim, so num_heads = 1
        return 1
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][0].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return kv_caches[0].shape[1]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_hidden_dim_size(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the hidden dimension from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[4] * kv_caches.shape[5]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ):
        # NHD: [..., NH, HS] — hidden_dim = shape[3] * shape[4]
        return kv_caches[0].shape[3] * kv_caches[0].shape[4]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # HND: [..., NH, BS, HS] — hidden_dim = NH * HS = shape[2] * shape[4]
        return kv_caches[0].shape[2] * kv_caches[0].shape[4]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][0].shape[1] * kv_caches[0][0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return kv_caches[0].shape[2]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_head_size(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the head size from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[5]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # Both NHD [..., NH, HS] and HND [..., BS, HS] have head_size last
        return kv_caches[0].shape[4]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return kv_caches[0].shape[2]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_tokens_per_layer(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the number of tokens per layer from the kv_caches
    (num_blocks * block_size or page_buffer_size)
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        # [num_blocks, num_layers, 2, block_size, num_heads, head_size]
        return kv_caches.shape[0] * kv_caches.shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        # List[num_layers] of [2, num_blocks, block_size, num_heads, head_size]
        k_cache_shape = kv_caches[0][0].shape
        return k_cache_shape[0] * k_cache_shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
        # List[num_layers] of [2, num_blocks, num_heads, block_size, head_size]
        # k_cache = kv_caches[0][0] → (NB, NH, BS, HS); tokens = NB * BS
        k_cache_shape = kv_caches[0][0].shape
        return k_cache_shape[0] * k_cache_shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        # List[num_layers] of [num_blocks, 2, block_size, num_heads, head_size]
        k_cache_shape = kv_caches[0][:, 0].shape
        return k_cache_shape[0] * k_cache_shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS:
        # List[num_layers] of [num_blocks, 2, num_heads, block_size, head_size]
        # k_cache = kv_caches[0][:, 0] → (NB, NH, BS, HS); tokens = NB * BS
        k_cache_shape = kv_caches[0][:, 0].shape
        return k_cache_shape[0] * k_cache_shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # List[num_layers] of [num_blocks, block_size, head_size]
        return kv_caches[0].shape[0] * kv_caches[0].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        # List[2] -> List[num_layers] of [page_buffer_size, num_heads, head_size]
        return kv_caches[0][0].shape[0]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        # List[num_layers] of [page_buffer_size, 1, head_size]
        return kv_caches[0].shape[0]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_elements_per_layer(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
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
        # List[num_layers] of [num_blocks, block_size, head_size] (MLA)
        return kv_caches[0].numel()
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        # List[2] -> List[num_layers] of
        # [page_buffer_size, num_heads, head_size] (separate K and V)
        return kv_caches[0][0].numel() * 2
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        # List[num_layers] of [page_buffer_size, 1, head_size] (MLA)
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
    )


def is_mla(gpu_kv_format: "lmc_ops.GPUKVFormat") -> bool:
    """
    Check if the GPU KV Format is MLA
    """
    return (
        gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS  # vllm MLA
        or gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS  # sglang MLA
    )


def get_dtype(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> torch.dtype:
    """
    Get the dtype from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.dtype
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
    ):
        return kv_caches[0].dtype
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][0].dtype
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def _split_token2d_kv(token2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    kv_cache_layer: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    *,
    use_mla: bool,
    gpu_kv_format: Optional["lmc_ops.GPUKVFormat"] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
