# SPDX-License-Identifier: Apache-2.0


# Third Party
import torch

# First Party
from lmcache.lmcache_native import (
    EngineKVFormat,
    TransferDirection,
    is_cross_layer,
    is_kv_list,
    is_mla,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.torch_ops._kv_format import (
    _is_fused_kv_format,
    _is_hnd_format,
    _is_kv_second_tuple_format,
    _is_pbs_fused_format,
    _is_two_major_format,
)
from lmcache.v1.platform.torch_ops._tensor_from_ptr import _tensor_from_ptr

_ELEMENT_SIZE_TO_DTYPE: dict[int, torch.dtype] = {
    # Maps the byte width of a KV-cache element to a representative torch dtype.
    # Only widths that commonly appear in KV caches are listed; 1-byte entries
    # are treated as uint8 (raw bytes), 2-byte as float16, 4-byte as float32.
    # Note: bfloat16 also has element_size == 2 but cannot be distinguished here;
    # callers that need exact dtype should supply it explicitly.
    1: torch.uint8,
    2: torch.float16,
    4: torch.float32,
}


def _is_ptr_tensor(x: object) -> bool:
    """Return True when *x* is a 1-D pointer tensor (int64 or uint64)."""
    return (
        isinstance(x, torch.Tensor)
        and x.dtype in (torch.int64, torch.uint64)
        and x.ndim == 1
    )


def _per_layer_paged_shape(
    engine_kv_format: EngineKVFormat,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
) -> tuple[int, ...]:
    """Return the logical shape of a single per-layer paged buffer tensor.

    Args:
        engine_kv_format: The format enum that describes how K/V tokens are laid out.
        nb: Number of blocks in the paged buffer (``shape_desc.nb``).
        bs: Tokens per block / block size (``shape_desc.bs``).
        nh: Number of attention heads (``shape_desc.nh``).
        hs: Per-head hidden size (``shape_desc.hs``).

    Returns:
        A tuple representing the shape needed to reconstruct one layer's tensor
        from a raw pointer via :func:`_tensor_from_ptr`.
    """
    fmt = int(engine_kv_format)
    if fmt == int(EngineKVFormat.NL_X_NBBS_ONE_HS):
        return (nb * bs, 1, hs)
    if fmt == int(EngineKVFormat.NL_X_NB_BS_HS):
        return (nb, bs, hs)
    if fmt == int(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS):
        return (2, nb, nh, bs, hs)
    if fmt == int(EngineKVFormat.NL_X_NB_TWO_NH_BS_HS):
        return (nb, 2, nh, bs, hs)
    if fmt in (
        int(EngineKVFormat.NL_X_NB_NH_BS_TWO_HS),
        int(EngineKVFormat.NL_X_NB_NH_BS_CS),
    ):
        # Blocks-first fused KV (HND): the desc's hs is the packed
        # 2 * head_size, so each layer is the raw [NB, NH, BS, 2 * HS].
        return (nb, nh, bs, hs)
    if fmt in (
        int(EngineKVFormat.NL_X_NB_BS_NH_TWO_HS),
        int(EngineKVFormat.NL_X_NB_BS_NH_CS),
    ):
        # Blocks-first fused KV (NHD): tokens before heads.
        return (nb, bs, nh, hs)
    if fmt == int(EngineKVFormat.NL_X_TWO_NB_BS_NH_HS):
        return (2, nb, bs, nh, hs)
    # Covers NL_X_NB_TWO_BS_NH_HS and any future NHD variants.
    return (nb, 2, bs, nh, hs)


def _infer_kv_dtype(
    paged_buffer_ptrs_tensor: object,
    lmcache_objects_ptrs: object,
    shape_desc: "PageBufferShapeDesc",
) -> torch.dtype:
    """Infer the KV element dtype from whichever inputs carry it.

    Inference order (first match wins):
    1. ``shape_desc.dtype`` — authoritative when set; correctly distinguishes
       float16 vs bfloat16 which share ``element_size == 2``).
    2. ``paged_buffer_ptrs_tensor`` — if it is a non-pointer tensor or a list
       of tensors (including nested SGLang MHA lists), the dtype of the first
       tensor is used.
    3. ``lmcache_objects_ptrs`` — if it is a list of tensors, the dtype of the
       first chunk tensor is used.
    4. ``shape_desc.element_size`` — looked up in :data:`_ELEMENT_SIZE_TO_DTYPE`
       (ambiguous for 2-byte types; kept only as last-resort fallback).
    5. ``torch.bfloat16`` — silent default when no other source is available.
    """
    # Prefer shape_desc.dtype — it is exact and avoids the element_size ambiguity.
    if shape_desc is not None:
        sd_dtype = getattr(shape_desc, "dtype", None)
        if sd_dtype is not None:
            return sd_dtype
    if isinstance(paged_buffer_ptrs_tensor, list) and paged_buffer_ptrs_tensor:
        first = paged_buffer_ptrs_tensor[0]
        if isinstance(first, list) and first and isinstance(first[0], torch.Tensor):
            return first[0].dtype
        if isinstance(first, torch.Tensor):
            return first.dtype
    if isinstance(paged_buffer_ptrs_tensor, torch.Tensor) and not _is_ptr_tensor(
        paged_buffer_ptrs_tensor
    ):
        return paged_buffer_ptrs_tensor.dtype
    if isinstance(lmcache_objects_ptrs, list) and lmcache_objects_ptrs:
        if isinstance(lmcache_objects_ptrs[0], torch.Tensor):
            return lmcache_objects_ptrs[0].dtype
    if shape_desc is not None and shape_desc.element_size > 0:
        dtype = _ELEMENT_SIZE_TO_DTYPE.get(shape_desc.element_size)
        if dtype is None:
            raise ValueError(
                f"Unsupported element_size {shape_desc.element_size!r} in "
                "shape_desc; cannot infer KV dtype. "
                f"Supported sizes: {sorted(_ELEMENT_SIZE_TO_DTYPE)}"
            )
        return dtype
    return torch.bfloat16


def _normalize_paged_layers(
    paged_buffer_ptrs_tensor: "torch.Tensor | list",
    engine_kv_format: EngineKVFormat,
    shape_desc: "PageBufferShapeDesc | None" = None,
    device: "torch.device | str | None" = None,
    dtype: "torch.dtype | None" = None,
) -> "torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]":
    """Normalize paged buffer input based on GPU KV format.

    Accepts either tensor-form inputs (list / Tensor) or a 1-D pointer tensor
    (int64 / uint64).  When a pointer tensor is provided *shape_desc*, *device*,
    and *dtype* must be supplied so the tensors can be reconstructed via
    :func:`_tensor_from_ptr`.

    Returns:
        - Single ``torch.Tensor`` for cross-layer formats.
        - ``list[list[torch.Tensor]]`` (2 x NL) for SGLang MHA formats.
        - ``list[(torch.Tensor, torch.Tensor)]`` (NL ``(K, V)`` pairs) for the
          per-layer tuple format (``NL_X_TWO_X_NB_BS_NH_HS``).
        - ``list[torch.Tensor]`` (per-layer) for all other formats.
    """
    if is_cross_layer(engine_kv_format):
        if isinstance(paged_buffer_ptrs_tensor, torch.Tensor):
            if _is_ptr_tensor(paged_buffer_ptrs_tensor):
                # 1-D pointer tensor with a single entry → reconstruct full tensor.
                if shape_desc is None or device is None or dtype is None:
                    raise ValueError(
                        "_normalize_paged_layers: shape_desc, device, and dtype are "
                        "required when paged_buffer_ptrs_tensor is a pointer tensor"
                    )
                nb = int(shape_desc.nb)
                nl = int(shape_desc.nl)
                bs = int(shape_desc.bs)
                nh = int(shape_desc.nh)
                hs = int(shape_desc.hs)
                if _is_hnd_format(engine_kv_format):
                    shape: tuple[int, ...] = (nb, nl, 2, nh, bs, hs)
                else:
                    shape = (nb, nl, 2, bs, nh, hs)
                ptr = int(paged_buffer_ptrs_tensor[0].item())
                return _tensor_from_ptr(ptr, shape, dtype, device)
            return paged_buffer_ptrs_tensor
        raise TypeError(
            "Cross-layer formats require a single torch.Tensor input; "
            "got: " + type(paged_buffer_ptrs_tensor).__name__
        )
    if is_kv_list(engine_kv_format):
        if _is_ptr_tensor(paged_buffer_ptrs_tensor):
            # 1-D pointer tensor [K_L0,...,K_LN-1, V_L0,...,V_LN-1] → nested list.
            if shape_desc is None or device is None or dtype is None:
                raise ValueError(
                    "_normalize_paged_layers: shape_desc, device, and dtype are "
                    "required when paged_buffer_ptrs_tensor is a pointer tensor"
                )
            nb = int(shape_desc.nb)
            nl = int(shape_desc.nl)
            bs = int(shape_desc.bs)
            nh = int(shape_desc.nh)
            hs = int(shape_desc.hs)
            is_flat = _is_pbs_fused_format(engine_kv_format)
            per_layer_shape: tuple[int, ...] = (
                (nb * bs, nh, hs) if is_flat else (nb, bs, nh, hs)
            )
            ptrs = [int(p.item()) for p in paged_buffer_ptrs_tensor]
            k_tensors = [
                _tensor_from_ptr(ptrs[i], per_layer_shape, dtype, device)
                for i in range(nl)
            ]
            v_tensors = [
                _tensor_from_ptr(ptrs[nl + i], per_layer_shape, dtype, device)
                for i in range(nl)
            ]
            return [k_tensors, v_tensors]
        if isinstance(paged_buffer_ptrs_tensor, list):
            # Already nested [[K tensors], [V tensors]]
            if (
                len(paged_buffer_ptrs_tensor) == 2
                and isinstance(paged_buffer_ptrs_tensor[0], list)
                and all(
                    isinstance(t, torch.Tensor)
                    for group in paged_buffer_ptrs_tensor
                    for t in group
                )
            ):
                return paged_buffer_ptrs_tensor
            # Flat list [K_L0, ..., K_LN-1, V_L0, ..., V_LN-1]
            if all(isinstance(t, torch.Tensor) for t in paged_buffer_ptrs_tensor):
                if len(paged_buffer_ptrs_tensor) % 2 != 0:
                    raise ValueError(
                        "Flat SGLang MHA list must have even length (2*NL)"
                    )
                half = len(paged_buffer_ptrs_tensor) // 2
                return [
                    paged_buffer_ptrs_tensor[:half],
                    paged_buffer_ptrs_tensor[half:],
                ]
        raise TypeError(
            "SGLang MHA formats require a list[list[torch.Tensor]], a flat "
            "list[torch.Tensor] (2*NL entries), or a 1-D pointer tensor; "
            "got: " + type(paged_buffer_ptrs_tensor).__name__
        )
    if _is_kv_second_tuple_format(engine_kv_format):
        if isinstance(paged_buffer_ptrs_tensor, list) and all(
            isinstance(t, (list, tuple))
            and len(t) == 2
            and all(isinstance(x, torch.Tensor) for x in t)
            for t in paged_buffer_ptrs_tensor
        ):
            return paged_buffer_ptrs_tensor
        raise TypeError(
            "Per-layer (K, V) tuple format requires a list of (K, V) tensor "
            "pairs; got: " + type(paged_buffer_ptrs_tensor).__name__
        )
    # Per-layer formats
    if _is_ptr_tensor(paged_buffer_ptrs_tensor):
        # 1-D pointer tensor [ptr_L0, ..., ptr_LN-1] → list of per-layer tensors.
        if shape_desc is None or device is None or dtype is None:
            raise ValueError(
                "_normalize_paged_layers: shape_desc, device, and dtype are "
                "required when paged_buffer_ptrs_tensor is a pointer tensor"
            )
        nb = int(shape_desc.nb)
        bs = int(shape_desc.bs)
        nh = int(shape_desc.nh)
        hs = int(shape_desc.hs)
        per_shape = _per_layer_paged_shape(engine_kv_format, nb, bs, nh, hs)
        block_stride = int(getattr(shape_desc, "block_stride_elems", 0) or 0)
        if block_stride and block_stride != bs * nh * hs:
            raise NotImplementedError(
                "Non-tight per-block strides (vLLM blocks-first pools) are "
                "not supported when reconstructing paged tensors from raw "
                "pointers in the non-CUDA fallback."
            )
        return [
            _tensor_from_ptr(int(p.item()), per_shape, dtype, device)
            for p in paged_buffer_ptrs_tensor
        ]
    if isinstance(paged_buffer_ptrs_tensor, list):
        if not all(isinstance(t, torch.Tensor) for t in paged_buffer_ptrs_tensor):
            raise TypeError(
                "paged_buffer_ptrs_tensor list must contain torch.Tensor entries"
            )
        return paged_buffer_ptrs_tensor
    raise TypeError(
        "paged_buffer_ptrs_tensor must be a list[torch.Tensor] or 1-D pointer tensor; "
        "got: " + type(paged_buffer_ptrs_tensor).__name__
    )


def _normalize_lmcache_objects(
    lmcache_objects_ptrs: "list[int] | list[torch.Tensor]",
    shape_desc: "PageBufferShapeDesc | None" = None,
    lmcache_chunk_size: "int | None" = None,
    engine_kv_format: "EngineKVFormat | None" = None,
    dtype: "torch.dtype | None" = None,
) -> list[torch.Tensor]:
    """Normalize LMCache object inputs to chunk tensors.

    Accepts either a list of chunk tensors or a ``list[int]`` of raw CPU pointers.
    When a pointer list is provided *shape_desc*, *lmcache_chunk_size*,
    *engine_kv_format*, and *dtype* must be supplied so the tensors can be
    reconstructed via :func:`_tensor_from_ptr` on the CPU.
    """
    if not isinstance(lmcache_objects_ptrs, list):
        raise TypeError(
            "lmcache_objects_ptrs must be a list[torch.Tensor] or list[int]; "
            "got: " + type(lmcache_objects_ptrs).__name__
        )
    if not lmcache_objects_ptrs:
        return []
    if isinstance(lmcache_objects_ptrs[0], torch.Tensor):
        return lmcache_objects_ptrs  # type: ignore[return-value]
    if isinstance(lmcache_objects_ptrs[0], int):
        # Pointer mode: reconstruct chunk tensors (always on CPU).
        if (
            shape_desc is None
            or lmcache_chunk_size is None
            or engine_kv_format is None
            or dtype is None
        ):
            raise ValueError(
                "_normalize_lmcache_objects: shape_desc, lmcache_chunk_size, "
                "engine_kv_format, and dtype are required when lmcache_objects_ptrs "
                "contains raw int pointers"
            )
        nl = int(shape_desc.nl)
        nh = int(shape_desc.nh)
        hs = int(shape_desc.hs)
        chunk_tokens = lmcache_chunk_size
        if is_mla(engine_kv_format):
            chunk_shape: tuple[int, ...] = (nl, chunk_tokens, hs)
        elif _is_fused_kv_format(engine_kv_format):
            # Single plane: hs is the packed 2 * head_size.
            chunk_shape = (nl, chunk_tokens, nh * hs)
        else:
            chunk_shape = (2, nl, chunk_tokens, nh * hs)
        return [
            _tensor_from_ptr(ptr, chunk_shape, dtype, "cpu")
            for ptr in lmcache_objects_ptrs
        ]
    raise TypeError(
        "lmcache_objects_ptrs must be a list[torch.Tensor] or list[int]; "
        "got list containing: " + type(lmcache_objects_ptrs[0]).__name__
    )


def multi_layer_block_kv_transfer(
    paged_buffer_ptrs_tensor: "torch.Tensor | list",
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Python fallback implementation of block-based multi-layer KV transfer.

    Signature intentionally mirrors the C++ binding so callers can invoke
    ``lmcache.device_ops.multi_layer_block_kv_transfer`` uniformly on native and
    fallback backends.

    Args:
        paged_buffer_ptrs_tensor: Paged buffer pointers or tensors.
        lmcache_objects_ptrs: LMCache object pointers or chunk tensors.
        block_ids: Ordered engine block IDs for the transfer.
        device: Target device for the transfer.
        direction: Transfer direction (H2D or D2H).
        shape_desc: Shape descriptor of the page buffer.
        lmcache_chunk_size: Chunk size of LMCache objects.
        engine_kv_format: GPU KV cache format.
        skip_prefix_n_blocks: Number of leading blocks to skip.

    Returns:
        None

    Raises:
        ValueError: If chunk size is invalid, or transfer direction is unsupported.
        TypeError: If input types do not match expected types.
    """
    if lmcache_chunk_size <= 0:
        raise ValueError("lmcache_chunk_size must be positive")
    if int(shape_desc.bs) <= 0 or lmcache_chunk_size % int(shape_desc.bs) != 0:
        raise ValueError(
            "lmcache_chunk_size must be a positive multiple of shape_desc.bs"
        )
    if skip_prefix_n_blocks < 0:
        raise ValueError("skip_prefix_n_blocks must be >= 0")

    is_d2h = int(direction) == int(TransferDirection.D2H)
    is_h2d = int(direction) == int(TransferDirection.H2D)
    if not (is_d2h or is_h2d):
        raise ValueError(f"Unsupported transfer direction: {direction!r}")

    kv_dtype = _infer_kv_dtype(
        paged_buffer_ptrs_tensor, lmcache_objects_ptrs, shape_desc
    )
    normalized = _normalize_paged_layers(
        paged_buffer_ptrs_tensor,
        engine_kv_format,
        shape_desc=shape_desc,
        device=device,
        dtype=kv_dtype,
    )
    object_tensors = _normalize_lmcache_objects(
        lmcache_objects_ptrs,
        shape_desc=shape_desc,
        lmcache_chunk_size=lmcache_chunk_size,
        engine_kv_format=engine_kv_format,
        dtype=kv_dtype,
    )
    n_block_ids = (
        int(block_ids.numel())
        if isinstance(block_ids, torch.Tensor)
        else len(block_ids)
    )
    blocks_per_object = lmcache_chunk_size // int(shape_desc.bs)
    block_size = int(shape_desc.bs)

    if is_cross_layer(engine_kv_format):
        _transfer_cross_layer(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    elif is_kv_list(engine_kv_format):
        _transfer_sglang_mha(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    elif is_mla(engine_kv_format):
        _transfer_per_layer_mla(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    elif _is_fused_kv_format(engine_kv_format):
        # Before the HND branch: the fused formats are HND/NHD too, but their
        # packed K/V axis needs the single-plane path.
        _transfer_per_layer_fused(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    elif _is_hnd_format(engine_kv_format):
        _transfer_per_layer_hnd(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    elif _is_kv_second_tuple_format(engine_kv_format):
        _transfer_per_layer_kv_tuple(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )
    else:
        _transfer_per_layer_nhd(
            normalized,
            object_tensors,
            block_ids,
            n_block_ids,
            blocks_per_object,
            block_size,
            engine_kv_format,
            is_d2h,
            skip_prefix_n_blocks,
        )


def _valid_block_range(
    object_idx: int,
    block_id_list: list[int],
    blocks_per_object: int,
    block_size: int,
    skip_prefix_n_blocks: int,
) -> tuple[list[int], int] | None:
    """Return valid engine block IDs and their LMCache object token offset.

    Args:
        object_idx: Index of the LMCache object/chunk being processed.
        block_id_list: Full ordered engine block ids for the transfer.
        blocks_per_object: Number of blocks represented by one LMCache object.
        block_size: Number of tokens per block.
        skip_prefix_n_blocks: Number of leading flat block positions to skip.

    Returns:
        ``None`` if this object has no valid blocks after skip handling.
        Otherwise, a tuple of valid engine block ids and the token offset
        within this LMCache object where those blocks start.
    """
    object_flat_start = object_idx * blocks_per_object
    valid_flat_start = max(object_flat_start, skip_prefix_n_blocks)
    valid_flat_end = min(object_flat_start + blocks_per_object, len(block_id_list))
    if valid_flat_start >= valid_flat_end:
        return None
    offset_in_object = (valid_flat_start - object_flat_start) * block_size
    return block_id_list[valid_flat_start:valid_flat_end], offset_in_object


def _valid_block_range_indices(
    object_idx: int,
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    skip_prefix_n_blocks: int,
) -> tuple[int, int, int] | None:
    """Return valid [start, end) range over flat block IDs and object token offset."""
    object_flat_start = object_idx * blocks_per_object
    valid_flat_start = max(object_flat_start, skip_prefix_n_blocks)
    valid_flat_end = min(object_flat_start + blocks_per_object, n_block_ids)
    if valid_flat_start >= valid_flat_end:
        return None
    offset_in_object = (valid_flat_start - object_flat_start) * block_size
    return valid_flat_start, valid_flat_end, offset_in_object


def _transfer_cross_layer(
    paged_tensor: torch.Tensor,
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle cross-layer formats: single tensor [NB, NL, 2, ...]."""
    # NHD: [NB, NL, 2, BS, NH, HS]  HND: [NB, NL, 2, NH, BS, HS]
    is_hnd = _is_hnd_format(engine_kv_format)
    num_layers = paged_tensor.shape[1]

    if is_hnd:
        # [NB, NL, 2, NH, BS, HS]
        nh = paged_tensor.shape[3]
        hs = paged_tensor.shape[5]
    else:
        # [NB, NL, 2, BS, NH, HS]
        nh = paged_tensor.shape[4]
        hs = paged_tensor.shape[5]

    # H2D: pre-transfer objects to paged device
    if not is_d2h and object_tensors:
        objs_on_device = [obj.to(paged_tensor.device) for obj in object_tensors]
    block_ids_dev = torch.as_tensor(
        block_ids, dtype=torch.long, device=paged_tensor.device
    )

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]

        if is_d2h:
            selected = paged_tensor.index_select(0, eff_idx)

        for layer_idx in range(num_layers):
            for kv in range(2):
                if is_d2h:
                    slice_t = selected[:, layer_idx, kv]
                    if is_hnd:
                        # N=n_valid, BS=block_size:
                        # [N, NH, BS, HS] -> [N, BS, NH, HS] -> [N*BS, NH*HS]
                        flat = slice_t.permute(0, 2, 1, 3).reshape(
                            n_valid * block_size, nh * hs
                        )
                    else:
                        # [N, BS, NH, HS] → [N*BS, NH*HS]
                        flat = slice_t.reshape(n_valid * block_size, nh * hs)
                    obj[kv, layer_idx, offset_in_object:token_end].copy_(
                        flat, non_blocking=True
                    )
                else:
                    obj_device = objs_on_device[object_idx]
                    src = obj_device[kv, layer_idx, offset_in_object:token_end]
                    if is_hnd:
                        # N=n_valid, BS=block_size:
                        # [N*BS, NH*HS] -> [N, BS, NH, HS] -> [N, NH, BS, HS]
                        src_blocks = src.reshape(n_valid, block_size, nh, hs).permute(
                            0, 2, 1, 3
                        )
                    else:
                        # N=n_valid, BS=block_size:
                        # [N*BS, NH*HS] -> [N, BS, NH, HS]
                        src_blocks = src.reshape(n_valid, block_size, nh, hs)
                    paged_tensor[:, layer_idx, kv].index_copy_(0, eff_idx, src_blocks)


def _transfer_sglang_mha(
    paged_tensors: list[list[torch.Tensor]],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle SGLang MHA formats: 2*NL tensors (list[list[Tensor]])."""
    # TWO_X_NL_X_NBBS_NH_HS: each tensor [NB*BS, NH, HS]
    # TWO_X_NL_X_NB_BS_NH_HS: each tensor [NB, BS, NH, HS]
    is_flat = _is_pbs_fused_format(engine_kv_format)
    num_layers = len(paged_tensors[0])

    # Determine target device from first tensor
    target_device = paged_tensors[0][0].device

    # H2D: pre-transfer objects
    if not is_d2h and object_tensors:
        objs_on_device = [obj.to(target_device) for obj in object_tensors]
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]
        if is_flat:
            # Flat token positions for all valid blocks:
            # block_id * block_size + token offset. Reused across layer/KV pairs.
            token_indices = (
                eff_idx[:, None] * block_size
                + torch.arange(block_size, dtype=torch.long, device=target_device)
            ).reshape(-1)

        for layer_idx in range(num_layers):
            for kv in range(2):
                layer_t = paged_tensors[kv][layer_idx]
                nh = layer_t.shape[-2]
                hs = layer_t.shape[-1]

                if is_d2h:
                    if is_flat:
                        # [NB*BS, NH, HS]
                        gathered = layer_t.index_select(0, token_indices)
                    else:
                        # [NB, BS, NH, HS]
                        gathered = layer_t.index_select(0, eff_idx).reshape(
                            n_valid * block_size, nh, hs
                        )
                    flat = gathered.reshape(n_valid * block_size, nh * hs)
                    obj[kv, layer_idx, offset_in_object:token_end].copy_(
                        flat, non_blocking=True
                    )
                else:
                    obj_device = objs_on_device[object_idx]
                    src = obj_device[kv, layer_idx, offset_in_object:token_end]
                    src_shaped = src.reshape(n_valid * block_size, nh, hs)
                    if is_flat:
                        # scatter into [NB*BS, NH, HS]
                        layer_t.index_copy_(0, token_indices, src_shaped)
                    else:
                        # N=n_valid, BS=block_size:
                        # [N*BS, NH, HS] -> [N, BS, NH, HS]
                        src_blocks = src_shaped.reshape(n_valid, block_size, nh, hs)
                        layer_t.index_copy_(0, eff_idx, src_blocks)


def _transfer_per_layer_kv_tuple(
    paged_tensors: list[list[torch.Tensor]],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Transfer the per-layer ``(K, V)`` tuple format."""
    if not paged_tensors or not object_tensors:
        return

    num_layers = len(paged_tensors)
    target_device = paged_tensors[0][0].device

    # H2D stages objects on the paged tensors' device once, up front.
    objs_on_device: list[torch.Tensor] = (
        [] if is_d2h else [obj.to(target_device) for obj in object_tensors]
    )
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]

        for layer_idx in range(num_layers):
            k_t, v_t = paged_tensors[layer_idx]
            for kv, layer_t in enumerate((k_t, v_t)):
                nh = layer_t.shape[-2]
                hs = layer_t.shape[-1]
                if is_d2h:
                    # [NB, BS, NH, HS] -> [n_valid * BS, NH * HS]
                    flat = layer_t.index_select(0, eff_idx).reshape(
                        n_valid * block_size, nh * hs
                    )
                    obj[kv, layer_idx, offset_in_object:token_end].copy_(
                        flat, non_blocking=True
                    )
                else:
                    src = objs_on_device[object_idx][
                        kv, layer_idx, offset_in_object:token_end
                    ]
                    # [n_valid * BS, NH * HS] -> [n_valid, BS, NH, HS]
                    src_blocks = src.reshape(n_valid, block_size, nh, hs)
                    layer_t.index_copy_(0, eff_idx, src_blocks)


def _transfer_per_layer_mla(
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle MLA per-layer formats: [NB, BS, HS]."""
    if not layer_tensors or not object_tensors:
        return

    is_flat = _is_pbs_fused_format(engine_kv_format)
    target_device = layer_tensors[0].device
    if is_flat:
        token_offsets = torch.arange(block_size, dtype=torch.long, device=target_device)
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]
        if is_flat:
            token_indices = (
                eff_idx[:, None] * block_size + token_offsets[None, :]
            ).reshape(-1)

        if is_d2h:
            hidden_size = layer_tensors[0].shape[-1]
            chunk_gpu = torch.empty(
                len(layer_tensors),
                n_valid * block_size,
                hidden_size,
                dtype=layer_tensors[0].dtype,
                device=target_device,
            )
            for layer_idx, layer in enumerate(layer_tensors):
                if is_flat:
                    dst = chunk_gpu[layer_idx].view(
                        n_valid * block_size, 1, hidden_size
                    )
                    torch.index_select(layer, 0, token_indices, out=dst)
                else:
                    dst = chunk_gpu[layer_idx].view(n_valid, block_size, hidden_size)
                    torch.index_select(layer, 0, eff_idx, out=dst)
            obj[:, offset_in_object:token_end].copy_(chunk_gpu, non_blocking=True)
        else:
            chunk_gpu = obj[:, offset_in_object:token_end].to(
                target_device, non_blocking=True
            )
            for layer_idx, layer in enumerate(layer_tensors):
                src = chunk_gpu[layer_idx]
                hidden_size = layer.shape[-1]
                if is_flat:
                    src_tokens = src.reshape(n_valid * block_size, 1, hidden_size)
                    layer.index_copy_(0, token_indices, src_tokens)
                else:
                    src_blocks = src.reshape(n_valid, block_size, hidden_size)
                    layer.index_copy_(0, eff_idx, src_blocks)


def _transfer_per_layer_hnd(
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle per-layer HND formats: heads before block tokens."""
    if not layer_tensors or not object_tensors:
        return

    target_device = layer_tensors[0].device
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    # Two-major keeps K and V as separate leading planes ([2, NB, NH, BS, HS]);
    # otherwise the size-2 axis sits after the blocks ([NB, 2, NH, BS, HS]).
    is_two_major = _is_two_major_format(engine_kv_format)
    first_layer = layer_tensors[0]
    if is_two_major:
        first_k = first_layer[0]
    else:
        first_k = first_layer[:, 0]
    _nb0, nh0, _bs0, hs0 = first_k.shape

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]

        if is_d2h:
            chunk_gpu = torch.empty(
                2,
                len(layer_tensors),
                n_valid * block_size,
                nh0 * hs0,
                dtype=first_k.dtype,
                device=target_device,
            )
            scratch = torch.empty(
                n_valid,
                nh0,
                block_size,
                hs0,
                dtype=first_k.dtype,
                device=target_device,
            )
            for layer_idx, layer in enumerate(layer_tensors):
                if is_two_major:
                    k_t, v_t = layer[0], layer[1]
                    torch.index_select(k_t, 0, eff_idx, out=scratch)
                    chunk_gpu[0, layer_idx].view(n_valid, block_size, nh0, hs0).copy_(
                        scratch.permute(0, 2, 1, 3)
                    )
                    torch.index_select(v_t, 0, eff_idx, out=scratch)
                    chunk_gpu[1, layer_idx].view(n_valid, block_size, nh0, hs0).copy_(
                        scratch.permute(0, 2, 1, 3)
                    )
                else:
                    # FlashInfer HND stores KV as [NB, 2, NH, BS, HS].
                    # Gather on dim=0 first so reads stay contiguous in memory;
                    # index_select on layer[:, 0]/layer[:, 1] non-contiguous views
                    # triggers slower element-wise gather reads.
                    selected = layer.index_select(0, eff_idx)
                    chunk_gpu[0, layer_idx].view(n_valid, block_size, nh0, hs0).copy_(
                        selected[:, 0].permute(0, 2, 1, 3)
                    )
                    chunk_gpu[1, layer_idx].view(n_valid, block_size, nh0, hs0).copy_(
                        selected[:, 1].permute(0, 2, 1, 3)
                    )
            obj[:, :, offset_in_object:token_end].copy_(chunk_gpu, non_blocking=True)
        else:
            chunk_gpu = obj[:, :, offset_in_object:token_end].to(
                target_device, non_blocking=True
            )
            for layer_idx, layer in enumerate(layer_tensors):
                if is_two_major:
                    k_t, v_t = layer[0], layer[1]
                else:
                    k_t, v_t = layer[:, 0], layer[:, 1]
                _nb, nh, _bs, hs = k_t.shape
                k_blocks = (
                    chunk_gpu[0, layer_idx]
                    .reshape(n_valid, block_size, nh, hs)
                    .permute(0, 2, 1, 3)
                )
                v_blocks = (
                    chunk_gpu[1, layer_idx]
                    .reshape(n_valid, block_size, nh, hs)
                    .permute(0, 2, 1, 3)
                )
                if not is_two_major:
                    layer.index_copy_(
                        0, eff_idx, torch.stack([k_blocks, v_blocks], dim=1)
                    )
                else:
                    k_t.index_copy_(0, eff_idx, k_blocks)
                    v_t.index_copy_(0, eff_idx, v_blocks)


def _transfer_per_layer_fused(
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle fused-K/V per-layer formats (kv_size == 1).

    The K/V pair stays packed inside each ``2 * head_size`` head, so every
    layer transfers as a single plane and the object layout is
    ``[NL, tokens, NH * 2 * HS]`` — byte-identical to the device kernel's.
    """
    if not layer_tensors or not object_tensors:
        return

    target_device = layer_tensors[0].device
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    # Callers pass either the raw 4-D registration ([NB, NH, BS, 2*HS] or
    # [NB, BS, NH, 2*HS]) or the canonical 5-D split with the size-2 axis
    # second-to-last; both share storage, so flatten the trailing pair away.
    layers = [
        layer.reshape(*layer.shape[:3], -1) if layer.dim() == 5 else layer
        for layer in layer_tensors
    ]
    is_hnd = _is_hnd_format(engine_kv_format)
    first = layers[0]
    if is_hnd:
        _nb0, nh0, _bs0, hs0 = first.shape
    else:
        _nb0, _bs0, nh0, hs0 = first.shape
    nl = len(layers)

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]
        # Tolerate legacy [2, NL, T, H]-shaped buffers: the flat storage is
        # reinterpreted as the single-plane layout.
        chunk_tokens = obj.numel() // (nl * nh0 * hs0)
        obj_view = obj.reshape(nl, chunk_tokens, nh0 * hs0)

        if is_d2h:
            chunk_gpu = torch.empty(
                nl,
                n_valid * block_size,
                nh0 * hs0,
                dtype=first.dtype,
                device=target_device,
            )
            for layer_idx, layer in enumerate(layers):
                selected = layer.index_select(0, eff_idx)
                if is_hnd:
                    # [n, NH, BS, 2*HS] -> tokens-major [n, BS, NH, 2*HS]
                    selected = selected.permute(0, 2, 1, 3)
                chunk_gpu[layer_idx].view(n_valid, block_size, nh0, hs0).copy_(selected)
            obj_view[:, offset_in_object:token_end].copy_(chunk_gpu, non_blocking=True)
        else:
            chunk_gpu = obj_view[:, offset_in_object:token_end].to(
                target_device, non_blocking=True
            )
            for layer_idx, layer in enumerate(layers):
                blocks = chunk_gpu[layer_idx].reshape(n_valid, block_size, nh0, hs0)
                if is_hnd:
                    blocks = blocks.permute(0, 2, 1, 3)
                layer.index_copy_(0, eff_idx, blocks)


def _transfer_per_layer_nhd(
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    n_block_ids: int,
    blocks_per_object: int,
    block_size: int,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
    skip_prefix_n_blocks: int,
) -> None:
    """Handle per-layer NHD formats: block tokens before heads."""
    if not layer_tensors or not object_tensors:
        return

    target_device = layer_tensors[0].device
    block_ids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=target_device)

    # Two-major keeps K and V as separate leading planes ([2, NB, BS, NH, HS]);
    # otherwise the size-2 axis sits after the blocks ([NB, 2, BS, NH, HS]).
    is_two_major = _is_two_major_format(engine_kv_format)
    first_layer = layer_tensors[0]
    if is_two_major:
        first_k = first_layer[0]
    else:
        first_k = first_layer[:, 0]
    _nb0, _bs0, nh0, hs0 = first_k.shape

    for object_idx, obj in enumerate(object_tensors):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        n_valid = idx_end - idx_start
        token_end = offset_in_object + n_valid * block_size
        eff_idx = block_ids_dev[idx_start:idx_end]

        if is_d2h:
            chunk_gpu = torch.empty(
                2,
                len(layer_tensors),
                n_valid * block_size,
                nh0 * hs0,
                dtype=first_k.dtype,
                device=target_device,
            )
            for layer_idx, layer in enumerate(layer_tensors):
                if is_two_major:
                    k_t, v_t = layer[0], layer[1]
                    torch.index_select(
                        k_t,
                        0,
                        eff_idx,
                        out=chunk_gpu[0, layer_idx].view(n_valid, block_size, nh0, hs0),
                    )
                    torch.index_select(
                        v_t,
                        0,
                        eff_idx,
                        out=chunk_gpu[1, layer_idx].view(n_valid, block_size, nh0, hs0),
                    )
                else:
                    # FlashInfer NHD stores KV as [NB, 2, BS, NH, HS].
                    # Gather on dim=0 first to avoid index_select from
                    # non-contiguous layer[:, 0]/layer[:, 1] views, which
                    # trigger slower element-wise gather reads.
                    selected = layer.index_select(0, eff_idx)
                    chunk_gpu[0, layer_idx].copy_(
                        selected[:, 0].reshape(n_valid * block_size, nh0 * hs0)
                    )
                    chunk_gpu[1, layer_idx].copy_(
                        selected[:, 1].reshape(n_valid * block_size, nh0 * hs0)
                    )
            obj[:, :, offset_in_object:token_end].copy_(chunk_gpu, non_blocking=True)
        else:
            chunk_gpu = obj[:, :, offset_in_object:token_end].to(
                target_device, non_blocking=True
            )
            for layer_idx, layer in enumerate(layer_tensors):
                if is_two_major:
                    k_t, v_t = layer[0], layer[1]
                    k_t.index_copy_(
                        0,
                        eff_idx,
                        chunk_gpu[0, layer_idx].reshape(n_valid, block_size, nh0, hs0),
                    )
                    v_t.index_copy_(
                        0,
                        eff_idx,
                        chunk_gpu[1, layer_idx].reshape(n_valid, block_size, nh0, hs0),
                    )
                else:
                    k_blocks = chunk_gpu[0, layer_idx].reshape(
                        n_valid, block_size, nh0, hs0
                    )
                    v_blocks = chunk_gpu[1, layer_idx].reshape(
                        n_valid, block_size, nh0, hs0
                    )
                    layer.index_copy_(
                        0, eff_idx, torch.stack([k_blocks, v_blocks], dim=1)
                    )


def _to_block_id_list(block_ids: torch.Tensor | list[int]) -> list[int]:
    """Convert block IDs from tensor/list form into a Python ``list[int]``."""
    if isinstance(block_ids, torch.Tensor):
        return [int(x) for x in block_ids.to(dtype=torch.int64).cpu().tolist()]
    if isinstance(block_ids, list):
        return [int(x) for x in block_ids]
    raise TypeError("block_ids must be a torch.Tensor or list[int]")
