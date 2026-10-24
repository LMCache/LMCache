# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec

__all__ = ["_blocked_scale_views", "_format_spec"]

# Each token's scale in a blocked-scale page is one fp32 (vllm
# ``indexer_k_quant_and_cache``); the rest of the row is its fp8 values.
_BLOCKED_SCALE_BYTES = 4


def _blocked_scale_views(
    blocks: torch.Tensor,
    block_size: int | None = None,
    row_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a blocked-scale page into token-indexed value and scale views.

    A blocked-scale block stores all tokens' values first, then all tokens'
    scales (``[BS x vals][BS x scales]``), while LMCache chunk rows are
    token-major ``[vals | scale]``. Indexing the returned views by
    ``[block, token]`` addresses one token's values or scale. The input may
    be a regular ``[NB, BS, row]`` view or a raw ``[NB, stride]`` view with
    padding between physical blocks.

    Args:
        blocks: One layer's blocked-scale page.
        block_size: Tokens per block. Required for a raw 2-D page.
        row_width: Bytes or elements per token row. Required for a raw 2-D
            page and used to validate a 3-D page.

    Returns:
        ``(vals, scales)`` views aliasing ``blocks``. The returned tensors are
        shaped ``[NB, BS, value_units]`` and ``[NB, BS, scale_units]``.

    Raises:
        ValueError: If the input rank or physical block layout is invalid.
    """
    if blocks.ndim == 3:
        num_blocks, inferred_block_size, inferred_row_width = blocks.shape
        if block_size is None:
            block_size = int(inferred_block_size)
        if row_width is None:
            row_width = int(inferred_row_width)
        if int(block_size) != int(inferred_block_size) or int(row_width) != int(
            inferred_row_width
        ):
            raise ValueError(
                "blocked-scale page shape does not match block geometry: "
                f"got {tuple(blocks.shape)}, expected block_size={block_size}, "
                f"row_width={row_width}"
            )
    elif blocks.ndim == 2:
        if block_size is None or row_width is None:
            raise ValueError(
                "block_size and row_width are required for a raw blocked-scale page"
            )
        num_blocks = int(blocks.shape[0])
        physical_block_elems = int(block_size) * int(row_width)
        if int(blocks.shape[1]) < physical_block_elems:
            raise ValueError(
                "raw blocked-scale page is smaller than one physical block"
            )
    else:
        raise ValueError(
            "blocked-scale page must be a [NB, BS, row] or [NB, stride] tensor, "
            f"got {tuple(blocks.shape)}"
        )

    if block_size is None or row_width is None:
        raise ValueError("blocked-scale block geometry is incomplete")
    block_size = int(block_size)
    row_width = int(row_width)
    if block_size <= 0 or row_width <= 0:
        raise ValueError("blocked-scale block_size and row_width must be positive")
    scale_units = _BLOCKED_SCALE_BYTES // blocks.element_size()
    val_units = row_width - scale_units
    if _BLOCKED_SCALE_BYTES % blocks.element_size() != 0 or val_units <= 0:
        raise ValueError("blocked-scale page has an invalid scale or value width")
    # ``as_strided`` preserves padding between physical blocks and never copies.
    flat = blocks.as_strided(
        (num_blocks, block_size * row_width),
        (int(blocks.stride(0)), 1),
    )
    vals = flat[:, : block_size * val_units].unflatten(1, (block_size, val_units))
    scales = flat[:, block_size * val_units :].unflatten(1, (block_size, scale_units))
    return vals, scales


def _format_spec(engine_kv_format: EngineKVFormat) -> "type[KVFormatSpec]":
    """Return the spec class owning *engine_kv_format*'s static layout facts.

    Args:
        engine_kv_format: The format to look up.

    Returns:
        The ``KVFormatSpec`` subclass declared for the format.

    Raises:
        ValueError: If the format has no spec.
    """
    # Imported lazily, not at module scope: the specs package reads
    # ``lmcache.device_ops``, which is resolved only once this module (the
    # torch baseline behind it) has been imported.
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec_class

    return get_spec_class(engine_kv_format)
