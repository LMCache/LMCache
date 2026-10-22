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

# First Party
from lmcache.v1.gpu_connector.kv_format.types import transfer_word_bytes


def _blocked_scale_views(
    blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a blocked-scale page into token-indexed value and scale views.

    A blocked-scale block stores all tokens' values first, then all tokens'
    scales (``[BS x vals][BS x scales]``), while LMCache chunk rows are
    token-major ``[vals | scale]``. Indexing the returned views by
    ``[block, token]`` addresses one token's values or scale, so callers can
    gather and scatter canonical rows without a whole-block copy.

    Args:
        blocks: One layer's ``[NB, BS, row]`` page; each block's ``BS * row``
            elements must be contiguous.

    Returns:
        ``(vals, scales)`` views aliasing ``blocks``, shaped
        ``[NB, BS, row - scale_units]`` and ``[NB, BS, scale_units]``.
    """
    num_blocks, block_size, row = blocks.shape
    # The per-token scale is one transfer word (fp32 for the indexer, 8 B of UE8M0
    # for the V4 main KV), the native kernels' contract.
    scale_units = (
        transfer_word_bytes(row * blocks.element_size()) // blocks.element_size()
    )
    val_units = row - scale_units
    # ``view`` rather than ``flatten``: a copy would silently drop the writes.
    flat = blocks.view(num_blocks, block_size * row)
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
