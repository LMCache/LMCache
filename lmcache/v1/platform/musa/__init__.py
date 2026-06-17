# SPDX-License-Identifier: Apache-2.0
"""MUSA-specific platform helpers."""

# Standard
from typing import Any

# First Party
from lmcache.v1.platform._registry import register_block_transfer_hook


def _block_transfer_hook(**kwargs: Any) -> bool:
    """Try the optional native MUSA block-transfer implementation.

    Args:
        **kwargs: Keyword arguments forwarded from
            ``python_ops_fallback.multi_layer_block_kv_transfer``.

    Returns:
        ``True`` when the native MUSA adapter completed the transfer,
        otherwise ``False`` so callers can continue through the generic
        fallback path.
    """
    # Keep this import inside the hook so platform package bootstrap only
    # registers the extension point; optional native symbols are loaded on use.
    # First Party
    from lmcache.v1.platform.musa.native_kv_transfer import (
        try_native_multi_layer_block_kv_transfer,
    )

    return try_native_multi_layer_block_kv_transfer(**kwargs)


register_block_transfer_hook("musa", _block_transfer_hook)
