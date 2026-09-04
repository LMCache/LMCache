# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.lmcache_native import EngineKVFormat

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec

__all__ = [
    "_format_spec",
    "_is_hnd_format",
    "_is_fused_kv_format",
    "_is_two_major_format",
    "_is_pbs_fused_format",
    "_is_kv_second_tuple_format",
]


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


def _is_hnd_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True when a KV format stores heads before block tokens (HND)."""
    return _format_spec(engine_kv_format).is_hnd


def _is_fused_kv_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True for formats whose K/V pair is packed in the trailing dim
    (kv_size == 1, shape_desc.hs == 2 * head_size)."""
    return _format_spec(engine_kv_format).is_fused_packed


def _is_two_major_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True when the size-2 K/V axis precedes the block axis."""
    return _format_spec(engine_kv_format).is_two_major


def _is_pbs_fused_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True when num_blocks and block_size are one folded PBS axis."""
    return _format_spec(engine_kv_format).is_pbs_fused


def _is_kv_second_tuple_format(engine_kv_format: EngineKVFormat) -> bool:
    """Return True when each per-layer entry is a (K, V) tuple."""
    return _format_spec(engine_kv_format).is_kv_second_tuple
