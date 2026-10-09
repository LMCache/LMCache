# SPDX-License-Identifier: Apache-2.0
"""MUSA transfer capability predicates derived from KV format traits."""

# First Party
from lmcache.lmcache_native import EngineKVFormat
from lmcache.v1.gpu_connector.kv_format import get_spec_class
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec


def _is_regular_per_layer_layout(spec: type[KVFormatSpec]) -> bool:
    """Return whether ``spec`` uses the dense per-layer pointer contract."""
    return (
        spec.is_layer_list
        and not spec.is_hnd
        and not spec.is_fused_packed
        and not spec.is_pbs_fused
        and not spec.is_kv_second_tuple
        and not spec.is_blocked_scale
    )


def is_supported_musa_mp_block_transfer_format(
    engine_kv_format: EngineKVFormat,
) -> bool:
    """Return whether the current MUSA MP block path accepts a format.

    Args:
        engine_kv_format: Engine KV format to classify.

    Returns:
        ``True`` for ordinary MLA, NHD ``TWO_NB`` per-layer layouts, and the
        current SGLang K/V-list layout. ``False`` for all other physical
        contracts.
    """
    spec = get_spec_class(engine_kv_format)
    if _is_regular_per_layer_layout(spec):
        return spec.is_mla or spec.is_two_major
    return (
        spec.is_kv_list
        and not spec.is_mla
        and not spec.is_pbs_fused
        and not spec.is_kv_second_tuple
        and not spec.is_blocked_scale
    )


def is_supported_musa_native_block_transfer_format(
    engine_kv_format: EngineKVFormat,
) -> bool:
    """Return whether the current optional native ABI accepts a format.

    Args:
        engine_kv_format: Engine KV format to classify.

    Returns:
        ``True`` for ordinary MLA and NHD ``TWO_NB`` per-layer layouts.
        ``False`` for K/V-list, HND, fused, blocked-scale, tuple, and other
        layouts that the current native ABI does not describe.
    """
    spec = get_spec_class(engine_kv_format)
    return _is_regular_per_layer_layout(spec) and (spec.is_mla or spec.is_two_major)
