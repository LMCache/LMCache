# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.lmcache_native import EngineKVFormat

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec

__all__ = ["_format_spec"]


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
