# SPDX-License-Identifier: Apache-2.0
"""The ``EngineKVFormat`` -> spec table, discovered from this folder.

Every ``specs/<engine_kv_format>.py`` defines one :class:`KVFormatSpec`
subclass. This imports them all and indexes each by the ``engine_kv_format`` it
declares, so adding a format is just dropping a new file here -- nothing in this
file changes.
"""

# Standard
from pathlib import Path
import importlib
import pkgutil

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
import lmcache.lmcache_native as lmcache_native


def _discover_specs() -> dict["lmcache_native.EngineKVFormat", type[KVFormatSpec]]:
    """Import every spec module in this folder and index it by its format."""
    specs: dict["lmcache_native.EngineKVFormat", type[KVFormatSpec]] = {}
    for module in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if module.name in ("base", "registry"):
            continue
        imported = importlib.import_module(f"{__package__}.{module.name}")
        for value in vars(imported).values():
            if (
                isinstance(value, type)
                and issubclass(value, KVFormatSpec)
                and value is not KVFormatSpec
            ):
                specs[value.engine_kv_format] = value
    return specs


SPECS = _discover_specs()

# Indexed by enum *value*: the native pybind ``EngineKVFormat`` and the
# pure-Python fallback one are distinct types with the same members, and both
# reach this table (e.g. from ``lmcache.v1.platform.torch_ops``).
_SPECS_BY_VALUE = {int(fmt): spec for fmt, spec in SPECS.items()}


def get_spec_class(fmt: "lmcache_native.EngineKVFormat") -> type[KVFormatSpec]:
    """Return the spec class for *fmt* -- the owner of its static facts.

    Args:
        fmt: The Engine KV format to look up.

    Returns:
        The :class:`KVFormatSpec` subclass declaring *fmt*, which carries the
        format's static layout facts (``is_mla``, ``is_hnd``, the structural
        shape, ``attention_backends``, ...) as class attributes.

    Raises:
        ValueError: If *fmt* has no spec.
    """
    spec = _SPECS_BY_VALUE.get(int(fmt))
    if spec is None:
        raise ValueError(f"Unknown Engine KV Format: {fmt}")
    return spec


def get_spec(
    kv_caches: DiscoverableKVCache, fmt: "lmcache_native.EngineKVFormat"
) -> KVFormatSpec:
    """Return a spec instance wrapping *kv_caches* of *fmt*."""
    return get_spec_class(fmt)(kv_caches)
