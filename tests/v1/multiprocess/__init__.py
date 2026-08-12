# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

lmcache_native = pytest.importorskip(
    "lmcache.lmcache_native",
    reason="lmcache.lmcache_native extension is not available",
)

_required_symbols = ("Bitmap", "PeriodicEventNotifier")
_missing_symbols = [
    symbol for symbol in _required_symbols if not hasattr(lmcache_native, symbol)
]
if _missing_symbols:
    pytest.skip(
        "lmcache.lmcache_native is missing required symbols for multiprocess "
        f"tests: {_missing_symbols}",
        allow_module_level=True,
    )
