# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

native_storage_ops = pytest.importorskip(
	"lmcache.native_storage_ops",
	reason="lmcache.native_storage_ops extension is not available",
)

_required_symbols = ("Bitmap", "PeriodicEventNotifier")
_missing_symbols = [
	symbol for symbol in _required_symbols if not hasattr(native_storage_ops, symbol)
]
if _missing_symbols:
	pytest.skip(
		"lmcache.native_storage_ops is missing required symbols for distributed "
		f"tests: {_missing_symbols}",
		allow_module_level=True,
	)
