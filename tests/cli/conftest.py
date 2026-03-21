# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for CLI tests.

The CLI arg-registration code transitively imports
``lmcache.native_storage_ops`` (a compiled C extension).  On CI runners
without a CUDA build the module is absent, so we insert a lightweight
stub into ``sys.modules`` before any CLI test touches the import chain.
"""

# Standard
from unittest.mock import MagicMock
import importlib
import sys
import types


def _ensure_native_stub():
    """Insert a mock for ``lmcache.native_storage_ops`` if it is not built."""
    mod_name = "lmcache.native_storage_ops"
    if importlib.util.find_spec(mod_name) is None:
        stub = types.ModuleType(mod_name)
        # Provide the symbols that downstream modules import at top level.
        stub.TTLLock = MagicMock()
        stub.Bitmap = MagicMock()
        stub.ParallelPatternMatcher = MagicMock()
        stub.RangePatternMatcher = MagicMock()
        sys.modules[mod_name] = stub


_ensure_native_stub()
