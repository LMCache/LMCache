# SPDX-License-Identifier: Apache-2.0
"""Resource stubs for testing the real server-module composition contract."""

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.engine_module import DiscoverableModule, discover_modules
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule
from lmcache.v1.multiprocess.modules.management import ManagementModule


@pytest.fixture
def stub_module_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[type[DiscoverableModule], MagicMock]:
    """Stub resource owners while keeping factories, management, and events real."""
    constructors: dict[type[DiscoverableModule], MagicMock] = {}

    def capture(self: DiscoverableModule, *args: Any, **kwargs: Any) -> None:
        constructors[type(self)](*args, **kwargs)

    for module_type in discover_modules():
        if module_type in (ManagementModule, KVEventModule):
            continue
        constructors[module_type] = MagicMock(name=module_type.__name__)
        monkeypatch.setattr(module_type, "__init__", capture)
        monkeypatch.setattr(module_type, "close", MagicMock())
        if hasattr(module_type, "touch_instance"):
            monkeypatch.setattr(module_type, "touch_instance", MagicMock())
    return constructors
