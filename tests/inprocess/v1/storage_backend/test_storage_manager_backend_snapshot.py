# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
import threading

# First Party
from lmcache.inprocess.v1.storage_backend.storage_manager import StorageManager


class _Backend:
    """Minimal backend used to observe iteration behavior."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel_request(self, req_id: str) -> None:
        self.cancelled = True


def _make_manager(backends: OrderedDict) -> StorageManager:
    """Create a StorageManager with only iteration state.

    Args:
        backends: Backend mapping to install.

    Returns:
        A manager with the locks used by backend iteration.
    """
    manager = StorageManager.__new__(StorageManager)
    manager.storage_backends = backends
    manager.manager_lock = threading.Lock()
    manager._freeze = False
    manager._freeze_lock = threading.RLock()
    manager._bypassed_backends = set()
    manager._bypass_lock = threading.RLock()
    return manager


def test_active_backends_use_snapshot() -> None:
    """Mutating the backend map after the first item must not break iteration."""
    first, second = _Backend(), _Backend()
    manager = _make_manager(OrderedDict([("A", first), ("B", second)]))

    iterator = manager.get_active_storage_backends()
    assert next(iterator)[0] == "A"
    manager.storage_backends.clear()

    assert [name for name, _ in iterator] == ["B"]


def test_cancel_request_uses_snapshot() -> None:
    """Cancelling a request must not break when a backend mutates the map."""
    first, second = _Backend(), _Backend()
    manager = _make_manager(OrderedDict([("A", first), ("B", second)]))

    def cancel_and_remove(req_id: str) -> None:
        first.cancelled = True
        manager.storage_backends.pop("B", None)

    first.cancel_request = cancel_and_remove
    manager.cancel_request("req")

    assert first.cancelled
    assert second.cancelled
