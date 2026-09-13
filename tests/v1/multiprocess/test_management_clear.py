# SPDX-License-Identifier: Apache-2.0
"""Tests for multiprocess management cache clearing."""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.multiprocess.modules.management import ManagementModule


def test_management_clear_preserves_locked_objects() -> None:
    """Management CLEAR uses non-forced storage clear and returns its status."""
    ctx = MagicMock(name="server_context")
    ctx.storage_manager.clear.return_value = False
    management = ManagementModule(ctx)

    assert management.clear() is False

    ctx.storage_manager.memcheck.assert_called()
    assert ctx.storage_manager.memcheck.call_count == 2
    ctx.storage_manager.clear.assert_called_once_with(force=False)


def test_management_clear_reports_complete_clear() -> None:
    """A complete storage clear is reported as True to the request client."""
    ctx = MagicMock(name="server_context")
    ctx.storage_manager.clear.return_value = True
    management = ManagementModule(ctx)

    assert management.clear() is True

    ctx.storage_manager.clear.assert_called_once_with(force=False)
