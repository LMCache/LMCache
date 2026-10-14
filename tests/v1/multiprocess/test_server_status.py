# SPDX-License-Identifier: Apache-2.0
"""Tests for the multiprocess server status response."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.server import MPCacheServer


@pytest.mark.parametrize("configured_engine", ["default", "blend"])
def test_report_status_includes_configured_engine(configured_engine: str) -> None:
    context = MagicMock()
    context.storage_manager.report_status.return_value = {"is_healthy": True}

    server = MPCacheServer(context, modules=[], configured_engine=configured_engine)
    status = server.report_status()

    assert status["configured_engine"] == configured_engine
    assert status["engine_type"] == "MPCacheServer"
