# SPDX-License-Identifier: Apache-2.0
"""FD channel availability must not prevent ordinary MP initialization."""

# Standard
from pathlib import Path
from unittest.mock import MagicMock
import logging
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod

pytestmark = pytest.mark.no_shared_allocator


def test_unsupported_fd_channel_does_not_log_startup_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A macOS server can initialize and close without an optional FD channel."""
    monkeypatch.setattr(sys, "platform", "darwin")
    logger = MagicMock(spec=logging.Logger)
    monkeypatch.setattr(mod, "logger", logger)
    module = mod.LMCacheDrivenTransferModule(
        MagicMock(), fd_socket_path=str(tmp_path / "fd.sock")
    )
    try:
        assert module.report_status()["registered_gpu_ids"] == []
    finally:
        module.close()

    logger.warning.assert_not_called()
    logger.error.assert_not_called()
    logger.exception.assert_not_called()
    assert any(
        "Skipping the fd channel" in call.args[0] for call in logger.info.call_args_list
    )


def test_supported_fd_channel_keeps_operational_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An OSError from channel startup is logged with its original traceback."""
    monkeypatch.setattr(sys, "platform", "linux")
    error = OSError("cannot bind fd channel")

    def check_exception_context(message: str, *args: object) -> None:
        assert sys.exc_info()[1] is error
        assert sys.exc_info()[2] is not None

    logger = MagicMock(spec=logging.Logger)
    logger.exception.side_effect = check_exception_context
    monkeypatch.setattr(mod, "logger", logger)
    monkeypatch.setattr(mod, "FdChannelServer", MagicMock(side_effect=error))
    module = mod.LMCacheDrivenTransferModule(
        MagicMock(), fd_socket_path=str(tmp_path / "fd.sock")
    )
    module.close()

    logger.exception.assert_called_once()
    assert "Cannot serve the fd channel" in logger.exception.call_args.args[0]
