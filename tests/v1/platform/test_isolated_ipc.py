# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.platform.isolated_ipc import is_isolated_ipc, set_isolated_ipc


def test_isolated_ipc_defaults_to_disabled() -> None:
    assert is_isolated_ipc() is False


def test_set_isolated_ipc_round_trip() -> None:
    previous = is_isolated_ipc()
    try:
        set_isolated_ipc(False)
        assert is_isolated_ipc() is False
        set_isolated_ipc(True)
        assert is_isolated_ipc() is True
    finally:
        set_isolated_ipc(previous)
