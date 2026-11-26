# SPDX-License-Identifier: Apache-2.0
"""Peer clients shared by several adapters: removed only by the last user."""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.transfer_channel import (
    acquire_transfer_channel_client,
    release_transfer_channel_client,
)


def test_client_is_removed_only_after_its_last_user_releases_it() -> None:
    context = MagicMock()
    context.get_transfer_channel_client.side_effect = lambda url: f"client:{url}"
    removed = context.remove_transfer_channel_client
    first = acquire_transfer_channel_client(context, "peer:1")
    assert (
        first == acquire_transfer_channel_client(context, "peer:1") == "client:peer:1"
    )
    acquire_transfer_channel_client(context, "peer:2")  # counted per peer
    release_transfer_channel_client(context, "peer:9")  # never acquired
    release_transfer_channel_client(context, "peer:1")
    removed.assert_not_called()
    release_transfer_channel_client(context, "peer:1")
    removed.assert_called_once_with("peer:1")
    # A peer acquired again after removal starts a fresh count.
    acquire_transfer_channel_client(context, "peer:1")
    release_transfer_channel_client(context, "peer:1")
    release_transfer_channel_client(context, "peer:2")
    assert [c.args[0] for c in removed.call_args_list] == ["peer:1", "peer:1", "peer:2"]
