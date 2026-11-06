# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.v1.client import heartbeat
from lmcache.v1.multiprocess import heartbeat as legacy_heartbeat


def test_old_import_path_exposes_the_same_objects() -> None:
    assert legacy_heartbeat.HeartbeatThread is heartbeat.HeartbeatThread
    assert legacy_heartbeat.send_ping is heartbeat.send_ping
