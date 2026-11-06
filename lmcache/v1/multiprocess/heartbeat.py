# SPDX-License-Identifier: Apache-2.0
"""Compatibility path for :mod:`lmcache.v1.client.heartbeat`."""

# First Party
from lmcache.v1.client.heartbeat import HeartbeatThread, send_ping

__all__ = ["HeartbeatThread", "send_ping"]
