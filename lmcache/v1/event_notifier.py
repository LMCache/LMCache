# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-export.

This module has been moved to ``lmcache.v1.platform.event_notifier``.
All symbols are re-exported here so that existing imports continue
to work.
"""

# First Party
from lmcache.v1.platform.event_notifier import (  # noqa: F401
    EventfdNotifier,
    EventNotifier,
    PipeNotifier,
    consume_fd,
    create_event_notifier,
)
