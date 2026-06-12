# SPDX-License-Identifier: Apache-2.0
"""Cross-platform abstraction layer for LMCache.

This package centralizes platform-specific primitives. It currently
exposes :class:`EventNotifier` -- a thin wake-up primitive used to
signal background loops from other threads.  On Linux it is backed by
``os.eventfd``; on macOS / other POSIX systems it falls back to
``os.pipe``.  Callers never touch ``os.eventfd`` directly.

Accelerator- and OS-specific implementations live in dedicated sub-
packages so each can evolve independently:

* :mod:`lmcache.v1.platform.cuda` -- CUDA-backed implementations.
* :mod:`lmcache.v1.platform.cpu`  -- CPU-only fallbacks.

Importing them here triggers their self-registration with
:mod:`lmcache.v1.platform._registry`, so the multiprocess adapter
can dispatch by ``tensor.device.type`` without any if/elif chain.
"""

# First Party
from lmcache.v1.platform.event_notifier import HAS_EVENTFD as HAS_EVENTFD
from lmcache.v1.platform.event_notifier import EventfdNotifier as EventfdNotifier
from lmcache.v1.platform.event_notifier import EventNotifier as EventNotifier
from lmcache.v1.platform.event_notifier import PipeNotifier as PipeNotifier
from lmcache.v1.platform.event_notifier import consume_fd as consume_fd
from lmcache.v1.platform.event_notifier import (
    create_event_notifier as create_event_notifier,
)

# Trigger backend self-registration with :mod:`_registry`.
import lmcache.v1.platform.cpu  # noqa: F401,E402
import lmcache.v1.platform.cuda  # noqa: F401,E402
