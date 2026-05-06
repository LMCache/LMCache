# SPDX-License-Identifier: Apache-2.0
"""Cross-platform abstraction layer for LMCache.

This package centralizes platform-specific primitives.  The top-level
exports here stay device-agnostic; accelerator- and OS-specific
implementations live in dedicated sub-packages so each can evolve
independently:

* ``event_notifier``           -- OS-level wake-up primitive
  (Linux ``os.eventfd`` / POSIX ``os.pipe`` fallback).
* :mod:`lmcache.v1.platform.stream` -- ``ExternalStreamLike`` protocol
  + dispatcher choosing a backend at runtime.
* :mod:`lmcache.v1.platform.cuda` -- CUDA / cupy-backed implementations.
* :mod:`lmcache.v1.platform.cpu`  -- CPU-only fallbacks.

Future accelerators (``xpu``, ``hpu``, ...) plug in by adding a sibling
sub-package next to ``cuda/`` and ``cpu/``.
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
from lmcache.v1.platform.stream import ExternalStreamLike as ExternalStreamLike
from lmcache.v1.platform.stream import make_external_stream as make_external_stream
