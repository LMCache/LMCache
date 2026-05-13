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
from lmcache.v1.platform.device_ctx import (
    InterprocessEventLike as InterprocessEventLike,
)
from lmcache.v1.platform.device_ctx import (
    event_from_ipc_handle as event_from_ipc_handle,
)
from lmcache.v1.platform.device_ctx import make_device_context as make_device_context
from lmcache.v1.platform.device_ctx import (
    make_interprocess_event as make_interprocess_event,
)
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

# Trigger backend self-registration with :mod:`_registry`.  The default
# (``cpu``) backend must register first so accelerator backends can
# transparently fall back to it.  Both imports are kept side-effect-only
# so callers never need to know which sub-package is active.
import lmcache.v1.platform.cpu  # noqa: F401,E402  pylint: disable=wrong-import-position
import lmcache.v1.platform.cuda  # noqa: F401,E402  pylint: disable=wrong-import-position

# Note: torch.cuda / cupy monkey-patching has been replaced by the
# explicit strategy-dispatch helpers in :mod:`device_ctx` and
# :mod:`stream` so no global patching is required at import time.


def __getattr__(name: str) -> object:
    """Lazy re-export of platform cache utilities.

    Deferred so that ``lmcache.c_ops`` (a compiled extension)
    is fully available by the time the class is first used.
    """
    if name == "CpuCacheContext":
        # First Party
        from lmcache.v1.platform.cache_context import (
            CpuCacheContext,
        )

        return CpuCacheContext

    if name == "create_cache_context":
        # First Party
        from lmcache.v1.platform.cache_context import (
            create_cache_context,
        )

        return create_cache_context

    raise AttributeError(name)
