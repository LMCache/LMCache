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

Backend availability is filesystem-driven: every direct sub-package
below ``platform/`` is auto-imported here, which fires lightweight
``__init__.py`` side effects such as ``register_availability``.
KV-cache IPC wrappers and ``BaseCacheContext`` subclasses are
discovered separately on first use via
:mod:`lmcache.v1.utils.subclass_discovery`, keyed by each subclass'
``device_type`` ClassVar.  Adding a new accelerator therefore
requires *zero* edits to this module -- drop a new
``platform/<backend>/`` package and it will be picked up
automatically.
"""

# Standard
import importlib
import pkgutil

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.event_notifier import HAS_EVENTFD as HAS_EVENTFD
from lmcache.v1.platform.event_notifier import EventfdNotifier as EventfdNotifier
from lmcache.v1.platform.event_notifier import EventNotifier as EventNotifier
from lmcache.v1.platform.event_notifier import PipeNotifier as PipeNotifier
from lmcache.v1.platform.event_notifier import consume_fd as consume_fd
from lmcache.v1.platform.event_notifier import (
    create_event_notifier as create_event_notifier,
)

logger = init_logger(__name__)


def _bootstrap_backends() -> None:
    """Import every direct sub-package under ``lmcache.v1.platform``.

    Backend ``__init__.py`` files should keep side effects lightweight
    (for example, availability predicates). KV-cache wrapper and
    cache-context classes are discovered from leaf modules lazily, so
    platform bootstrap stays free of the circular import chain through
    ``lmcache.gpu_connector``.
    """
    for _, short_name, is_pkg in pkgutil.iter_modules(__path__):
        if not is_pkg:
            continue
        full_name = "%s.%s" % (__name__, short_name)
        try:
            importlib.import_module(full_name)
        except Exception as exc:
            logger.warning("Failed to import platform backend %s: %s", full_name, exc)


_bootstrap_backends()
