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

Abstract base classes live in :mod:`lmcache.v1.platform.base`, one per
``.py`` file.  The universal registry
(:mod:`lmcache.v1.platform._registry`) scans that package automatically
and discovers concrete subclasses in the device sub-packages keyed by
their ``device_type`` ClassVar.  Adding a new base class only requires
dropping a new file in ``platform/base/``; adding a new device
implementation only requires adding a subclass file in the device
sub-package.  No other code changes are needed.

Backend availability predicates are registered from each device
sub-package's ``__init__.py``.  The sub-packages are auto-imported here
so those side effects fire at startup.
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

    Each device backend sub-package registers availability predicates from
    its ``__init__.py``.  The ``base/`` sub-package is intentionally
    included so its modules are importable; it carries no side effects.
    Importing the sub-packages is enough -- we deliberately do **not**
    force-import the heavy ``cache_context`` leaf module here, so platform
    bootstrap stays free of the circular import chain through
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
