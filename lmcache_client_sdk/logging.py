# SPDX-License-Identifier: Apache-2.0
"""Logging helpers for the standalone ``lmcache_client_sdk`` package.

The package is self-contained and must not import from the full ``lmcache``
distribution, so this module provides a minimal logger factory that mirrors
the project's ``init_logger`` convention with the standard library only.
"""

# Standard
import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def init_logger(name: str) -> logging.Logger:
    """Return a named logger with a sane default configuration.

    The logger is created lazily; configuration is applied once per process
    through the root logger.

    Args:
        name: Logger name, typically the calling module's ``__name__``.

    Returns:
        A standard library logger emitting to stderr.
    """
    logging.basicConfig(level=logging.INFO, format=_FORMAT)
    return logging.getLogger(name)
