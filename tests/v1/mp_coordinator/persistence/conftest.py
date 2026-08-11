# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the persistence tests."""

# Standard
from collections.abc import Callable, Iterator
import logging

# Third Party
import pytest


class _Collector(logging.Handler):
    """Handler keeping every message it is given."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Record ``record``'s formatted message."""
        self.messages.append(record.getMessage())


@pytest.fixture
def logged() -> Iterator[Callable[[logging.Logger], list[str]]]:
    """Collect messages from a module logger.

    ``caplog`` cannot do this for an lmcache logger:
    ``lmcache.logging.init_logger`` disables propagation, and it applies
    ``LMCACHE_LOG_LEVEL`` to the logger itself — CI sets that to
    ``CRITICAL``, so records are dropped before any handler runs and
    ``caplog.at_level`` (which raises the *root* level) has no effect.
    Attaching here, and lowering the logger's own level, works either way.

    Yields:
        A function taking a module's ``logger`` and returning the list its
        messages will land in.
    """
    attached: list[tuple[logging.Logger, _Collector, int]] = []

    def attach(module_logger: logging.Logger) -> list[str]:
        collector = _Collector()
        attached.append((module_logger, collector, module_logger.level))
        module_logger.setLevel(logging.DEBUG)
        module_logger.addHandler(collector)
        return collector.messages

    yield attach
    for module_logger, collector, level in attached:
        module_logger.removeHandler(collector)
        module_logger.setLevel(level)
