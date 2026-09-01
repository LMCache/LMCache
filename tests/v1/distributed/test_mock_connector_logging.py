# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import logging

# First Party
from lmcache.v1.storage_backend.connector import mock_connector
from lmcache.v1.storage_backend.connector.mock_connector import MockConnector


class _RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _FakeLRU:
    def __init__(self, objs):
        self._objs = objs

    async def batched_get(self, keys):
        return self._objs


class _FakePressureManager:
    async def on_batched_get(self, mock_objs):
        return None


def test_batched_get_missing_object_logs_render():
    """Regression for #4539 case 2.

    ``_batched_get`` warned with ``logger.warning(f"...{i}", f" ...{n}")``. The
    stray comma turned the message tail into a positional arg, so rendering the
    record as ``msg % args`` raised ``TypeError`` inside the logging machinery
    and the intended message was dropped. Ruff PLE1205/PLE1206 cannot see
    through the f-string here, so guard the format string with a test that
    renders every emitted record.
    """
    handler = _RecordingHandler()
    logger = mock_connector.logger
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        conn = MockConnector.__new__(MockConnector)
        conn.lru_store = _FakeLRU([None])
        conn.pressure_manager = _FakePressureManager()

        result = asyncio.run(conn._batched_get(["dummy-key"]))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    assert result == []
    # getMessage() would raise TypeError before the fix; it must render cleanly.
    messages = [record.getMessage() for record in handler.records]
    assert "Mock object is None on 0 out of 1 objects" in messages
