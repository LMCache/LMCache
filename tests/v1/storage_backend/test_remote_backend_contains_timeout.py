# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class _HangingConnector:
    """Connector whose exists() never completes promptly."""

    async def exists(self, key: CacheEngineKey) -> bool:
        await asyncio.sleep(1.0)
        return True


def _make_key() -> CacheEngineKey:
    """Return a test key.

    Returns:
        A cache engine key.
    """
    return CacheEngineKey("test-model", 1, 0, 0, torch.float16)


def test_contains_returns_false_on_blocking_timeout() -> None:
    """A hung remote contains() must not block the caller indefinitely."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    try:
        backend = RemoteBackend.__new__(RemoteBackend)
        backend.connection = _HangingConnector()
        backend.loop = loop
        backend._mla_worker_id_as0_mode = False
        backend.config = type(
            "Config", (), {"blocking_timeout_secs": 0.01, "extra_config": None}
        )()

        start = time.monotonic()
        assert backend.contains(_make_key()) is False
        assert time.monotonic() - start < 0.5
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()
