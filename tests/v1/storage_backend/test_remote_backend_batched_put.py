# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
import threading

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


def _make_key(chunk_hash: int) -> CacheEngineKey:
    """Create a test key with the given chunk hash.

    Args:
        chunk_hash: Chunk hash for the key.

    Returns:
        A cache engine key.
    """
    return CacheEngineKey("test-model", 1, 0, chunk_hash, torch.float16)


def _make_backend() -> RemoteBackend:
    """Create a RemoteBackend without running its constructor.

    Returns:
        A backend with only the state used by the batched-put callbacks.
    """
    backend = RemoteBackend.__new__(RemoteBackend)
    backend.lock = threading.Lock()
    backend.put_tasks = set()
    backend._put_failed_count = 0
    backend.loop = None
    return backend


def test_batched_put_failure_is_counted_and_clears_tasks() -> None:
    """A failed batch must count failures and clear in-flight keys."""
    backend = _make_backend()
    key = _make_key(0)
    backend.put_tasks.add(key)
    callback_calls = []

    future: Future = Future()
    future.set_exception(RuntimeError("boom"))

    backend.batched_put_callback(future, [key], lambda k: callback_calls.append(k))

    assert backend.put_tasks == set()
    assert backend._put_failed_count == 1
    assert callback_calls == []


def test_batched_put_success_calls_callback_once_per_key() -> None:
    """A successful batch must invoke the per-key callback exactly once."""
    backend = _make_backend()
    keys = [_make_key(0), _make_key(1)]
    backend.put_tasks.update(keys)
    callback_calls = []

    future: Future = Future()
    future.set_result(None)

    backend.batched_put_callback(future, keys, lambda k: callback_calls.append(k))

    assert backend.put_tasks == set()
    assert backend._put_failed_count == 0
    assert callback_calls == keys


def test_batched_submit_registers_keys_before_scheduling(
    monkeypatch,
) -> None:
    """Keys must be visible in put_tasks when the batch is scheduled."""
    backend = _make_backend()
    keys = [_make_key(0)]
    observed = {}

    class _Conn:
        def support_batched_put(self) -> bool:
            return True

        async def batched_put(self, keys, objs) -> None:
            return None

    class _Serializer:
        def serialize(self, obj):
            return obj

    class _Obj:
        def ref_count_up(self) -> None:
            pass

        def ref_count_down(self) -> None:
            pass

    backend.connection = _Conn()
    backend.serializer = _Serializer()
    backend._mla_worker_id_as0_mode = False

    def fake_run(coro, loop):
        observed["put_tasks"] = set(backend.put_tasks)
        coro.close()
        future: Future = Future()
        future.set_result(None)
        return future

    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", fake_run)

    backend.batched_submit_put_task(keys, [_Obj()])

    assert observed["put_tasks"] == set(keys)
