# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the native S3 connector's error handling.

``awscrt`` is a real LMCache dependency, so only the ``s3.S3Request``
boundary is stubbed here.  The stub reproduces the two properties of the
real completion path that these tests are about (see
``awscrt.s3._S3RequestCore._on_finish``):

1. ``finished_future`` is resolved *before* ``on_done`` is invoked, so an
   exception raised by ``on_done`` can never reach the awaiting coroutine.
2. ``on_done`` runs on an aws-crt thread, not on the event loop, so such an
   exception is handed to the unraisable hook and printed as
   ``Exception ignored in: <awscrt.s3._S3RequestCore object at 0x...>``.

Driving that boundary needs no S3 endpoint, no credentials, no GPU and no
network.
"""

# Standard
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, List, Optional
import asyncio
import threading

# Third Party
from awscrt import s3
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, start_loop_in_thread_with_exceptions
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.s3_connector import S3Connector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


@dataclass
class Completion:
    """One scripted aws-crt completion for a stubbed ``s3.S3Request``.

    Attributes:
        status_code: HTTP status handed to ``on_done`` (and to ``on_headers``
            for HEAD requests). ``None`` models the "unknown status" that
            aws-crt reports when the C layer gives status 0.
        error: Exception aws-crt would set on ``finished_future``. ``None``
            models a request that aws-crt considers successful, whatever the
            HTTP status is.
        content_length: Value of the ``Content-Length`` response header,
            reported through ``on_headers``. Only used by HEAD requests.
        body: Response body delivered through ``on_body``. Only used by GET
            requests.
    """

    status_code: Optional[int] = 200
    error: Optional[Exception] = None
    content_length: Optional[int] = None
    body: bytes = b""


class StubS3Request:
    """Stand-in for a single in-flight ``s3.S3Request``.

    The request completes immediately, on a separate thread, the way aws-crt
    completes one on its own IO threads.
    """

    def __init__(self, kwargs: dict[str, Any], completion: Completion) -> None:
        """Record the request and complete it with the scripted outcome.

        Args:
            kwargs: Keyword arguments the connector passed to ``s3.S3Request``.
            completion: The outcome to deliver to the connector's callbacks.
        """
        self.kwargs = kwargs
        self.completion = completion
        self.finished_future: Future = Future()
        # Exception raised by ``on_done``, which aws-crt swallows.
        self.callback_error: Optional[Exception] = None

        thread = threading.Thread(target=self._complete, name="stub-aws-crt")
        thread.start()
        thread.join()

    def _complete(self) -> None:
        """Deliver headers, body and completion, in aws-crt's order."""
        completion = self.completion

        on_headers = self.kwargs.get("on_headers")
        if on_headers is not None:
            headers = []
            if completion.content_length is not None:
                headers.append(("Content-Length", str(completion.content_length)))
            on_headers(completion.status_code, headers)

        on_body = self.kwargs.get("on_body")
        if on_body is not None and completion.body:
            on_body(completion.body, 0)

        # The future is resolved before ``on_done`` runs, exactly as
        # ``_S3RequestCore._on_finish`` does it.
        if completion.error is not None:
            self.finished_future.set_exception(completion.error)
        else:
            self.finished_future.set_result(None)

        on_done = self.kwargs.get("on_done")
        if on_done is not None:
            try:
                on_done(error=completion.error, status_code=completion.status_code)
            except Exception as exc:
                # aws-crt cannot propagate this: the future is already
                # resolved and this is not the event loop thread. Record it
                # so tests can assert the connector does not rely on it.
                self.callback_error = exc


@dataclass
class StubS3RequestFactory:
    """Replacement for ``s3.S3Request`` that completes every request at once.

    Completions are scripted per ``s3.S3RequestType``. When several are
    scripted for a type they are consumed in order and the last one stays in
    effect for any further request of that type. Unscripted types get a
    successful completion.

    Attributes:
        completions: Scripted completions, keyed by request type.
        requests: Every request the connector has constructed, in order.
    """

    completions: dict[Any, List[Completion]] = field(default_factory=dict)
    requests: List[StubS3Request] = field(default_factory=list)

    def script(self, request_type: Any, *completions: Completion) -> None:
        """Queue completions for a request type.

        Args:
            request_type: The ``s3.S3RequestType`` to script.
            completions: Completions to deliver, in order.
        """
        self.completions.setdefault(request_type, []).extend(completions)

    def of_type(self, request_type: Any) -> List[StubS3Request]:
        """Return the recorded requests of one type.

        Args:
            request_type: The ``s3.S3RequestType`` to filter on.

        Returns:
            The matching requests, in construction order.
        """
        return [r for r in self.requests if r.kwargs.get("type") is request_type]

    @property
    def callback_errors(self) -> List[Exception]:
        """Return the exceptions raised by ``on_done`` callbacks."""
        return [r.callback_error for r in self.requests if r.callback_error is not None]

    def __call__(self, **kwargs: Any) -> StubS3Request:
        """Construct and immediately complete a stubbed request.

        Args:
            kwargs: The keyword arguments the connector passes through.

        Returns:
            The completed stub request.
        """
        queued = self.completions.get(kwargs.get("type"), [])
        if len(queued) > 1:
            completion = queued.pop(0)
        elif queued:
            completion = queued[0]
        else:
            completion = Completion()

        request = StubS3Request(kwargs, completion)
        self.requests.append(request)
        return request


def create_test_metadata(kv_shape=(1, 2, 16, 8, 128), chunk_size=16) -> LMCacheMetadata:
    """Build metadata for a small CPU-only chunk layout."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        chunk_size=chunk_size,
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Build a cache key that is unique per ``key_id``."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


@pytest.fixture
def async_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions,
        args=(loop,),
        name="test-s3-loop",
    )
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


@pytest.fixture
def local_cpu_backend(memory_allocator):
    config = LMCacheEngineConfig.from_legacy(chunk_size=16)
    metadata = create_test_metadata()
    return LocalCPUBackend(config, metadata, memory_allocator=memory_allocator)


@pytest.fixture
def stub_s3_request(monkeypatch):
    """Replace ``s3.S3Request`` for the duration of one test."""
    factory = StubS3RequestFactory()
    monkeypatch.setattr(s3, "S3Request", factory)
    return factory


@pytest.fixture
def connector(async_loop, local_cpu_backend, stub_s3_request):
    conn = S3Connector(
        s3_endpoint="s3://test-bucket.s3.example.com",
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        s3_num_io_threads=1,
        s3_prefer_http2=False,
        s3_region="us-east-1",
        s3_enable_s3express=False,
        disable_tls=True,
        aws_access_key_id="test-access-key-id",
        aws_secret_access_key="test-secret-access-key",
    )
    yield conn
    # Drain the job executor on the loop before the loop is stopped, so its
    # workers are not left pending.
    run(async_loop, conn.pq_executor.shutdown_async(wait=True))


def run(loop, coro):
    """Run a coroutine on the connector's loop and wait for it."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=10.0)


def allocate_chunk(connector, local_cpu_backend) -> MemoryObj:
    """Allocate one full chunk, the same way ``get`` does."""
    memory_obj = local_cpu_backend.allocate(
        connector.meta_shapes,
        connector.meta_dtypes,
        connector.meta_fmt,
    )
    assert memory_obj is not None
    return memory_obj


def script_head_hit(stub_s3_request, connector) -> None:
    """Script a HEAD response advertising one full chunk."""
    stub_s3_request.script(
        s3.S3RequestType.DEFAULT,
        Completion(status_code=200, content_length=connector.full_chunk_size_bytes),
    )


class TestErrorPropagation:
    """Failures reported by aws-crt must reach the awaiting coroutine."""

    def test_put_http_error_is_not_swallowed(
        self, connector, async_loop, local_cpu_backend, stub_s3_request
    ):
        stub_s3_request.script(s3.S3RequestType.PUT_OBJECT, Completion(status_code=400))
        key = create_test_key(1)
        memory_obj = allocate_chunk(connector, local_cpu_backend)

        run(async_loop, connector.put(key, memory_obj))

        # Raising from the completion callback is a no-op, so the connector
        # must not depend on it.
        assert stub_s3_request.callback_errors == []
        # A failed upload must not be recorded as a stored object, otherwise
        # later lookups treat the key as a hit that S3 cannot serve.
        assert key.to_string() not in connector.object_size_cache
        memory_obj.ref_count_down()

    def test_get_http_error_is_not_swallowed(
        self, connector, async_loop, stub_s3_request
    ):
        script_head_hit(stub_s3_request, connector)
        stub_s3_request.script(s3.S3RequestType.GET_OBJECT, Completion(status_code=403))

        result = run(async_loop, connector.get(create_test_key(2)))

        assert stub_s3_request.callback_errors == []
        # A rejected download must not be handed back as cache content.
        assert result is None

    def test_batched_get_http_error_is_not_swallowed(
        self, connector, async_loop, stub_s3_request
    ):
        script_head_hit(stub_s3_request, connector)
        stub_s3_request.script(s3.S3RequestType.GET_OBJECT, Completion(status_code=403))
        keys = [create_test_key(3), create_test_key(4)]

        results = run(async_loop, connector.batched_get(keys))

        assert stub_s3_request.callback_errors == []
        assert results == [None, None]


class TestCircuitBreaker:
    """Persistent HTTP failures must trip the breaker, 404s must not."""

    def test_http_error_counts_toward_breaker(
        self, connector, async_loop, local_cpu_backend, stub_s3_request
    ):
        stub_s3_request.script(s3.S3RequestType.PUT_OBJECT, Completion(status_code=400))
        memory_obj = allocate_chunk(connector, local_cpu_backend)

        run(async_loop, connector.put(create_test_key(5), memory_obj))

        assert connector.connection_failures == 1
        memory_obj.ref_count_down()

    def test_repeated_http_errors_disable_the_connection(
        self, connector, async_loop, local_cpu_backend, stub_s3_request
    ):
        stub_s3_request.script(s3.S3RequestType.PUT_OBJECT, Completion(status_code=400))

        for key_id in range(connector.max_connection_failures):
            memory_obj = allocate_chunk(connector, local_cpu_backend)
            run(async_loop, connector.put(create_test_key(key_id), memory_obj))
            memory_obj.ref_count_down()

        assert connector.connection_disabled is True

        # Once disabled, no further request may be issued.
        issued = len(stub_s3_request.of_type(s3.S3RequestType.PUT_OBJECT))
        memory_obj = allocate_chunk(connector, local_cpu_backend)
        run(async_loop, connector.put(create_test_key(99), memory_obj))
        assert len(stub_s3_request.of_type(s3.S3RequestType.PUT_OBJECT)) == issued
        memory_obj.ref_count_down()

    def test_crt_error_with_http_status_counts_toward_breaker(
        self, connector, async_loop, local_cpu_backend, stub_s3_request
    ):
        # What a real backend produces: aws-crt fails the request with an
        # error whose message carries no connection-level keyword, so only
        # the HTTP status identifies it as non-retryable.
        stub_s3_request.script(
            s3.S3RequestType.PUT_OBJECT,
            Completion(
                status_code=400,
                error=RuntimeError("AWS_ERROR_S3_INVALID_RESPONSE_STATUS"),
            ),
        )
        memory_obj = allocate_chunk(connector, local_cpu_backend)

        run(async_loop, connector.put(create_test_key(6), memory_obj))

        assert connector.connection_failures == 1
        memory_obj.ref_count_down()

    def test_head_miss_does_not_count_toward_breaker(
        self, connector, async_loop, stub_s3_request
    ):
        # A 404 is an expected cache miss, not a broken connection.
        stub_s3_request.script(
            s3.S3RequestType.DEFAULT,
            Completion(status_code=404, error=RuntimeError("404 Not Found")),
        )
        key = create_test_key(7)

        assert run(async_loop, connector.exists(key)) is False
        assert run(async_loop, connector.get(key)) is None

        assert connector.connection_failures == 0
        assert connector.connection_disabled is False
        # The miss must short-circuit before any download is issued.
        assert stub_s3_request.of_type(s3.S3RequestType.GET_OBJECT) == []


class TestSuccessPaths:
    """Successful transfers must keep working unchanged."""

    @pytest.mark.parametrize("status_code", [200, 201])
    def test_put_success(
        self, connector, async_loop, local_cpu_backend, stub_s3_request, status_code
    ):
        stub_s3_request.script(
            s3.S3RequestType.PUT_OBJECT, Completion(status_code=status_code)
        )
        key = create_test_key(8)
        memory_obj = allocate_chunk(connector, local_cpu_backend)

        run(async_loop, connector.put(key, memory_obj))

        assert connector.object_size_cache[key.to_string()] == (
            memory_obj.get_physical_size()
        )
        assert connector.connection_failures == 0
        assert stub_s3_request.callback_errors == []
        memory_obj.ref_count_down()

    @pytest.mark.parametrize("status_code", [200, 206, None])
    def test_get_success(self, connector, async_loop, stub_s3_request, status_code):
        # ``status_code is None`` is aws-crt's "unknown status", which the
        # connector treats as success.
        payload = b"lmcache" * 16
        script_head_hit(stub_s3_request, connector)
        stub_s3_request.script(
            s3.S3RequestType.GET_OBJECT,
            Completion(status_code=status_code, body=payload),
        )

        result = run(async_loop, connector.get(create_test_key(9)))

        assert isinstance(result, MemoryObj)
        assert bytes(result.byte_array)[: len(payload)] == payload
        assert connector.connection_failures == 0
        assert stub_s3_request.callback_errors == []
        result.ref_count_down()
