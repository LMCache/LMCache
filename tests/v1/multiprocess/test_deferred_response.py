# SPDX-License-Identifier: Apache-2.0
"""A handler that returns a ``Deferred`` is answered when the ``Deferred``
completes, on every gRPC request server, and its worker is free meanwhile."""

# Standard
from typing import Any
import importlib.util
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.client.transport.grpc_impl.client import GrpcMultiprocessClient
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.grpc_impl.server_builder import (
    build_grpc_request_server,
)
from lmcache.v1.transport.errors import MPRequestError
from lmcache.v1.transport.futures import Deferred, unwrap_deferred_type

_NATIVE = pytest.param(
    "native",
    marks=pytest.mark.skipif(
        importlib.util.find_spec("pygrpc_native") is None,
        reason="pygrpc-native is not installed",
    ),
)


class _Module:
    """``ping`` answers instance 1 when ``pending`` completes, others at once."""

    def __init__(self) -> None:
        self.pending: Deferred[bool] = Deferred()
        self.entered = threading.Event()

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def ping(self, instance_id: int | None) -> Deferred[bool]:
        if instance_id != 1:
            return Deferred.completed(True)
        self.entered.set()
        return self.pending


@pytest.fixture(params=["sync", "aio", _NATIVE])
def served(request: pytest.FixtureRequest) -> Any:
    module = _Module()
    modules: list[Any] = [module]
    server = build_grpc_request_server(
        modules,
        MPServerConfig(
            transport="grpc",
            host="127.0.0.1",
            port=0,
            max_cpu_workers=2,
            max_gpu_workers=1,
            grpc_server_mode=request.param,
        ),
    )
    server.start()
    client = GrpcMultiprocessClient(  # type: ignore[abstract]
        f"grpc://127.0.0.1:{server.bound_port}"
    )
    yield module, client
    client.close()
    server.close()


def test_the_reply_waits_for_the_deferred_and_the_worker_does_not(served) -> None:
    """The one affinity worker serves the client's next request meanwhile."""
    module, client = served
    first = client.ping(1)
    assert module.entered.wait(5)
    assert client.ping(2).result(5) is True
    assert not first.query()
    module.pending.set_result(True)
    assert first.result(5) is True


def test_a_deferred_failure_fails_the_request(served) -> None:
    module, client = served
    first = client.ping(1)
    assert module.entered.wait(5)
    module.pending.set_exception(RuntimeError("late failure"))
    with pytest.raises(MPRequestError, match="late failure"):
        first.result(5)


def test_the_wire_type_of_a_deferred_annotation_is_its_argument() -> None:
    assert unwrap_deferred_type(Deferred[tuple[bytes, bool]]) == tuple[bytes, bool]
    assert unwrap_deferred_type(tuple[bytes, bool]) == tuple[bytes, bool]
    assert unwrap_deferred_type(bool) is bool
