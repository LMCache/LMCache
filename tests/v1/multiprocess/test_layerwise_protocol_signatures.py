# SPDX-License-Identifier: Apache-2.0
"""Guard the layer-wise handler signatures against their protocol definitions.

``MessageQueueServer.add_handler`` validates each handler against the declared
``payload_classes`` and ``response_class`` when the MP server boots. A mismatch
therefore does not surface as a failing unit test -- it aborts server startup
with ``ValueError: Handler signature does not match``. These tests move that
check into CI, where a missing or wrong annotation is cheap to find.
"""

# Standard
from types import SimpleNamespace
import inspect
import threading
import types

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer_layerwise import (
    LMCacheLayerwiseTransferModule,
)
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocols.base import RequestType

pytestmark = pytest.mark.layerwise


@pytest.mark.parametrize(
    "request_type,handler_name",
    [
        (RequestType.RETRIEVE_LAYERWISE, "retrieve_layerwise"),
        (
            RequestType.REGISTER_LAYERWISE_IPC_EVENT_POOL,
            "register_layerwise_ipc_event_pool",
        ),
        # The rejecting override stays registered, so it is still validated
        # against the base RETRIEVE protocol at server startup.
        (RequestType.RETRIEVE, "retrieve"),
    ],
)
def test_handler_signature_matches_protocol(request_type, handler_name):
    """Every layer-wise handler must satisfy the server's startup check."""
    func = getattr(LMCacheLayerwiseTransferModule, handler_name)
    # add_handler() receives a bound method, so bind to a sentinel to drop
    # ``self`` from the inspected signature without constructing a module.
    # ``_inspect_handler_signature`` never touches ``self``, so None is fine.
    bound = types.MethodType(func, object())
    assert MessageQueueServer._inspect_handler_signature(None, request_type, bound)


def test_response_channel_is_keyword_only():
    """``response_channel`` must stay keyword-only.

    ``_inspect_handler_signature`` only counts POSITIONAL_ONLY and
    POSITIONAL_OR_KEYWORD parameters against ``payload_classes``. Making this
    parameter positional would add a sixth argument to a five-payload protocol
    and break server startup.
    """
    sig = inspect.signature(LMCacheLayerwiseTransferModule.retrieve_layerwise)
    assert sig.parameters["response_channel"].kind is inspect.Parameter.KEYWORD_ONLY


def test_plain_retrieve_is_rejected_not_raised():
    """A per-chunk RETRIEVE must fail as a value, never as an exception.

    A server node started with ``--layerwise-batch > 0`` serves the layer-wise
    path exclusively. ``REGISTER_KV_CACHE`` is identical for both connectors,
    so a worker running the per-chunk connector only reveals the mismatch on
    its first retrieve. The response must still be a well-formed
    ``(handle, succeeded)`` tuple: ``mq.py`` merely logs an exception escaping
    a blocking handler and sends no reply, which would strand the worker for
    the full ``mq_timeout``.
    """
    module = object.__new__(LMCacheLayerwiseTransferModule)
    module._ctx = SimpleNamespace(layerwise_batch=8)
    released = []
    module._release_failed_retrieve_locks = lambda key, instance_id: released.append(
        (key, instance_id)
    )

    assert module.retrieve("sentinel-key", 7, [[0]], b"producer") == (b"", False)
    assert released == [("sentinel-key", 7)]


def test_plain_retrieve_rejection_survives_lock_release_failure():
    """Cleanup failure must not suppress the terminal response."""
    module = object.__new__(LMCacheLayerwiseTransferModule)
    module._ctx = SimpleNamespace(layerwise_batch=4)

    def _boom(key, instance_id):
        raise RuntimeError("lock release failed")

    module._release_failed_retrieve_locks = _boom

    assert module.retrieve("sentinel-key", 7, [[0]], b"producer") == (b"", False)


def test_retrieve_layerwise_bypasses_the_rejecting_override(monkeypatch):
    """``retrieve_layerwise`` must delegate to the *base* retrieve loop.

    The layer-wise handler reuses the inherited retrieve loop and swaps only
    the copy strategy through ``_transfer_object_group``. Because this class
    also overrides ``retrieve`` to reject the per-chunk request type, that
    delegation has to go through ``super()``: calling ``self.retrieve`` would
    make the layer-wise path reject itself.
    """
    calls = []

    def _base_retrieve(
        self, key, instance_id, gpu_block_ids, event_ipc_handle, skip_first_n_tokens=0
    ):
        calls.append(instance_id)
        return b"completion-handle", True

    monkeypatch.setattr(LMCacheDrivenTransferModule, "retrieve", _base_retrieve)

    module = object.__new__(LMCacheLayerwiseTransferModule)
    module._tls = threading.local()

    result = module.retrieve_layerwise("sentinel-key", 7, [[0]], b"producer")

    # No copy was enqueued, so there is no session and no per-layer events:
    # the handler widens the base pair into its own triple unchanged.
    assert result == (b"completion-handle", True, True)
    assert calls == [7], "retrieve_layerwise did not reach the base retrieve"


def test_a_transport_without_the_layerwise_methods_is_rejected_up_front():
    """A client that cannot stream layers must fail when the worker binds it.

    The layer-wise methods live on ``LayerwiseRequestClient`` rather than on
    ``RequestClient`` precisely so that a transport which never implemented
    them does not inherit empty ``...`` bodies. Were they on the base, a
    non-implementing client would return ``None`` from the middle of a
    retrieve; here it has to be refused while the error can still name the
    flag that caused it.
    """
    # First Party
    from lmcache.v1.multiprocess.transfer_context.worker_transfer_layerwise import (  # noqa: E501
        LMCacheLayerwiseTransferContext,
    )
    from lmcache.v1.multiprocess.transport.base import RequestClient

    class NonLayerwiseClient(RequestClient):
        """Stands in for any transport that serves only the per-chunk path."""

    with pytest.raises(TypeError) as excinfo:
        LMCacheLayerwiseTransferContext(1, NonLayerwiseClient())

    message = str(excinfo.value)
    assert "retrieve_layerwise" in message
    assert "--layerwise-batch" in message, "error must name the flag to change"


def test_the_layerwise_methods_are_off_the_shared_request_client():
    """Keep the layer-wise surface off the transport-neutral protocol.

    Re-adding either method to ``RequestClient`` would hand every transport a
    silent no-op stub again, which is the regression this split exists to
    prevent.
    """
    # First Party
    from lmcache.v1.multiprocess.transport.base import RequestClient
    from lmcache.v1.multiprocess.transport.base_layerwise import (
        LayerwiseRequestClient,
    )

    layerwise_only = {"register_layerwise_ipc_event_pool", "retrieve_layerwise"}

    # ``dir()`` rather than ``__protocol_attrs__``: the latter is a
    # ``typing.Protocol`` internal that only exists on Python 3.12+, and CI runs
    # 3.10 through 3.13. ``dir()`` is also the stricter check here, since it
    # would catch the methods arriving by inheritance as well as by definition.
    assert not (layerwise_only & set(dir(RequestClient))), (
        "layer-wise methods leaked back onto the shared RequestClient"
    )
    assert layerwise_only <= set(dir(LayerwiseRequestClient))


def test_the_zmq_client_implements_the_layerwise_protocol():
    """ZMQ is the transport that serves this path, so it must satisfy it."""
    # First Party
    from lmcache.v1.multiprocess.transport.zmq_impl.client import (
        ZmqMultiprocessClient,
    )

    for name in ("register_layerwise_ipc_event_pool", "retrieve_layerwise"):
        assert name in vars(ZmqMultiprocessClient), (
            f"ZmqMultiprocessClient must define {name} itself"
        )
