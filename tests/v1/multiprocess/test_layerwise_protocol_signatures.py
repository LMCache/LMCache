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
from lmcache.v1.multiprocess.modules.lmcache_layerwise_transfer import (
    LMCacheLayerwiseTransferModule,
)
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocols.base import RequestType


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
