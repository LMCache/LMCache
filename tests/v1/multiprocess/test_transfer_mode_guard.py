# SPDX-License-Identifier: Apache-2.0
"""Transfer-mode mismatch registration returns a useful RPC error."""

from unittest.mock import MagicMock

import pytest
import zmq

from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
from lmcache.v1.multiprocess.modules.transfer_mode_guard import (
    create_transfer_mode_guard,
)
from lmcache.v1.multiprocess.request_handler import iter_request_handlers
from lmcache.v1.multiprocess.rpc import get_rpc_spec
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    MessageQueueServer,
    RemoteHandlerError,
)
from lmcache.v1.multiprocess.transport.zmq_impl.server import add_handler_helper


@pytest.mark.parametrize(
    ("supported", "rejected"),
    [
        ("engine_driven", {"register_kv_cache", "register_q_cache"}),
        ("lmcache_driven", {"register_kv_cache_engine_driven_context"}),
        ("auto", set()),
    ],
)
def test_guard_registers_only_unsupported_modes(supported, rejected):
    module = create_transfer_mode_guard(MagicMock(), supported)
    assert {entry.operation for entry in iter_request_handlers(module)} == rejected
    assert module.report_status() == {"supported_transfer_mode": supported}


@pytest.mark.parametrize(
    ("supported", "operation", "requested"),
    [
        ("engine_driven", "register_kv_cache", "lmcache_driven"),
        ("engine_driven", "register_q_cache", "lmcache_driven"),
        (
            "lmcache_driven",
            "register_kv_cache_engine_driven_context",
            "engine_driven",
        ),
    ],
)
def test_guard_error_names_requested_and_supported_modes(
    supported, operation, requested
):
    module = create_transfer_mode_guard(MagicMock(), supported)
    handler = {
        entry.operation: entry.handler for entry in iter_request_handlers(module)
    }[operation]
    with pytest.raises(
        ValueError,
        match=rf"requested transfer mode '{requested}'.*supported_transfer_mode='{supported}'",
    ):
        handler(*([None] * len(get_rpc_spec(operation).payload_types)))


def test_guard_rejects_unknown_server_mode():
    with pytest.raises(ValueError, match="Unsupported supported_transfer_mode"):
        create_transfer_mode_guard(MagicMock(), "invalid")


def test_mode_mismatch_reaches_client_without_timeout():
    context = zmq.Context()
    server = MessageQueueServer("tcp://127.0.0.1:*", context)
    server_url = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    module = create_transfer_mode_guard(MagicMock(), "lmcache_driven")
    for entry in iter_request_handlers(module):
        add_handler_helper(server, entry.operation, entry.handler)
    server.start()
    client = RequestClientFactory.create(server_url, context=context)
    try:
        future = client.register_kv_cache_engine_driven_context(
            RegisterEngineDrivenContextPayload(
                instance_id=7,
                model_name="test-model",
                world_size=1,
                block_size=16,
                num_layers=1,
                hidden_dim_size=128,
                dtype_str="float16",
                use_mla=False,
                num_physical_slots=16,
            )
        )
        with pytest.raises(
            RemoteHandlerError,
            match="requested transfer mode 'engine_driven'.*"
            "supported_transfer_mode='lmcache_driven'",
        ):
            future.result(timeout=1)
    finally:
        client.close()
        server.close()
        context.term()
