# SPDX-License-Identifier: Apache-2.0
"""Tests for explicit MP transfer-mode mismatch registration failures."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import zmq

# First Party
from lmcache.v1.multiprocess.custom_types import (
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.modules.transfer_mode_guard import (
    TransferModeGuardModule,
)
from lmcache.v1.multiprocess.mq import (
    MessageQueueServer,
    RemoteHandlerError,
)
from lmcache.v1.multiprocess.protocol import get_payload_classes
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.server import add_handler_helper
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory


@pytest.mark.parametrize(
    ("supported", "rejected"),
    [
        (
            "engine_driven",
            {
                RequestType.REGISTER_KV_CACHE,
                RequestType.REGISTER_Q_CACHE,
            },
        ),
        (
            "lmcache_driven",
            {RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT},
        ),
        ("auto", set()),
    ],
)
def test_guard_registers_only_unsupported_modes(
    supported: str, rejected: set[RequestType]
) -> None:
    module = TransferModeGuardModule(MagicMock(), supported)

    assert {spec.request_type for spec in module.get_handlers()} == rejected
    assert module.report_status() == {"supported_transfer_mode": supported}


@pytest.mark.parametrize(
    ("supported", "request_type", "requested"),
    [
        ("engine_driven", RequestType.REGISTER_KV_CACHE, "lmcache_driven"),
        ("engine_driven", RequestType.REGISTER_Q_CACHE, "lmcache_driven"),
        (
            "lmcache_driven",
            RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            "engine_driven",
        ),
    ],
)
def test_guard_error_names_requested_and_supported_modes(
    supported: str,
    request_type: RequestType,
    requested: str,
) -> None:
    module = TransferModeGuardModule(MagicMock(), supported)
    handler = {spec.request_type: spec.handler for spec in module.get_handlers()}[
        request_type
    ]

    with pytest.raises(
        ValueError,
        match=(
            rf"requested transfer mode '{requested}'.*"
            rf"supported_transfer_mode='{supported}'"
        ),
    ):
        handler(*([None] * len(get_payload_classes(request_type))))


def test_guard_rejects_unknown_server_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported supported_transfer_mode"):
        TransferModeGuardModule(MagicMock(), "invalid")


def test_mode_mismatch_reaches_client_without_timeout() -> None:
    """A mismatched registration completes with the server's useful error."""
    context = zmq.Context()
    server = MessageQueueServer("tcp://127.0.0.1:*", context)
    server_url = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    module = TransferModeGuardModule(MagicMock(), "lmcache_driven")
    for spec in module.get_handlers():
        add_handler_helper(server, spec.request_type, spec.handler)
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
            match=(
                "requested transfer mode 'engine_driven'.*"
                "supported_transfer_mode='lmcache_driven'"
            ),
        ):
            future.result(timeout=1)
    finally:
        client.close()
        server.close()
        context.term()
