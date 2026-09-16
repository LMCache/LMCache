# SPDX-License-Identifier: Apache-2.0
"""Public transfer configuration queries over the standard MQ client."""

# Standard
from unittest.mock import MagicMock

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.transport.zmq_impl.client import ZmqMultiprocessClient
from lmcache.v1.multiprocess.transport.zmq_impl.server import get_zmq_handler_specs


@pytest.mark.parametrize("separate_object_groups", [False, True])
def test_config_exposes_active_setting_and_null_block_support(
    separate_object_groups: bool,
) -> None:
    context = MagicMock()
    context.separate_object_groups = separate_object_groups
    module = ManagementModule(context)
    config = module.get_server_config()
    assert config == {
        "separate_object_groups": separate_object_groups,
        "supports_null_block_id": True,
    }
    assert (
        msgspec.msgpack.decode(
            msgspec.msgpack.encode(config),
            type=get_response_class(RequestType.GET_SERVER_CONFIG),
        )
        == config
    )
    handlers = {
        spec.request_type: spec.handler for spec in get_zmq_handler_specs(module)
    }
    assert handlers[RequestType.GET_SERVER_CONFIG]() == config


def test_public_client_queries_server_config() -> None:
    queue = MagicMock()
    client = ZmqMultiprocessClient(queue)
    future = client.get_server_config()
    queue.submit_request.assert_called_once_with(
        RequestType.GET_SERVER_CONFIG, [], dict[str, bool]
    )
    assert future is queue.submit_request.return_value
