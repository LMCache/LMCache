# SPDX-License-Identifier: Apache-2.0
"""Registration contracts through the real module and request transports."""

# Standard
from collections.abc import Iterator
from typing import Any, Literal, cast
from unittest.mock import MagicMock

# Third Party
import msgspec
import pytest
import zmq

# First Party
from lmcache.utils import EngineType
from lmcache.v1.client.transport.errors import MPRequestError
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.client.transport.zmq_impl.mq_client import MessageQueueClient
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import KVEventProfile
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from lmcache.v1.multiprocess.transport.zmq_impl.mq import msgspec_encode
from lmcache.v1.transport.errors import InvalidRequestError
from lmcache.v1.transport.futures import MessagingFuture, MessagingStream
from lmcache.v1.transport.grpc._proto_gen import (
    lmcache_driven_service_pb2,
)
from lmcache.v1.transport.grpc.method_registry import (
    get_method_codec_registry,
)
from lmcache.v1.transport.zmq.wire import (
    ZMQ_ERROR_OPERATION,
    decode_error_response,
    decode_operation,
    encode_operation,
)
from tests.v1.multiprocess.transport_test_utils import requires_pygrpc_native

PROFILE = KVEventProfile("engine:sha256:bytes:v1", 16)
GROUP = EngineGroupInfo(0, (0,), tokens_per_block=16)


class RejectingModule:
    """Exercise typed rejection on a serialized operation other than registration."""

    @request_handler(HandlerType.SERIALIZED)
    def clear(self, force: bool = False) -> None:
        """Refuse the request without changing any state."""
        raise InvalidRequestError("clear rejected without changing state")


def _payload(
    instance: int = 1, profile: KVEventProfile | None = PROFILE
) -> tuple[Any, ...]:
    return (instance, [], "model", 2, EngineType.VLLM, {}, [GROUP], profile)


@pytest.fixture
def registration(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[transfer.LMCacheDrivenTransferModule, list[MagicMock]]]:
    """Stub device boundaries; retain actual registration/layout state."""
    context = MagicMock()
    context.chunk_size = 256
    context.layout_desc_registry = LayoutDescRegistry()
    created: list[MagicMock] = []

    def create_context(*args: Any, **kwargs: Any) -> MagicMock:
        cache = MagicMock(num_layers=1)
        cache.kv_layer_groups_manager.num_object_groups = 1
        cache.kv_layer_groups_manager.get_attn_desc.return_value = AttnWindowDesc([-1])
        created.append(cache)
        return cache

    monkeypatch.setattr(transfer, "DeviceHostFuncDispatcher", MagicMock())
    monkeypatch.setattr(transfer, "create_cache_context", create_context)
    monkeypatch.setattr(transfer, "get_event_ipc_backend", MagicMock())
    monkeypatch.setattr(
        transfer, "get_layout_desc", lambda *a, **kw: MemoryLayoutDesc([], [])
    )
    monkeypatch.setattr(transfer, "torch_dev", MagicMock())
    module = transfer.LMCacheDrivenTransferModule(context)
    try:
        yield module, created
    finally:
        module.close()


@pytest.mark.parametrize("profile_id,block_size", [("", 16), ("v1", 0), ("v1", -1)])
def test_profile_rejects_empty_identity_and_nonpositive_blocks(
    profile_id: str, block_size: int
) -> None:
    with pytest.raises(ValueError, match="ID and positive block size"):
        KVEventProfile(profile_id, block_size)


@pytest.mark.parametrize("instance", [1, 2])
@pytest.mark.parametrize("profile", [None, KVEventProfile("different", 16)])
def test_profile_conflict_does_not_mutate_registration(
    registration: tuple[transfer.LMCacheDrivenTransferModule, list[MagicMock]],
    instance: int,
    profile: KVEventProfile | None,
) -> None:
    module, created = registration
    module.register_kv_cache(*_payload())
    before = module.context_entries_snapshot()[1].last_seen
    layout = module.context.layout_desc_registry.find("model", 2)
    with pytest.raises(ValueError, match="conflicts with registration"):
        module.register_kv_cache(*_payload(instance, profile))
    assert module.context_entries_snapshot()[1].last_seen == before
    assert len(created) == module.tracked_instance_count() == 1
    assert module.context.layout_desc_registry.find("model", 2) is layout
    created[0].close.assert_not_called()


@pytest.mark.parametrize(
    "groups",
    [
        [],
        [msgspec.structs.replace(GROUP, engine_group_id=1)],
        [msgspec.structs.replace(GROUP, tokens_per_block=32)],
        [msgspec.structs.replace(GROUP, sw_size_tokens=128)],
        [msgspec.structs.replace(GROUP, recurrent_state=True)],
        [msgspec.structs.replace(GROUP, extra_object_group_tag=1)],
    ],
)
def test_unsupported_groups_fail_before_opening_device_context(
    registration: tuple[transfer.LMCacheDrivenTransferModule, list[MagicMock]],
    groups: list[EngineGroupInfo],
) -> None:
    module, created = registration
    payload = list(_payload())
    payload[6] = groups
    with pytest.raises(ValueError, match="full-attention engine group"):
        module.register_kv_cache(*payload)
    assert created == [] and module.tracked_instance_count() == 0


def test_profile_lifetime_and_identical_reregistration(
    registration: tuple[transfer.LMCacheDrivenTransferModule, list[MagicMock]],
) -> None:
    module, created = registration
    module.register_kv_cache(*_payload())
    module.register_kv_cache(*_payload())
    module.register_kv_cache(*_payload(2))
    assert len(created) == 2
    assert all(
        e.event_profile == PROFILE for e in module.context_entries_snapshot().values()
    )
    module.unregister_kv_cache(1)
    with pytest.raises(ValueError, match="conflicts"):
        module.register_kv_cache(*_payload(3, None))
    module.unregister_kv_cache(2)
    module.register_kv_cache(*_payload(3, None))
    assert module.context_entries_snapshot()[3].event_profile is None


@pytest.mark.parametrize("world,block", [(0, 16), (2, 3)])
def test_profile_rejects_incompatible_chunk_geometry(
    registration: tuple[transfer.LMCacheDrivenTransferModule, list[MagicMock]],
    world: int,
    block: int,
) -> None:
    module, created = registration
    payload = list(_payload(profile=KVEventProfile("geometry", block)))
    payload[3] = world
    with pytest.raises(ValueError, match="positive world size.*divides"):
        module.register_kv_cache(*payload)
    assert created == [] and module.tracked_instance_count() == 0


def test_profile_cannot_upgrade_an_active_unprofiled_registration(
    registration: tuple[transfer.LMCacheDrivenTransferModule, list[MagicMock]],
) -> None:
    module, created = registration
    module.register_kv_cache(*_payload(profile=None))
    with pytest.raises(ValueError, match="conflicts"):
        module.register_kv_cache(*_payload(2))
    assert len(created) == 1


def test_profile_cannot_change_instance_namespace(
    registration: tuple[transfer.LMCacheDrivenTransferModule, list[MagicMock]],
) -> None:
    module, created = registration
    module.register_kv_cache(*_payload())
    payload = list(_payload())
    payload[2] = "other-model"
    with pytest.raises(ValueError, match="instance context mismatch"):
        module.register_kv_cache(*payload)
    assert len(created) == 1


@pytest.mark.parametrize(
    ("transport", "mode"),
    [
        ("zmq", "sync"),
        ("grpc", "sync"),
        ("grpc", "aio"),
        pytest.param("grpc", "native", marks=requires_pygrpc_native),
    ],
)
def test_registration_errors_reach_client_and_leave_server_usable(
    registration: tuple[transfer.LMCacheDrivenTransferModule, list[MagicMock]],
    transport: Literal["zmq", "grpc"],
    mode: Literal["sync", "aio", "native"],
) -> None:
    """Invalid registrations fail with a reason, not a timeout, on real sockets."""
    module, created = registration
    config = MPServerConfig(
        transport=transport,
        host="127.0.0.1",
        port=0,
        grpc_server_mode=mode,
        max_cpu_workers=1,
    )
    server: Any = create_request_server([module, cast(Any, RejectingModule())], config)
    url = (
        server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        if transport == "zmq"
        else f"grpc://127.0.0.1:{server.bound_port}"
    )
    server.start()
    client = RequestClientFactory.create(url, context=zmq.Context.instance())
    try:
        assert client.register_kv_cache(*_payload()).result(timeout=5) == PROFILE
        with pytest.raises(MPRequestError) as error:
            client.register_kv_cache(*_payload(2, KVEventProfile("other", 16))).result(
                timeout=5
            )
        expected_code = (
            "INVALID_ARGUMENT" if transport == "grpc" else "invalid_argument"
        )
        assert error.value.code == expected_code
        assert (
            "KV event profile conflicts with registration 1 "
            "for model 'model', world size 2"
        ) in error.value.details
        with pytest.raises(MPRequestError) as rejected:
            client.clear(False).result(timeout=5)
        assert rejected.value.code == expected_code
        assert "clear rejected without changing state" in rejected.value.details
        if transport == "zmq":
            with zmq.Context.instance().socket(zmq.DEALER) as socket:
                socket.setsockopt(zmq.RCVTIMEO, 5000)
                socket.connect(url)
                # Old seven-field registration must receive an explicit arity error.
                socket.send_multipart(
                    [
                        msgspec.msgpack.encode(7),
                        encode_operation("register_kv_cache"),
                        *[msgspec_encode(v, type(v)) for v in _payload()[:-1]],
                    ]
                )
                _, operation, payload = socket.recv_multipart()
                assert decode_operation(operation) == ZMQ_ERROR_OPERATION
                decoded = decode_error_response(payload)
                assert decoded.code == "invalid_argument" and "count" in decoded.message
        assert client.register_kv_cache(*_payload(2)).result(timeout=5) == PROFILE
        assert len(created) == module.tracked_instance_count() == 2
        client.unregister_kv_cache(1).result(timeout=5)
        client.unregister_kv_cache(2).result(timeout=5)
        assert client.register_kv_cache(*_payload(3, None)).result(timeout=5) is None
        if transport == "zmq":
            with zmq.Context.instance().socket(zmq.DEALER) as socket:
                socket.setsockopt(zmq.RCVTIMEO, 5000)
                socket.connect(url)
                prefix = [
                    msgspec.msgpack.encode(8),
                    encode_operation("register_kv_cache"),
                ]
                socket.send_multipart(
                    prefix + [msgspec_encode(v, type(v)) for v in _payload(4, None)]
                )
                assert socket.recv_multipart() == prefix
    finally:
        client.close()
        server.close()


def test_none_profile_preserves_protobuf_request_and_success_bytes() -> None:
    codec = get_method_codec_registry().by_full_name[
        "lmcache.mp.LMCacheDrivenService.RegisterKvCache"
    ]
    old = lmcache_driven_service_pb2.RegisterKvCacheRequest(
        instance_id=1,
        model_name="model",
        world_size=2,
        engine_type="vllm",
    )
    payload: tuple[Any, ...] = (1, [], "model", 2, EngineType.VLLM, {}, [], None)
    request = codec.request_encoder(payload, {})
    assert request.SerializeToString() == old.SerializeToString()
    assert codec.request_decoder(request) == payload
    assert codec.response_encoder(None).SerializeToString() == b""
    assert (
        codec.response_decoder(
            lmcache_driven_service_pb2.RegisterKvCacheResponse.FromString(b"")
        )
        is None
    )
    assert codec.response_decoder(codec.response_encoder(PROFILE)) == PROFILE
    qstore = get_method_codec_registry().by_full_name[
        "lmcache.mp.QStoreService.RegisterQCache"
    ]
    assert (
        qstore.request_encoder(payload[:-1], {}).SerializeToString()
        == old.SerializeToString()
    )
    assert (
        qstore.request_decoder(qstore.request_encoder(payload[:-1], {})) == payload[:-1]
    )
    assert qstore.response_encoder(None).SerializeToString() == b""
    assert qstore.response_decoder(qstore.response_encoder(None)) is None
    native = codec.request_encoder(_payload(), {})
    assert codec.request_decoder(native)[-1] == PROFILE


def test_zmq_missing_profile_reports_arity_not_union_attribute_error() -> None:
    client = MessageQueueClient(
        "inproc://missing-profile-test", context=zmq.Context.instance()
    )
    try:
        future: MessagingFuture[Any] | MessagingStream[Any] = client.submit_request(
            "register_kv_cache", list(_payload()[:-1])
        )
        assert isinstance(future, MessagingFuture)
        with pytest.raises(ValueError, match="Payload count mismatch"):
            future.result(timeout=5)
    finally:
        client.close()
