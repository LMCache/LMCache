# SPDX-License-Identifier: Apache-2.0
"""Cache identity, transient attributes and transport contracts."""

# Standard
from dataclasses import replace

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.cache_identity import ObjectKeyIdentity, decode_tags, get_request_tags
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.distributed.key_codec import encode_storage_key
from lmcache.v1.distributed.storage_controllers.store_controller import StoreListener
from lmcache.v1.mp_coordinator.utils.encoding import decode_key, encode_key
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.session import Session
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    common_pb2,
    p2p_service_pb2,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    compile_request_codec_for_types,
)


def test_identity_and_immutable_tags() -> None:
    tags = {"tenant": "a", "version": "1"}
    identity = ObjectKeyIdentity(
        chunk_hash=b"1234", kv_rank=1, model_name="model", tags=tags
    )
    key = ObjectKey(identity=identity)
    same = replace(key, identity=replace(identity, tags=dict(reversed(tags.items()))))
    assert key == same and hash(key) == hash(same)
    assert encode_storage_key(key) == encode_storage_key(same)
    tags["tenant"] = "b"
    assert identity.tags["tenant"] == "a"
    with pytest.raises(TypeError, match="immutable"):
        identity.tags["tenant"] = "b"  # type: ignore[index]
    assert key != replace(key, identity=replace(identity, tags=tags))
    assert key != replace(key, identity=replace(identity, model_name="another"))
    assert key != replace(key, identity=replace(identity, cache_salt="tenant-b"))
    assert key != ObjectKey(identity=replace(identity, chunk_hash=b"5678"))
    assert key != ObjectKey(identity=replace(identity, kv_rank=2))
    assert key != ObjectKey(identity=replace(identity, object_group_id=1))
    assert decode_tags(identity.tags_token()) == identity.tags
    assert (
        ObjectKeyIdentity(
            chunk_hash=b"1234", kv_rank=1, model_name="model"
        ).tags_token()
        == ""
    )


@pytest.mark.parametrize("token", ["x2-abc", "x1-!", "x1-e30"])
def test_identity_rejects_invalid_tokens(token: str) -> None:
    with pytest.raises(ValueError):
        decode_tags(token)


@pytest.mark.parametrize("tags", [{"": "v"}, {"a": 1}])
def test_invalid_tags(tags: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        ObjectKeyIdentity(chunk_hash=b"1234", kv_rank=1, model_name="model", tags=tags)


def test_identity_transport_and_object_expansion() -> None:
    ipc = IPCCacheServerKey.from_token_ids(
        "model",
        2,
        None,
        [1, 2],
        end=2,
        request_configs={
            "lmcache.tag.tenant": "a",
            "lmcache.tag.app": "v1",
            "lmcache.xattr.owner": "alice",
            "lmcache.xattr.priority": 1,
            "lmcache.skip_load": True,
        },
    )
    for codec in (msgspec.msgpack, msgspec.json):
        decoded = codec.decode(codec.encode(ipc), type=IPCCacheServerKey)
        assert decoded == ipc and hash(decoded) == hash(ipc)
        assert list(get_request_tags(decoded.request_configs)) == ["app", "tenant"]
        assert decoded.request_configs == ipc.request_configs
        groups = ipc_key_to_object_keys(decoded, [b"1234", b"5678"], [0, 1])
        assert len(groups) == 2 and all(len(group) == 4 for group in groups)
        for key in (key for group in groups for key in group):
            assert key.model_name == ipc.model_name
            assert key.identity.tags == get_request_tags(ipc.request_configs)
            assert key.xattrs == {"owner": "alice", "priority": 1}
            encoded = key.to_encoded_object_key()
            assert (
                codec.decode(codec.encode(encoded), type=type(encoded)).to_object_key()
                == key
            )
            assert decode_key(encode_key(key)) == key
    assert (
        replace(
            ipc, request_id="another", request_configs={"hint": 1}, num_kv_readers=3
        )
        == ipc
    )
    requests = [
        ipc,
        replace(ipc, token_ids=(3, 4)),
        replace(ipc, start=1),
        replace(ipc, end=1),
    ]
    assert len(set(requests)) == 4
    legacy = msgspec.to_builtins(ipc)
    del legacy["cache_salt"]
    assert msgspec.convert(legacy, type=IPCCacheServerKey).cache_salt == ""


def test_identity_protobuf_roundtrip() -> None:
    identity = ObjectKeyIdentity(
        chunk_hash=b"1234",
        kv_rank=0,
        model_name="model",
        tags={"tenant": "a", "app": "v1"},
    )
    ipc = IPCCacheServerKey.from_token_ids(
        "model", 1, 0, [1], request_configs={"lmcache.tag.tenant": "a"}
    )
    encode, decode = compile_request_codec_for_types(
        common_pb2.IpcCacheServerKey, (IPCCacheServerKey,)
    )
    message = encode((ipc,), {})
    assert message.model_name == "model"
    (recovered_ipc,) = decode(
        common_pb2.IpcCacheServerKey.FromString(message.SerializeToString())
    )
    assert recovered_ipc.request_configs == ipc.request_configs
    key = ObjectKey(identity=identity, xattrs={"owner": "alice"})
    encode, decode = compile_request_codec_for_types(
        p2p_service_pb2.P2pUnlockObjectsRequest, (list[ObjectKey],)
    )
    message = encode(([key],), {})
    (keys,) = decode(type(message).FromString(message.SerializeToString()))
    assert keys == [key]
    assert list(keys[0].identity.tags) == ["app", "tenant"]
    assert keys[0].xattrs == key.xattrs


def test_session_release_checks_identity() -> None:
    lookup = IPCCacheServerKey.from_token_ids(
        "model",
        1,
        None,
        [1],
        end=1,
        request_configs={"lmcache.tag.dtype": "bf16"},
    )
    session = Session("r", TokenHasher(chunk_size=1))
    session.begin_lookup(lookup, (-1,))
    session.record_prefetch_result(1, (0,))
    worker = replace(lookup, worker_id=0)
    snapshot = session.prepare_failed_retrieve_release(worker)
    assert snapshot is not None
    wrong = replace(worker, request_configs={"lmcache.tag.dtype": "fp8"})
    assert session.prepare_failed_retrieve_release(wrong) is None
    assert not session.claim_failed_retrieve_release(1, wrong, snapshot[-1])
    assert session.claim_failed_retrieve_release(1, worker, snapshot[-1])


def test_xattrs_are_mutable_and_do_not_change_identity() -> None:
    xattrs = {"owner": "alice", "priority": 1}
    key = ObjectKey(
        identity=ObjectKeyIdentity(chunk_hash=b"1234", kv_rank=0, model_name="model"),
        xattrs=xattrs,
    )
    other = replace(key, xattrs={"owner": "bob"})
    assert key == other and hash(key) == hash(other)
    assert encode_storage_key(key) == encode_storage_key(other)
    assert key.to_encoded_object_key().to_object_key().xattrs == {}
    assert key.xattrs.pop("owner") == "alice"
    assert key.xattrs == {"priority": 1}
    assert xattrs["owner"] == "alice"
    assert key == other and hash(key) == hash(other)


def test_store_queue_preserves_xattrs() -> None:
    key = ObjectKey(
        identity=ObjectKeyIdentity(chunk_hash=b"1234", kv_rank=0, model_name="model"),
        xattrs={"owner": "alice"},
    )
    listener = StoreListener()
    try:
        listener.on_l1_keys_write_finished([key])
        [queued] = listener.pop_pending_keys()
        assert queued == key and queued.xattrs == key.xattrs
        assert key.xattrs == {"owner": "alice"}
    finally:
        listener.close()
