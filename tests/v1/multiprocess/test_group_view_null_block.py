# SPDX-License-Identifier: Apache-2.0
"""Per-group null block metadata remains compatible with older wire payloads."""

# Standard
from types import ModuleType

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo


@pytest.mark.parametrize("codec", [msgspec.json, msgspec.msgpack])
def test_old_group_metadata_defaults_to_null_zero(codec: ModuleType) -> None:
    old_payload = codec.encode({"engine_group_id": 0, "layer_indices": [0]})
    decoded = codec.decode(old_payload, type=EngineGroupInfo)
    assert decoded.null_block_id == 0


@pytest.mark.parametrize("codec", [msgspec.json, msgspec.msgpack])
@pytest.mark.parametrize("null_block_id", [None, -1, 0, 7])
def test_group_null_policy_round_trip(
    codec: ModuleType, null_block_id: int | None
) -> None:
    group = EngineGroupInfo(
        1,
        (2, 3),
        tokens_per_block=256,
        recurrent_state=True,
        null_block_id=null_block_id,
    )
    assert codec.decode(codec.encode(group), type=EngineGroupInfo) == group


def test_grpc_null_policy_wire_presence() -> None:
    common_pb2 = pytest.importorskip(
        "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen.common_pb2"
    )
    # First Party
    from lmcache.v1.multiprocess.transport.grpc_impl.codecs.common import (
        get_message_codecs,
    )

    # An older sender has no oneof field and retains the legacy null-zero policy.
    old = common_pb2.EngineGroupInfo(engine_group_id=0)
    assert old.WhichOneof("null_block_policy") is None
    assert old.null_block_id == 0

    for kwargs in (
        {"null_block_id": 0},
        {"null_block_id": -1},
        {"no_null_block": True},
    ):
        original = common_pb2.EngineGroupInfo(engine_group_id=0, **kwargs)
        restored = common_pb2.EngineGroupInfo.FromString(original.SerializeToString())
        assert restored == original
        assert restored.WhichOneof("null_block_policy") == next(iter(kwargs))

    codec = get_message_codecs()[1]
    assert codec.reader(old).null_block_id == 0
    for null_block_id in (None, -1, 0, 7):
        value = EngineGroupInfo(
            1,
            (2, 3),
            tokens_per_block=256,
            sw_size_tokens=128,
            extra_object_group_tag=4,
            recurrent_state=True,
            null_block_id=null_block_id,
        )
        encoded = common_pb2.EngineGroupInfo()
        codec.writer(encoded, value)
        assert codec.reader(encoded) == value
