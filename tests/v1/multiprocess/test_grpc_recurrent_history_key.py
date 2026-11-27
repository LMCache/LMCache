# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.transport.grpc import proto_codec
from lmcache.v1.transport.grpc._proto_gen.common_pb2 import IpcCacheServerKey


@pytest.mark.parametrize("history_copy", [None, b"", b"history"])
def test_history_copy_round_trip(history_copy: bytes | None) -> None:
    key = IPCCacheServerKey(
        model_name="model",
        world_size=1,
        worker_id=None,
        token_bytes=b"",
        start=0,
        end=0,
        request_id="request",
        history_copy=history_copy,
    )
    encode = proto_codec.compile_response_encoder_for_type(
        IpcCacheServerKey, IPCCacheServerKey
    )
    decode = proto_codec.compile_response_decoder_for_type(
        IpcCacheServerKey, IPCCacheServerKey
    )

    message = encode(key)
    assert message.HasField("history_copy") == (history_copy is not None)
    restored = decode(IpcCacheServerKey.FromString(message.SerializeToString()))
    assert restored.history_copy == history_copy
