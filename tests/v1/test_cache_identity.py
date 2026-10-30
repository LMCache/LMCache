# SPDX-License-Identifier: Apache-2.0
"""Cache identity tag contracts."""

# Standard
from dataclasses import replace

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.distributed.api import (
    EncodedObjectKey,
    ObjectKey,
    ObjectKeyIdentity,
)
from lmcache.v1.distributed.key_codec import decode_tags, encode_tags


def test_identity_tags_are_canonical_and_immutable() -> None:
    tags = {"tenant": "a", "version": "1"}
    identity = ObjectKeyIdentity(
        chunk_hash=b"1234", kv_rank=1, model_name="model", tags=tags
    )
    key = ObjectKey(identity=identity)
    reordered = replace(
        key, identity=replace(identity, tags=dict(reversed(tags.items())))
    )

    assert key == reordered
    assert hash(key) == hash(reordered)
    assert decode_tags(encode_tags(identity.tags)) == identity.tags

    tags["tenant"] = "b"
    assert identity.tags["tenant"] == "a"
    with pytest.raises(TypeError, match="immutable"):
        identity.tags["tenant"] = "b"  # type: ignore[index]
    assert key != replace(key, identity=replace(identity, tags=tags))


@pytest.mark.parametrize("token", ["x2-abc", "x1-!", "x1-e30"])
def test_decode_tags_rejects_invalid_tokens(token: str) -> None:
    with pytest.raises(ValueError):
        decode_tags(token)


@pytest.mark.parametrize("tags", [{"": "v"}, {"a": 1}])
def test_identity_rejects_invalid_tags(tags: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        ObjectKeyIdentity(chunk_hash=b"1234", kv_rank=1, model_name="model", tags=tags)


def test_encoded_object_key_round_trip_preserves_tags() -> None:
    key = ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=b"1234",
            kv_rank=1,
            model_name="model",
            tags={"tenant": "a", "app": "v1"},
        )
    )
    encoded = key.to_encoded_object_key()
    reordered_encoded = replace(
        encoded, tags=dict(reversed(list(encoded.tags.items())))
    )

    assert hash(encoded) == hash(reordered_encoded)

    for codec in (msgspec.msgpack, msgspec.json):
        recovered = codec.decode(codec.encode(encoded), type=EncodedObjectKey)
        assert recovered.to_object_key() == key
        assert list(recovered.tags) == ["app", "tenant"]
