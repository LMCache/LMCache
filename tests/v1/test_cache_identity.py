# SPDX-License-Identifier: Apache-2.0
"""Cache identity tag contracts."""

# Standard
from dataclasses import replace

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    EncodedObjectKey,
    ObjectKey,
    ObjectKeyIdentity,
    get_request_tags,
    ipc_key_to_object_keys,
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


def test_request_tags_reach_expanded_object_keys() -> None:
    ipc_key = IPCCacheServerKey.from_token_ids(
        model_name="model",
        world_size=2,
        worker_id=None,
        token_ids=[1, 2],
        request_configs={
            "lmcache.tag.tenant": "a",
            "lmcache.tag.app": "v1",
            "lmcache.skip_load": True,
        },
    )

    groups = ipc_key_to_object_keys(
        ipc_key,
        [b"1234"],
        [0, 1],
        DEFAULT_SHARD_REPLICATION_DESC,
        group_layout_descs={},
    )

    assert get_request_tags(ipc_key.request_configs) == {
        "app": "v1",
        "tenant": "a",
    }
    assert all(
        key.identity.tags == {"app": "v1", "tenant": "a"}
        for group in groups
        for key in group
    )


@pytest.mark.parametrize(
    "request_configs",
    [
        {"lmcache.tag.priority": 1},
        {"lmcache.tag.": "value"},
    ],
)
def test_request_tags_reject_invalid_values(
    request_configs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="string values"):
        get_request_tags(request_configs)


class TestTagToken:
    """The tag token is part of an L2 file name, so its size is a constraint."""

    def test_round_trip(self):
        # First Party
        from lmcache.v1.distributed.key_codec import decode_tags, encode_tags

        tags = {"kv_dtype": "bf16,f8e4m3"}
        assert dict(decode_tags(encode_tags(tags))) == tags

    def test_known_names_ride_as_indices(self):
        """A known name must not cost its spelling; unknown ones still work."""
        # First Party
        from lmcache.v1.distributed.key_codec import decode_tags, encode_tags

        known = encode_tags({"kv_dtype": "bf16"})
        unknown = encode_tags({"revision": "abc"})
        assert known.startswith("x2-")
        assert len(known) < len(unknown)
        assert dict(decode_tags(unknown)) == {"revision": "abc"}

    def test_legacy_tokens_still_decode(self):
        """Objects written before the compact encoding stay readable."""
        # Standard
        import base64

        # Third Party
        import msgspec

        # First Party
        from lmcache.v1.distributed.key_codec import decode_tags

        payload = msgspec.json.encode({"kv_dtype": "float16"})
        legacy = "x1-" + base64.urlsafe_b64encode(payload).decode().rstrip("=")
        assert dict(decode_tags(legacy)) == {"kv_dtype": "float16"}

    def test_a_tampered_token_is_rejected(self):
        # First Party
        from lmcache.v1.distributed.key_codec import decode_tags

        with pytest.raises(ValueError):
            decode_tags("x2-not-a-real-payload")

    def test_the_mixed_group_token_fits_a_file_name(self):
        """Qwen3-1.7B's bf16 name was 255 bytes with the JSON token."""
        # First Party
        from lmcache.v1.distributed.key_codec import encode_tags

        assert len(encode_tags({"kv_dtype": "bf16"})) <= 20
        assert len(encode_tags({"kv_dtype": "bf16,f8e4m3"})) <= 30
