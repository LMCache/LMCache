# SPDX-License-Identifier: Apache-2.0
"""Typed object identities preserve the existing construction contracts."""

# Standard
from dataclasses import replace
from types import ModuleType

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, ObjectKeyIdentity


def test_existing_constructors_share_one_identity() -> None:
    identity = ObjectKeyIdentity(
        chunk_hash=b"chunk",
        model_name="model",
        kv_rank=7,
        object_group_id=3,
        cache_salt="tenant",
    )
    positional = ObjectKey(b"chunk", "model", 7, 3, "tenant")
    keyword = ObjectKey(
        chunk_hash=b"chunk",
        model_name="model",
        kv_rank=7,
        object_group_id=3,
        cache_salt="tenant",
    )
    typed = ObjectKey(identity=identity)
    assert positional == keyword == typed
    assert len({positional, keyword, typed}) == 1
    assert positional.identity == identity


def test_existing_flat_replacements_preserve_other_fields() -> None:
    key = ObjectKey(b"chunk", "model", 7, 3, "tenant")
    replace_key = replace
    updated = replace_key(key, object_group_id=0, cache_salt="")
    assert updated == ObjectKey(b"chunk", "model", 7)
    assert key.object_group_id == 3 and key.cache_salt == "tenant"
    assert replace_key(key, chunk_hash=b"other") == ObjectKey(
        b"other", "model", 7, 3, "tenant"
    )
    assert replace_key(key, model_name="other", kv_rank=0) == ObjectKey(
        b"chunk", "other", 0, 3, "tenant"
    )


@pytest.mark.parametrize("codec", [msgspec.json, msgspec.msgpack])
def test_structural_roundtrip_preserves_identity(codec: ModuleType) -> None:
    key = ObjectKey(b"chunk", "model", 7, 3, "tenant")
    recovered = codec.decode(codec.encode(key), type=ObjectKey)
    assert recovered == key
    assert recovered.identity == key.identity
    assert hash(recovered) == hash(key)


def test_existing_field_validation_is_preserved() -> None:
    with pytest.raises(ValueError, match="object_group_id"):
        ObjectKey(b"chunk", "model", 0, object_group_id=-1)
    with pytest.raises(ValueError, match="cache_salt"):
        ObjectKey(b"chunk", "model", 0, cache_salt="bad/salt")
