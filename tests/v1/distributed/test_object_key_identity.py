# SPDX-License-Identifier: Apache-2.0
"""Object keys expose immutable matching fields through their identity."""

# Standard
from dataclasses import replace
from types import ModuleType

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.cache_identity import ObjectKeyIdentity
from lmcache.v1.distributed.api import ObjectKey


def test_typed_identity_equality_and_replacement() -> None:
    identity = ObjectKeyIdentity(
        chunk_hash=b"chunk",
        model_name="model",
        kv_rank=7,
        object_group_id=3,
        cache_salt="tenant",
    )
    key = ObjectKey(identity=identity)
    assert key == ObjectKey(identity=identity)
    assert len({key, ObjectKey(identity=identity)}) == 1
    updated = replace(key, identity=replace(identity, object_group_id=0, cache_salt=""))
    assert updated.identity == ObjectKeyIdentity(b"chunk", 7, "model")
    assert key.identity == identity


@pytest.mark.parametrize("codec", [msgspec.json, msgspec.msgpack])
def test_structural_roundtrip_preserves_identity(codec: ModuleType) -> None:
    key = ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=b"chunk",
            model_name="model",
            kv_rank=7,
            object_group_id=3,
            cache_salt="tenant",
        )
    )
    recovered = codec.decode(codec.encode(key), type=ObjectKey)
    assert recovered == key
    assert recovered.identity == key.identity
    assert hash(recovered) == hash(key)


def test_existing_field_validation_is_preserved() -> None:
    with pytest.raises(ValueError, match="object_group_id"):
        ObjectKey(
            identity=ObjectKeyIdentity(
                chunk_hash=b"chunk", model_name="model", kv_rank=0, object_group_id=-1
            )
        )
    with pytest.raises(ValueError, match="cache_salt"):
        ObjectKey(
            identity=ObjectKeyIdentity(
                chunk_hash=b"chunk",
                model_name="model",
                kv_rank=0,
                cache_salt="bad/salt",
            )
        )


@pytest.mark.parametrize(
    "name", ["chunk_hash", "model_name", "kv_rank", "object_group_id", "cache_salt"]
)
def test_matching_fields_are_only_available_on_identity(name: str) -> None:
    identity = ObjectKeyIdentity(b"chunk", 7, "model", 3, "tenant")
    key = ObjectKey(identity=identity)
    assert getattr(key.identity, name) == getattr(identity, name)
    with pytest.raises(AttributeError):
        getattr(key, name)
