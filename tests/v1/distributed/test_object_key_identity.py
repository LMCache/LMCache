# SPDX-License-Identifier: Apache-2.0
"""Typed object identities preserve the existing construction contracts."""

# Standard
from dataclasses import replace
from types import ModuleType
import copy

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


def test_derive_matches_the_constructor() -> None:
    base = ObjectKeyIdentity(
        chunk_hash=b"chunk",
        model_name="model",
        kv_rank=7,
        object_group_id=3,
        cache_salt="tenant",
        tags={"b": "2", "a": "1"},
    )
    derived = base.derive(chunk_hash=b"other", kv_rank=1)
    constructed = ObjectKeyIdentity(
        chunk_hash=b"other",
        model_name="model",
        kv_rank=1,
        object_group_id=3,
        cache_salt="tenant",
        tags={"a": "1", "b": "2"},
    )
    assert derived == constructed
    assert hash(derived) == hash(constructed)
    assert derived.tags == base.tags
    assert base.derive() == base
    assert hash(base.derive()) == hash(base)


def test_equal_keys_hash_equal_however_they_were_built() -> None:
    flat = ObjectKey(b"chunk", "model", 7, 3, "tenant", tags={"a": "1"})
    identity = ObjectKeyIdentity(
        chunk_hash=b"chunk",
        model_name="model",
        kv_rank=7,
        object_group_id=3,
        cache_salt="tenant",
        tags={"a": "1"},
    )
    typed = ObjectKey(identity=identity)
    derived = ObjectKey(identity=identity.derive())
    assert flat == typed == derived
    assert hash(flat) == hash(typed) == hash(derived)
    assert len({flat, typed, derived}) == 1


def test_serialized_state_keeps_equality_and_drops_the_hash_cache() -> None:
    key = ObjectKey(b"chunk", "model", 7, 3, "tenant")
    hash_before = hash(key)
    hash(key.identity)
    # The cached hash is process-local (string hashes are seeded per
    # process), so the state handed to pickle and copy must not carry it.
    assert "_hash" not in key.__getstate__()
    assert "_hash" not in key.identity.__getstate__()
    recovered = copy.deepcopy(key)
    # Checked before anything hashes ``recovered`` and re-caches.
    assert "_hash" not in vars(recovered)
    assert "_hash" not in vars(recovered.identity)
    assert recovered == key
    assert hash(recovered) == hash_before
