# SPDX-License-Identifier: Apache-2.0
"""Typed object identities preserve the existing construction contracts."""

# Standard
from collections.abc import Callable
from dataclasses import replace
from types import ModuleType
import pickle  # noqa: TID251
import subprocess
import sys

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, ObjectKeyIdentity
from tests.import_utils import lmcache_subprocess_env


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


@pytest.mark.parametrize(
    "derive",
    [
        lambda key: replace(key, object_group_id=0),
        lambda key: replace(key, identity=replace(key.identity, object_group_id=0)),
        lambda key: ObjectKey(identity=key.identity, object_group_id=0),
    ],
    ids=["replace-key", "replace-identity", "identity-override"],
)
def test_keys_derived_after_hashing_hash_by_their_own_identity(
    derive: Callable[[ObjectKey], ObjectKey],
) -> None:
    key = ObjectKey(b"chunk", "model", 7, 3, "tenant", tags={"tenant": "a"})
    hash(key)
    derived = derive(key)
    expected = ObjectKey(b"chunk", "model", 7, 0, "tenant", tags={"tenant": "a"})
    assert derived == expected
    assert hash(derived) == hash(expected)
    assert derived not in {key: None}


def test_extra_attributes_do_not_affect_hash_or_equality() -> None:
    plain = ObjectKey(b"chunk", "model", 7)
    enriched = ObjectKey(b"chunk", "model", 7, extra_attributes={"request": "r1"})
    hash(plain)
    assert enriched == plain
    assert hash(enriched) == hash(plain)
    assert replace(plain, extra_attributes={"request": "r2"}) in {enriched}


@pytest.mark.parametrize(
    "encode",
    [msgspec.msgpack.encode, msgspec.json.encode],
    ids=["msgpack", "json"],
)
def test_encoding_is_independent_of_prior_hashing(
    encode: Callable[[ObjectKey], bytes],
) -> None:
    hashed = ObjectKey(b"chunk", "model", 7, extra_attributes={"request": "r1"})
    hash(hashed)
    fresh = ObjectKey(b"chunk", "model", 7, extra_attributes={"request": "r1"})
    assert encode(hashed) == encode(fresh)


@pytest.mark.parametrize("codec", [msgspec.json, msgspec.msgpack])
def test_decoded_key_lists_find_the_original_keys(codec: ModuleType) -> None:
    keys = [ObjectKey(bytes([i]), "model", 7, tags={"tenant": "a"}) for i in range(4)]
    slots = {key: index for index, key in enumerate(keys)}
    decoded = codec.decode(codec.encode(keys), type=list[ObjectKey])
    assert [slots[key] for key in decoded] == list(range(4))


_PICKLE_HASHED_KEY = """
import pickle, sys
from lmcache.v1.distributed.api import ObjectKey
key = ObjectKey(b"chunk", "model", 7, cache_salt="tenant")
hash(key)
sys.stdout.buffer.write(pickle.dumps(key))
"""


def test_key_pickled_after_hashing_rehashes_in_another_process() -> None:
    # A fixed seed differs from this process's random one, so str and bytes
    # hashes differ between the two processes.
    env = {**lmcache_subprocess_env(), "PYTHONHASHSEED": "1"}
    payload = subprocess.run(
        [sys.executable, "-c", _PICKLE_HASHED_KEY],
        env=env,
        capture_output=True,
        check=True,
    ).stdout
    recovered = pickle.loads(payload)
    fresh = ObjectKey(b"chunk", "model", 7, cache_salt="tenant")
    assert recovered == fresh
    assert recovered in {fresh: None}
