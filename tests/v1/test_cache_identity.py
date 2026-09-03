# SPDX-License-Identifier: Apache-2.0
"""Tests for versioned cache representation identity."""

# Standard
from typing import Any

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_identity import (
    CACHE_IDENTITY_REVISION_TAG,
    BaseCacheIdentity,
    CacheIdentity,
    CacheIdentityError,
    CacheIdentityMismatchError,
    CacheRepresentationIdentity,
    cache_identity_from_request_configs,
    cache_identity_revision,
    materialize_cache_identity_revision,
    namespace_chunk_hash,
    namespace_chunk_hashes,
)
from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey


def _identity(
    *,
    weight_revision: str = "weights-42",
    kv_dtype: str = "bfloat16",
    drop_policy_revision: str | None = None,
) -> CacheIdentity:
    """Build a complete identity with concise defaults for tests."""
    return CacheIdentity(
        base=BaseCacheIdentity(
            model_revision="model-abc",
            tokenizer_revision="tokenizer-def",
            weight_revision=weight_revision,
            adapter_revision="lora-7",
        ),
        representation=CacheRepresentationIdentity(
            topology_fingerprint="hybrid:mamba=24,attention=4",
            backend_revision="flash-attention-3.1",
            kv_dtype=kv_dtype,
            quantization_revision="fp8-v2",
            compression_revision="cachegen-v1",
            drop_algorithm_id=(
                "dapo-attention-score" if drop_policy_revision is not None else None
            ),
            drop_policy_revision=drop_policy_revision,
        ),
    )


def test_revision_is_deterministic_across_request_config_order() -> None:
    """Field insertion order must not alter the identity namespace."""
    configs: dict[str, Any] = _identity().to_request_configs()
    reversed_configs = dict(reversed(tuple(configs.items())))

    assert cache_identity_revision(configs) == cache_identity_revision(reversed_configs)
    assert cache_identity_from_request_configs(configs) == _identity()


@pytest.mark.parametrize(
    ("override", "expected_field"),
    [
        ({"weight_revision": "weights-43"}, "base.weight_revision"),
        ({"kv_dtype": "float16"}, "representation.kv_dtype"),
        ({"drop_policy_revision": "policy-2"}, "representation.drop_policy_revision"),
    ],
)
def test_compatibility_fails_closed_on_any_revision_mismatch(
    override: dict[str, str], expected_field: str
) -> None:
    """Semantic, physical, and policy changes must all reject reuse."""
    candidate_options = {"drop_policy_revision": "policy-1", **override}
    candidate = _identity(**candidate_options)
    reference = _identity(drop_policy_revision="policy-1")

    assert not reference.is_compatible_with(candidate)
    assert expected_field in reference.mismatched_fields(candidate)
    with pytest.raises(CacheIdentityMismatchError, match=expected_field):
        reference.require_compatible(candidate)


def test_drop_algorithm_and_policy_must_be_revised_together() -> None:
    """A half-specified token-drop representation is ambiguous."""
    with pytest.raises(CacheIdentityError, match="must be set together"):
        CacheRepresentationIdentity(
            topology_fingerprint="dense",
            backend_revision="backend-1",
            kv_dtype="bfloat16",
            drop_algorithm_id="dapo-attention-score",
        )


def test_partial_and_unknown_request_identities_are_rejected() -> None:
    """Once identity keying is requested, missing and misspelled fields fail."""
    with pytest.raises(CacheIdentityError, match="missing"):
        cache_identity_revision({"lmcache.cache_identity.model_revision": "model-abc"})
    with pytest.raises(CacheIdentityError, match="unknown"):
        cache_identity_revision({"lmcache.cache_identity.model_revison": "model-abc"})
    with pytest.raises(CacheIdentityError, match="unknown"):
        cache_identity_revision({"lmcache.cache_identity": {"revision": "v1"}})


def test_internal_revision_roundtrip_and_mixed_form_rejected() -> None:
    """Materialized keys may carry a digest, but cannot mix both forms."""
    revision = _identity().revision
    assert cache_identity_revision({CACHE_IDENTITY_REVISION_TAG: revision}) == revision
    with pytest.raises(CacheIdentityError, match="cannot be mixed"):
        cache_identity_revision(
            {
                CACHE_IDENTITY_REVISION_TAG: revision,
                "lmcache.cache_identity.model_revision": "model-abc",
            }
        )
    configs: dict[str, Any] = _identity().to_request_configs()
    configs["lmcache.cache_identity.weight_revision"] = None
    with pytest.raises(CacheIdentityError, match="must be a string"):
        cache_identity_revision(configs)
    with pytest.raises(CacheIdentityError, match="unknown"):
        cache_identity_revision({"lmcache.cache_identity.revision": revision})
    with pytest.raises(CacheIdentityError, match="must have form"):
        cache_identity_revision({CACHE_IDENTITY_REVISION_TAG: "v1:not-a-digest"})
    with pytest.raises(CacheIdentityError, match="must have form"):
        cache_identity_revision({CACHE_IDENTITY_REVISION_TAG: None})


def test_explicit_null_optional_request_identity_is_rejected() -> None:
    """A configured optional field must be a string rather than JSON null."""
    configs: dict[str, Any] = _identity().to_request_configs()
    configs["lmcache.cache_identity.adapter_revision"] = None

    with pytest.raises(CacheIdentityError, match="must be a string"):
        cache_identity_revision(configs)


def test_namespace_preserves_legacy_hash_and_separates_revisions() -> None:
    """Unversioned keys remain stable while distinct revisions cannot collide."""
    chunk_hash = b"token-prefix-hash"
    revision_a = _identity(weight_revision="weights-a").revision
    revision_b = _identity(weight_revision="weights-b").revision

    assert namespace_chunk_hash(chunk_hash, "") is chunk_hash
    assert namespace_chunk_hash(chunk_hash, revision_a) == namespace_chunk_hash(
        chunk_hash, revision_a
    )
    assert namespace_chunk_hash(chunk_hash, revision_a) != namespace_chunk_hash(
        chunk_hash, revision_b
    )
    chunk_hashes = [chunk_hash, b"second-prefix-hash"]
    assert namespace_chunk_hashes(chunk_hashes, "") is chunk_hashes
    assert namespace_chunk_hashes(chunk_hashes, revision_a) == [
        namespace_chunk_hash(item, revision_a) for item in chunk_hashes
    ]


def test_materialized_revision_keeps_unrelated_request_configuration() -> None:
    """One compact tag replaces only the structured identity fields."""
    configs: dict[str, Any] = {
        **_identity().to_request_configs(),
        "lmcache.ttl": 60,
    }
    materialized = materialize_cache_identity_revision(configs)

    assert materialized is not None
    assert materialized["lmcache.ttl"] == 60
    assert materialized["lmcache.tag.cache_identity_revision"] == _identity().revision
    assert not any(key.startswith("lmcache.cache_identity.") for key in materialized)


def test_in_process_key_identity_survives_string_and_dict_roundtrips() -> None:
    """Legacy CacheEngineKey includes the canonical revision as an internal tag."""
    configs = _identity().to_request_configs()
    key = CacheEngineKey("model", 1, 0, 0x1234, torch.bfloat16, configs)
    changed = CacheEngineKey(
        "model",
        1,
        0,
        0x1234,
        torch.bfloat16,
        _identity(weight_revision="weights-new").to_request_configs(),
    )

    assert key != changed
    assert "cache_identity_revision%v1:" in key.to_string()
    assert CacheEngineKey.from_string(key.to_string()) == key
    assert CacheEngineKey.from_dict(key.to_dict()) == key


def test_mp_key_identity_survives_ipc_and_namespaces_every_object_group() -> None:
    """MP keys bind the revision without changing ObjectKey's wire schema."""
    key = IPCCacheServerKey.from_token_ids(
        model_name="model",
        world_size=2,
        worker_id=None,
        token_ids=[1, 2, 3, 4],
        request_configs=_identity().to_request_configs(),
    )
    legacy_key = IPCCacheServerKey.from_token_ids(
        model_name="model",
        world_size=2,
        worker_id=None,
        token_ids=[1, 2, 3, 4],
    )

    identity_groups = ipc_key_to_object_keys(key, [b"chunk"], [0, 3])
    legacy_groups = ipc_key_to_object_keys(legacy_key, [b"chunk"], [0, 3])
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(key), type=IPCCacheServerKey
    )
    stale_wire_value = msgspec.msgpack.decode(msgspec.msgpack.encode(key))
    stale_wire_value["cache_identity_revision"] = "v1:" + "0" * 64
    recomputed = msgspec.msgpack.decode(
        msgspec.msgpack.encode(stale_wire_value), type=IPCCacheServerKey
    )

    assert key != legacy_key
    assert decoded == key
    assert decoded.cache_identity_revision == key.cache_identity_revision
    assert recomputed.cache_identity_revision == key.cache_identity_revision
    assert (
        key.no_worker_id_version().cache_identity_revision
        == key.cache_identity_revision
    )
    assert all(obj.chunk_hash != b"chunk" for group in identity_groups for obj in group)
    assert all(obj.chunk_hash == b"chunk" for group in legacy_groups for obj in group)
    assert identity_groups[0][0].chunk_hash == identity_groups[1][0].chunk_hash
