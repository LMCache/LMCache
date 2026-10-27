# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the IPC cache key contract."""

# Standard
from collections.abc import Sequence

# Third Party
import pytest

# First Party
from lmcache_client_sdk.key import IPCCacheServerKey


def _key(
    token_ids: Sequence[int] = (1, 2, 3),
    request_id: str = "req-1",
    cache_salt: str = "",
) -> IPCCacheServerKey:
    """Build a key through the public factory."""
    return IPCCacheServerKey.from_token_ids(
        model_name="Qwen/Qwen3-8B",
        world_size=2,
        worker_id=0,
        token_ids=token_ids,
        start=0,
        end=len(token_ids),
        request_id=request_id,
        cache_salt=cache_salt,
        num_kv_readers=2,
    )


def test_from_token_ids_packs_and_exposes_fields() -> None:
    """Check that the factory packs tokens and exposes the key fields."""
    key = _key()
    assert key.model_name == "Qwen/Qwen3-8B"
    assert key.world_size == 2
    assert key.worker_id == 0
    assert key.num_tokens == 3
    assert key.num_kv_readers == 2
    assert key.request_id == "req-1"


def test_request_id_not_part_of_equality() -> None:
    """Check that keys with different request ids compare equal."""
    assert _key(request_id="req-1") == _key(request_id="req-2")


def test_cache_salt_is_part_of_equality() -> None:
    """Check that different cache salts distinguish keys."""
    assert _key(cache_salt="user-a") != _key(cache_salt="user-b")


def test_no_worker_id_version_for_lookup() -> None:
    """Check that the lookup copy drops the worker id."""
    lookup_key = _key().no_worker_id_version()
    assert lookup_key.worker_id is None
    assert lookup_key.num_tokens == 3
    assert lookup_key.model_name == _key().model_name


def test_salt_forbidden_characters_rejected() -> None:
    """Check that forbidden salt characters raise a ValueError."""
    with pytest.raises(ValueError):
        _key(cache_salt="a@b")


def test_overlong_salt_rejected() -> None:
    """Check that an overlong salt raises a ValueError."""
    with pytest.raises(ValueError):
        _key(cache_salt="x" * 129)


def test_ragged_token_bytes_rejected() -> None:
    """Check that a non-whole token buffer raises a ValueError."""
    with pytest.raises(ValueError):
        IPCCacheServerKey(
            model_name="Qwen/Qwen3-8B",
            world_size=2,
            worker_id=0,
            token_bytes=b"\x00\x00\x00",
            start=0,
            end=0,
            request_id="req-1",
        )
