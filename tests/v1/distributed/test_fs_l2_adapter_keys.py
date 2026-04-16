# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for fs_l2_adapter key serialization helpers.

These helpers round-trip ObjectKey <-> filename. Salted keys are marked
by a leading ``@@`` prefix so legacy keys with ``@`` in the model name
are still parseable.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _filename_to_object_key,
    _object_key_to_filename,
)


class TestFilenameRoundtrip:
    """_object_key_to_filename and _filename_to_object_key must be
    exact inverses for both legacy and salted keys."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "llama",
            "meta-llama/Llama-3",  # has '/', must survive PATH_SLASH_REPLACEMENT
            "ns@model",  # has '@', exercises rsplit parsing
        ],
    )
    @pytest.mark.parametrize("cache_salt", ["", "alice", "user-abc_123.xyz:42"])
    def test_roundtrip(self, model_name: str, cache_salt: str):
        key = ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name=model_name,
            kv_rank=42,
            cache_salt=cache_salt,
        )
        fn = _object_key_to_filename(key)
        assert fn.endswith(".data")
        # All filenames now start with @@ (unified format).
        assert fn.startswith("@@")
        if cache_salt:
            assert fn.startswith("@@" + cache_salt + "@")
        else:
            assert fn.startswith("@@@")  # @@ + empty salt + @
        parsed = _filename_to_object_key(fn)
        assert parsed == key

    def test_legacy_format_still_parseable(self):
        """Files written before the @@ prefix was required must remain
        readable (with a deprecation warning)."""
        legacy = "llama@0x0000002a@deadbeef.data"
        parsed = _filename_to_object_key(legacy)
        assert parsed == ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="llama",
            kv_rank=42,
            cache_salt="",
        )

    def test_salted_format(self):
        fn = "@@alice@llama@0x0000002a@deadbeef.data"
        parsed = _filename_to_object_key(fn)
        assert parsed == ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="llama",
            kv_rank=42,
            cache_salt="alice",
        )

    def test_empty_salt_format(self):
        """Empty salt now uses @@@model@... (not legacy)."""
        fn = "@@@llama@0x0000002a@deadbeef.data"
        parsed = _filename_to_object_key(fn)
        assert parsed == ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="llama",
            kv_rank=42,
            cache_salt="",
        )

    def test_non_data_file_returns_none(self):
        assert _filename_to_object_key("not-a-data-file.txt") is None

    def test_malformed_filename_returns_none(self):
        # Missing field separator — cannot split into model/rank/hash.
        assert _filename_to_object_key("garbage.data") is None

    def test_malformed_salted_filename_returns_none(self):
        # @@ prefix but nothing else — no salt separator.
        assert _filename_to_object_key("@@onlyprefix.data") is None


class TestIpcKeyToObjectKeys:
    """ipc_key_to_object_keys reads cache_salt from the ipc_key itself —
    there is no separate parameter, so callers cannot accidentally drop
    the salt."""

    def test_forwards_cache_salt_single_worker(self):
        # First Party
        from lmcache.v1.distributed.api import ipc_key_to_object_keys
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

        k = IPCCacheEngineKey.from_token_ids(
            model_name="m",
            world_size=1,
            worker_id=0,
            token_ids=[1, 2, 3],
            cache_salt="alice",
        )
        out = ipc_key_to_object_keys(k, [b"h1", b"h2"])
        assert len(out) == 2
        assert all(o.cache_salt == "alice" for o in out)

    def test_forwards_cache_salt_scheduler_path(self):
        """worker_id=None explodes one chunk into one ObjectKey per worker."""
        # First Party
        from lmcache.v1.distributed.api import ipc_key_to_object_keys
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

        k = IPCCacheEngineKey.from_token_ids(
            model_name="m",
            world_size=4,
            worker_id=None,
            token_ids=[1, 2, 3],
            cache_salt="alice",
        )
        out = ipc_key_to_object_keys(k, [b"h1"])
        assert len(out) == 4
        assert all(o.cache_salt == "alice" for o in out)

    def test_empty_salt_passes_through(self):
        # First Party
        from lmcache.v1.distributed.api import ipc_key_to_object_keys
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

        k = IPCCacheEngineKey.from_token_ids(
            model_name="m",
            world_size=1,
            worker_id=0,
            token_ids=[1],
        )
        out = ipc_key_to_object_keys(k, [b"h1"])
        assert all(o.cache_salt == "" for o in out)


class TestIPCCacheEngineKeyCacheSalt:
    """cache_salt on IPCCacheEngineKey: validation + wire compat."""

    def test_reject_at_in_salt(self):
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

        with pytest.raises(ValueError, match="cache_salt"):
            IPCCacheEngineKey.from_token_ids(
                model_name="m",
                world_size=1,
                worker_id=0,
                token_ids=[1],
                cache_salt="a@b",
            )

    def test_reject_slash_in_salt(self):
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

        with pytest.raises(ValueError, match="cache_salt"):
            IPCCacheEngineKey.from_token_ids(
                model_name="m",
                world_size=1,
                worker_id=0,
                token_ids=[1],
                cache_salt="tenant/alice",
            )

    def test_no_worker_id_version_preserves_salt(self):
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

        k = IPCCacheEngineKey.from_token_ids(
            model_name="m",
            world_size=4,
            worker_id=2,
            token_ids=[1],
            cache_salt="alice",
        )
        k2 = k.no_worker_id_version()
        assert k2.worker_id is None
        assert k2.cache_salt == "alice"

    def test_wire_compat_old_payload_decodes(self):
        """An old 7-field msgspec payload must decode cleanly on new code
        with cache_salt defaulting to ""."""
        # Third Party
        import msgspec

        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

        old_payload = {
            "model_name": "m",
            "world_size": 1,
            "worker_id": 0,
            "token_ids": (1, 2),
            "start": 0,
            "end": 2,
            "request_id": "r1",
        }
        wire = msgspec.msgpack.encode(old_payload)
        decoded = msgspec.msgpack.decode(wire, type=IPCCacheEngineKey)
        assert decoded.cache_salt == ""

    def test_wire_compat_new_payload_roundtrip(self):
        # Third Party
        import msgspec

        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

        k = IPCCacheEngineKey.from_token_ids(
            model_name="m",
            world_size=1,
            worker_id=0,
            token_ids=[1, 2],
            cache_salt="alice",
        )
        wire = msgspec.msgpack.encode(k)
        decoded = msgspec.msgpack.decode(wire, type=IPCCacheEngineKey)
        assert decoded == k
        assert decoded.cache_salt == "alice"
