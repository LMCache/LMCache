# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for fs_l2_adapter key serialization helpers.

These helpers round-trip ObjectKey <-> filename. ``cache_salt`` is
appended as a trailing field when non-empty; unsalted keys use the
3-field shape that matches what older LMCache builds wrote to disk.
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
    """``_object_key_to_filename`` and ``_filename_to_object_key`` are
    exact inverses for both the 3-field (unsalted) and 4-field (salted)
    shapes."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "llama",
            "meta-llama/Llama-3",  # has '/', must survive PATH_SLASH_REPLACEMENT
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
        # Salted filenames gain a trailing "@<salt>" before ".data".
        if cache_salt:
            assert fn.endswith("@" + cache_salt + ".data")
        parsed = _filename_to_object_key(fn)
        assert parsed == key

    def test_unsalted_format(self):
        """Unsalted keys use the 3-field shape — identical to the
        pre-cache_salt filename format, so existing caches stay valid."""
        fn = "llama@0x0000002a@deadbeef.data"
        parsed = _filename_to_object_key(fn)
        assert parsed == ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="llama",
            kv_rank=42,
            cache_salt="",
        )

    def test_salted_format(self):
        """Salted keys append ``@<cache_salt>`` before the extension."""
        fn = "llama@0x0000002a@deadbeef@alice.data"
        parsed = _filename_to_object_key(fn)
        assert parsed == ObjectKey(
            chunk_hash=b"\xde\xad\xbe\xef",
            model_name="llama",
            kv_rank=42,
            cache_salt="alice",
        )

    def test_non_data_file_returns_none(self):
        assert _filename_to_object_key("not-a-data-file.txt") is None

    def test_too_few_fields_returns_none(self):
        assert _filename_to_object_key("just-one-field.data") is None

    def test_too_many_fields_returns_none(self):
        assert _filename_to_object_key("a@b@c@d@e.data") is None

    def test_salt_with_forbidden_char_returns_none(self):
        # A filename that parses into 4 fields but whose trailing "salt"
        # slot contains a char ObjectKey.__post_init__ rejects (NUL here
        # is impossible in filenames, so use the length cap instead).
        too_long_salt = "x" * 129
        fn = f"llama@0x0000002a@deadbeef@{too_long_salt}.data"
        assert _filename_to_object_key(fn) is None


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
