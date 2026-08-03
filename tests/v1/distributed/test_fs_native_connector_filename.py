# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the C++ FS connector filename encoding.

These exercise the real pybind-wrapped ``LMCacheFSClient`` to keep
``FSConnector::key_to_filename`` in sync with the Python serialization in
``native_connector_l2_adapter._object_key_to_string`` (its input) and
``fs_l2_adapter._object_key_to_filename`` (the filename it must produce).

They require the C++ extension to be built and are skipped otherwise.
"""

# Standard
import os
import select
import shutil
import tempfile

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _object_key_to_filename,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    _object_key_to_string,
)
from lmcache.v1.platform import consume_fd

# Skipped entirely when the native FS extension is not built.
lmcache_fs = pytest.importorskip("lmcache.lmcache_fs")
LMCacheFSClient = lmcache_fs.LMCacheFSClient


def _drain_one_completion(client, expected_fid, timeout=5.0):
    """Block until the completion for *expected_fid* is drained.

    Returns the ``(ok, error)`` pair reported by the C++ connector. A
    ``set`` whose ``key_to_filename`` throws surfaces here as
    ``ok=False`` with the exception text in *error* — that is exactly
    the regression this module guards against.
    """
    poll = select.poll()
    poll.register(client.event_fd(), select.POLLIN)
    deadline_remaining = timeout
    while deadline_remaining > 0:
        events = poll.poll(int(deadline_remaining * 1000))
        if not events:
            break  # timed out
        try:
            consume_fd(client.event_fd())
        except BlockingIOError:
            pass
        for fid, ok, error, _result in client.drain_completions():
            if fid == expected_fid:
                return ok, error
        deadline_remaining = timeout  # event consumed; keep waiting
    raise AssertionError(
        f"timed out waiting for completion of future_id={expected_fid}"
    )


@pytest.fixture
def fs_client(tmp_path):
    """A real ``LMCacheFSClient`` rooted at a per-test temp directory."""
    base = tmp_path / "fs_root"
    base.mkdir()
    client = LMCacheFSClient(str(base), 1)
    yield client, str(base)
    client.close()
    shutil.rmtree(base, ignore_errors=True)


class TestFSConnectorFilenameEncoding:
    """The C++ connector must accept the current ``_object_key_to_string``
    wire format (4 fields unsalted / 5 salted) and emit filenames that
    match ``_object_key_to_filename``."""

    def test_unsalted_key_filename_matches_python(self, fs_client):
        client, base = fs_client
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
            object_group_id=5,
        )
        key_str = _object_key_to_string(key)
        assert key_str == "llama@000000ff@5@00010203"

        buf = memoryview(bytearray(16))
        fid = client.submit_batch_set([key_str], [buf])
        ok, error = _drain_one_completion(client, fid)
        assert ok, f"set failed (key_to_filename rejected the key): {error}"

        assert os.listdir(base) == [_object_key_to_filename(key)]
        assert os.listdir(base) == ["llama@0x000000ff@5@00010203.data"]

    def test_salted_key_filename_matches_python(self, fs_client):
        client, base = fs_client
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
            object_group_id=0,
            cache_salt="alice",
        )
        key_str = _object_key_to_string(key)
        assert key_str == "llama@000000ff@0@00010203@alice"

        buf = memoryview(bytearray(16))
        fid = client.submit_batch_set([key_str], [buf])
        ok, error = _drain_one_completion(client, fid)
        assert ok, f"set failed (key_to_filename rejected the key): {error}"

        assert os.listdir(base) == [_object_key_to_filename(key)]
        assert os.listdir(base) == ["llama@0x000000ff@0@00010203@alice.data"]

    def test_model_name_slash_is_escaped(self, fs_client):
        client, base = fs_client
        key = ObjectKey(
            chunk_hash=b"\xab\xcd",
            model_name="org/model",
            kv_rank=1,
            object_group_id=2,
        )
        key_str = _object_key_to_string(key)

        buf = memoryview(bytearray(8))
        fid = client.submit_batch_set([key_str], [buf])
        ok, error = _drain_one_completion(client, fid)
        assert ok, f"set failed: {error}"

        # '/' in model_name becomes '-SEP-' on disk, matching the
        # Python adapter.
        assert os.listdir(base) == [_object_key_to_filename(key)]
        assert os.listdir(base)[0].startswith("org-SEP-model@")
