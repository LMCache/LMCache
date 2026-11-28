# SPDX-License-Identifier: Apache-2.0
"""Native connector batches must address the same objects as scalar writers."""

# Standard
from pathlib import Path
import select

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import (
    encode_native_storage_keys,
    encode_storage_key,
)


def test_batch_preserves_order_and_identity_transitions() -> None:
    first = ObjectKey(b"a", "org/model", 1, 2, tags={"a": "红", "b": "/@"})
    other_group = ObjectKey(b"b", "另/model", 3, 4, tags={"b": "/@", "a": "红"})
    salted = ObjectKey(b"a", "org/model", 1, 2, "租户", tags={"a": "红", "b": "/@"})
    new_tags = ObjectKey(b"a", "org/model", 1, 2, tags={"a": "blue"})
    plain = ObjectKey(b"c", "org/model", 1, 2)
    for keys in (
        [],
        [first],
        [first, first, other_group, salted, first],
        [first, other_group, new_tags],
        [plain, salted, plain],
        [plain, plain],
    ):
        assert encode_native_storage_keys(keys) == [encode_storage_key(k) for k in keys]


def test_batch_reads_files_written_with_scalar_keys(tmp_path: Path) -> None:
    native = pytest.importorskip("lmcache.lmcache_fs")
    client = native.LMCacheFSClient(str(tmp_path), 1, "", False, 0, 1, 0)
    keys = [
        ObjectKey(bytes([i]), "org/model", 0, tags={"tenant": "a"}) for i in range(2)
    ]
    payloads = [bytearray(b"first"), bytearray(b"other")]
    try:
        stored = client.submit_batch_set(
            [encode_storage_key(k) for k in keys], [memoryview(p) for p in payloads]
        )
        assert select.select([client.event_fd()], [], [], 5)[0]
        [(future, ok, error, _)] = client.drain_completions()
        assert future == stored and ok, error
        buffers = [bytearray(len(p)) for p in payloads]
        loaded = client.submit_batch_get(
            encode_native_storage_keys(keys), [memoryview(b) for b in buffers]
        )
        assert select.select([client.event_fd()], [], [], 5)[0]
        [(future, ok, error, bits)] = client.drain_completions()
        assert future == loaded and ok, error
        assert bits == [True, True]
        assert buffers == payloads
    finally:
        client.close()
