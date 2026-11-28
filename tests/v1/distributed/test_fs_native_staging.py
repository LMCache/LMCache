# SPDX-License-License-Identifier: Apache-2.0
"""Staging-file behavior of the native C++ filesystem connector.

Every store writes through its own staging file, and the staging name
must not derive from the final filename: the key codecs cap nothing, and
long model paths or tenant salts put final names within ~30 bytes of
filesystems' filename component limit. These tests pin the two
properties from the review of the unique-staging change:

- a final name just under the limit still stores: a staging name
  derived from the final one appended ".tmp.<pid>-<n>" to it and
  failed every store of such keys with ENAMETOOLONG, although the
  keys store fine;
- two concurrent stores of one key both succeed and publish exactly
  one writer's full payload, with no staging residue left behind.
"""

# Standard
from pathlib import Path
import select
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import encode_fs_filename, encode_storage_key

try:
    # First Party
    from lmcache.lmcache_fs import LMCacheFSClient
except ImportError:
    LMCacheFSClient = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(
    LMCacheFSClient is None,
    reason="native FS extension is not built",
)


def _key(model_name: str) -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes.fromhex("ff" * 32),
        model_name=model_name,
        kv_rank=0,
    )


def _wait_completions(
    client: LMCacheFSClient,
    future_ids: list[int],
    timeout: float = 10.0,
) -> dict[int, tuple[bool, str, list[bool] | None]]:
    """Drain the single completion stream until every listed future lands.

    The stream is shared, so completions of a concurrent batch's other
    futures must not be dropped while waiting for the first.
    """
    pending = set(future_ids)
    results: dict[int, tuple[bool, str, list[bool] | None]] = {}
    deadline = time.monotonic() + timeout
    while pending:
        for fid, ok, error, per_key in client.drain_completions():
            if fid in pending:
                results[fid] = (ok, error, per_key)
                pending.discard(fid)
        if not pending:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"timed out waiting for futures {pending}")
        select.select([client.event_fd()], [], [], min(remaining, 0.05))
    return results


@pytest.mark.parametrize("relative_tmp_dir", ["", "staging"])
def test_store_with_final_filename_near_the_255_byte_limit(
    tmp_path: Path,
    relative_tmp_dir: str,
) -> None:
    """A final name just under the filesystem limit must still store."""
    # A real-world-shaped model path, padded so the final filename is 250
    # bytes: inside the 255-byte limit, but past what any final-name plus
    # ".tmp.<suffix>" staging can reach (255 - ~28). The baseline already
    # includes the terminating "/" (-> "-SEP-"), so every padding "s" adds
    # exactly one byte.
    model_root = "/models/hub/models--org--Llama-3.3-70B-Instruct/snapshots"
    unpadded = _key(model_root + "/")
    model = model_root + "/" + "s" * (250 - len(encode_fs_filename(unpadded)))
    obj = _key(model)
    final_name = encode_fs_filename(obj)
    assert len(final_name) == 250
    key = encode_storage_key(obj)

    size = 4096
    payload = bytes(range(256)) * (size // 256)
    dest = bytearray(size)

    client = LMCacheFSClient(str(tmp_path), 1, relative_tmp_dir, False)
    try:
        [store] = _wait_completions(
            client, [client.submit_batch_set([key], [memoryview(payload)])]
        ).values()
        assert store[0], store[1]
        assert (tmp_path / final_name).read_bytes() == payload

        [load] = _wait_completions(
            client, [client.submit_batch_get([key], [memoryview(dest)])]
        ).values()
        assert load[0], load[1]
        assert list(load[2]) == [True]
        assert bytes(dest) == payload

        residue = [e.name for e in tmp_path.rglob("*") if ".tmp" in e.name]
        assert residue == []
    finally:
        client.close()


def test_concurrent_same_key_stores_publish_one_full_payload(
    tmp_path: Path,
) -> None:
    """Two concurrent stores of one key publish one writer's full payload.

    With a shared staging filename, the second writer's O_TRUNC cut into
    the first writer's staged bytes and one rename published a torn file
    even though both stores reported success. Per-write staging files
    make the published file exactly one store's complete payload.
    """
    size = 8 << 20
    payload_a = b"\xa5" * size
    payload_b = b"\x5a" * size

    client = LMCacheFSClient(str(tmp_path), 2, "", False)
    try:
        for round_index in range(8):
            obj = _key(f"test_model-{round_index}")
            key = encode_storage_key(obj)
            [store_a, store_b] = _wait_completions(
                client,
                [
                    client.submit_batch_set([key], [memoryview(payload_a)]),
                    client.submit_batch_set([key], [memoryview(payload_b)]),
                ],
            ).values()
            assert store_a[0], f"round {round_index}: store a failed: {store_a[1]}"
            assert store_b[0], f"round {round_index}: store b failed: {store_b[1]}"

            published = (tmp_path / encode_fs_filename(obj)).read_bytes()
            assert published in (payload_a, payload_b), (
                f"round {round_index}: the published file is torn"
            )

            residue = [e.name for e in tmp_path.rglob("*") if ".tmp" in e.name]
            assert residue == [], f"round {round_index}: staging files left: {residue}"
    finally:
        client.close()
