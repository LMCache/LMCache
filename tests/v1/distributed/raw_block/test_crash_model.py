# SPDX-License-Identifier: Apache-2.0
"""A crash model for reuse after a delete.

Every device write call is logged; device images for process crashes at
sampled points are rebuilt from the log (calls that finished land whole,
posix calls in flight as an in-order prefix, io_uring requests in flight
each as nothing, a block prefix or all), recovered, and every recovered key
must hold the bytes of one write of that key."""

# Standard
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import importlib.util
import itertools
import random
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.raw_block import RawBlockCore, encode_object_key
from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import (
    IoUringBlockIO,
    PosixBlockIO,
)
from tests.v1.distributed.raw_block.test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io is not installed",
)

ALIGN = RAW_BLOCK_CI_BLOCK_ALIGN
HEADER = RAW_BLOCK_CI_HEADER_BYTES


def _spec(i: int):
    return encode_object_key(make_object_key(i))


def _config(tmp_path: Path, engine: str, data_bytes: int, odirect: bool = False):
    total = RAW_BLOCK_CI_META_TOTAL_BYTES + data_bytes
    path = make_raw_block_file(tmp_path, total)
    return replace(
        make_raw_block_core_config(path, total),
        io_engine=engine,
        use_odirect=odirect,
        meta_enable_periodic=False,
    )


class _WriteLog:
    """Records every engine write call with start/end tickets and its bytes."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ticket = itertools.count()
        self.calls: list[dict] = []

    def wrap(self, real, kind: str):
        log = self

        def write_buffers(self, offsets, buffers, payload_lens, total_lens, *a, **k):
            sqes = []
            for off, buf, pl, tl in zip(
                offsets, buffers, payload_lens, total_lens, strict=True
            ):
                mv = memoryview(buf).cast("B")
                data = bytes(mv[:pl]) + bytes(tl - pl)
                sqes.append((int(off), data))
            with log.lock:
                call = {"start": next(log.ticket), "end": None, "kind": kind}
                call["sqes"] = sqes
                log.calls.append(call)
            real(self, offsets, buffers, payload_lens, total_lens, *a, **k)
            with log.lock:
                call["end"] = next(log.ticket)

        return write_buffers

    def image(self, base: bytes, t: int, rng: random.Random) -> bytearray:
        """Device bytes after a process crash at ticket ``t``.

        Completed calls land whole. A call in flight lands partly: posix as
        a prefix of its buffer sequence (pwrite in order, each a prefix);
        io_uring per request independently: nothing, a block prefix, or all.
        """
        img = bytearray(base)
        done = sorted(
            (c for c in self.calls if c["end"] is not None and c["end"] < t),
            key=lambda c: c["end"],
        )
        flying = [
            c
            for c in self.calls
            if c["start"] < t and (c["end"] is None or c["end"] > t)
        ]
        rng.shuffle(flying)
        for c in done:
            for off, data in c["sqes"]:
                img[off : off + len(data)] = data
        for c in flying:
            sqes = c["sqes"]
            if c["kind"] == "posix":
                cut = rng.randrange(len(sqes) + 1)
                for j, (off, data) in enumerate(sqes[: cut + 1]):
                    n = len(data) if j < cut else _prefix(rng, len(data))
                    img[off : off + n] = data[:n]
            else:
                for off, data in sqes:
                    n = rng.choice([0, len(data), _prefix(rng, len(data))])
                    img[off : off + n] = data[:n]
        return img


def _prefix(rng: random.Random, n: int) -> int:
    return min(n, ALIGN * rng.randrange(0, n // ALIGN + 1))


def _run_workload(core, versions, vcount, seed: int, ops: int = 120) -> _WriteLog:
    """Three threads of puts (1-3 keys, batched on io_uring), forced deletes
    and checkpoints on 24 keys; returns the log of every device write."""
    log = _WriteLog()
    vlock = threading.Lock()
    errors: list[BaseException] = []

    def worker(wid: int) -> None:
        rng = random.Random(seed * 100 + wid)
        try:
            for _ in range(ops):
                op = rng.random()
                if op < 0.6:
                    keys = rng.sample(range(24), rng.randint(1, 3))
                    objs, specs = [], []
                    for k in keys:
                        size = rng.choice([1, 2, 3, 5, 7]) * ALIGN - rng.choice(
                            [0, 100]
                        )
                        data = random.Random(next(vcount)).randbytes(size)
                        with vlock:
                            versions.setdefault(k, []).append(data)
                        objs.append(make_memory_obj(data))
                        specs.append(_spec(k))
                    core.put_many(specs, objs)
                elif op < 0.9:
                    keys = rng.sample(range(24), rng.randint(1, 3))
                    core.delete_many([_spec(k).encoded for k in keys], force=True)
                else:
                    core.checkpoint_now()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    with (
        patch.object(
            PosixBlockIO, "write_buffers", log.wrap(PosixBlockIO.write_buffers, "posix")
        ),
        patch.object(
            IoUringBlockIO,
            "write_buffers",
            log.wrap(IoUringBlockIO.write_buffers, "uring"),
        ),
    ):
        threads = [threading.Thread(target=worker, args=(w,)) for w in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
    assert not errors, errors
    return log


def _check_recovery(config, versions, where: str) -> tuple[int, list[str]]:
    """Recover the device file; return loaded keys and wrong-byte keys."""
    crashed = RawBlockCore(config)
    bad: list[str] = []
    loaded = 0
    try:
        keys = [k for k in range(24) if crashed.contains_key(_spec(k).encoded)]
        metas = crashed.get_metadata_many([_spec(k).encoded for k in keys])
        for k, meta in zip(keys, metas, strict=True):
            assert meta is not None
            obj = make_empty_memory_obj(meta.size)
            if crashed.load_many_into([_spec(k).encoded], [obj]) != [True]:
                continue
            loaded += 1
            if memory_obj_bytes(obj) not in versions.get(k, []):
                bad.append(f"{where} key={k}")
    finally:
        crashed.close()
    return loaded, bad


ENGINE_PARAMS = pytest.mark.parametrize(
    "engine,odirect",
    [("posix", False), ("io_uring", False), ("io_uring", True)],
    ids=["posix", "uring", "uring-odirect"],
)


@ENGINE_PARAMS
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_crash_at_any_point_never_returns_bytes_of_another_write(
    tmp_path, engine, odirect, seed
):
    """Concurrent puts (batched for io_uring), deletes and checkpoints on a
    small device so space is reused constantly. At sampled crash points the
    device image is built from the write log under the process-crash model,
    recovered, and every recovered key must load exactly the bytes of one
    write of that key."""
    config = _config(tmp_path, engine, 640 * 1024, odirect)
    versions: dict[int, list[bytes]] = {}
    vcount = itertools.count(seed * 1_000_000)
    core = RawBlockCore(config)
    base = Path(config.device_path).read_bytes()
    log = _run_workload(core, versions, vcount, seed)
    core.close()

    last = max(c["end"] for c in log.calls)
    rng = random.Random(seed)
    bad: list[str] = []
    total = 0
    for t in sorted(rng.sample(range(1, last + 1), min(60, last))):
        for _ in range(3):
            Path(config.device_path).write_bytes(log.image(base, t, rng))
            loaded, b = _check_recovery(config, versions, f"t={t}")
            total += loaded
            bad += b
    assert total > 50, total
    assert not bad, bad[:10]


@ENGINE_PARAMS
@pytest.mark.parametrize("seed", [1, 2])
def test_second_crash_after_recovering_from_a_first_one(
    tmp_path, engine, odirect, seed
):
    """Crash, recover (possibly past a torn checkpoint copy, with loaded
    entries listed by both containers), run again and crash again."""
    config = _config(tmp_path, engine, 640 * 1024, odirect)
    versions: dict[int, list[bytes]] = {}
    vcount = itertools.count(seed * 1_000_000)
    rng = random.Random(seed)
    core = RawBlockCore(config)
    base1 = Path(config.device_path).read_bytes()
    log1 = _run_workload(core, versions, vcount, seed)
    core.close()
    last1 = max(c["end"] for c in log1.calls)
    bad: list[str] = []
    total = 0
    for t1 in sorted(rng.sample(range(last1 // 2, last1 + 1), 6)):
        Path(config.device_path).write_bytes(log1.image(base1, t1, rng))
        core = RawBlockCore(config)
        base2 = Path(config.device_path).read_bytes()
        log2 = _run_workload(core, versions, vcount, seed + t1, ops=60)
        core.close()
        last2 = max(c["end"] for c in log2.calls)
        for t2 in sorted(rng.sample(range(1, last2 + 1), min(25, last2))):
            Path(config.device_path).write_bytes(log2.image(base2, t2, rng))
            loaded, b = _check_recovery(config, versions, f"t1={t1} t2={t2}")
            total += loaded
            bad += b
    assert total > 50, total
    assert not bad, bad[:10]


def test_dropping_a_loaded_index_also_forgets_its_retired_ranges(tmp_path):
    """Startup retires the space of an entry whose header cannot be read, then
    drops the whole index because the data-region marker is gone: the retired
    range must go with it."""
    config = _config(tmp_path, "posix", 1 << 20)
    core = RawBlockCore(config)
    assert core.put_many([_spec(1)], [make_memory_obj(bytes(2 * ALIGN))]).results == [
        True
    ]
    offset = core.entry_offset(_spec(1).encoded)
    marker = core.data_base_offset()
    core.close()
    with open(config.device_path, "r+b") as f:
        f.seek(marker)
        f.write(bytes(ALIGN))
    real_read = PosixBlockIO.read_buffers

    def unreadable(self, offsets, *args, **kwargs):
        if list(offsets) == [offset]:
            return [False]
        return real_read(self, offsets, *args, **kwargs)

    with patch.object(PosixBlockIO, "read_buffers", unreadable):
        core = RawBlockCore(config)
    try:
        status = core.report_status()
        assert core.indexed_key_count() == 0
        assert status["allocated_bytes"] == 0, status
        assert status["quarantined_bytes"] == 0, status
    finally:
        core.close()
