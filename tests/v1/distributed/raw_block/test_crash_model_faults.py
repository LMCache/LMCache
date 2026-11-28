# SPDX-License-Identifier: Apache-2.0
"""The crash model under buffer preparation failures and on a tight device
(stores run out of space, the device is reused constantly), also checking
that no space leaks."""

# Standard
from pathlib import Path
from unittest.mock import patch
import importlib.util
import itertools
import random
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.raw_block import RawBlockCore
from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import (
    IoUringBlockIO,
    PosixBlockIO,
)
from tests.v1.distributed.raw_block.test_crash_model import (
    ALIGN,
    _config,
    _spec,
)
from tests.v1.distributed.raw_block.test_utils import (
    make_memory_obj,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io is not installed",
)


@pytest.mark.parametrize("engine,odirect", [("io_uring", False), ("posix", False)])
@pytest.mark.parametrize("seed", [11, 12, 13, 14, 15, 16])
@pytest.mark.parametrize("fail_rate", [0.0, 0.15])
def test_crash_model_with_prep_failures(tmp_path, engine, odirect, seed, fail_rate):
    """The crash-model workload with buffer preparation failing at random:
    every recovered key must still hold the bytes of one of its writes."""
    # First Party
    from tests.v1.distributed.raw_block import test_crash_model as cm

    config = _config(tmp_path, engine, 640 * 1024, odirect)
    versions: dict[int, list[bytes]] = {}
    vcount = itertools.count(seed * 1_000_000)
    core = RawBlockCore(config)
    base = Path(config.device_path).read_bytes()
    # Fault injection: buffer preparation has no public hook.
    real = RawBlockCore._prepare_write_payload
    frng = random.Random(seed)
    flock = threading.Lock()

    def prep(self, obj):
        with flock:
            fail = frng.random() < fail_rate
        if fail:
            raise MemoryError("injected")
        return real(self, obj)

    with patch.object(RawBlockCore, "_prepare_write_payload", prep):
        log = cm._run_workload(core, versions, vcount, seed)
    core.close()
    last = max(c["end"] for c in log.calls)
    rng = random.Random(seed)
    bad: list[str] = []
    for t in sorted(rng.sample(range(1, last + 1), min(60, last))):
        for _ in range(3):
            Path(config.device_path).write_bytes(log.image(base, t, rng))
            _, b = cm._check_recovery(config, versions, f"t={t}")
            bad += b
    assert not bad, bad[:10]


def _tight_workload(core, versions, vcount, seed, *, ops, max_batch, fail_rate):
    """cm._run_workload on a device that runs out of space: batches up to
    ``max_batch`` keys, optional buffer preparation failures."""
    # First Party
    from tests.v1.distributed.raw_block import test_crash_model as cm

    log = cm._WriteLog()
    vlock = threading.Lock()
    errors: list[BaseException] = []
    # Fault injection: buffer preparation has no public hook.
    real = RawBlockCore._prepare_write_payload
    frng = random.Random(seed)

    def prep(self, obj):
        with vlock:
            fail = frng.random() < fail_rate
        if fail:
            raise MemoryError("injected")
        return real(self, obj)

    def worker(wid: int) -> None:
        rng = random.Random(seed * 100 + wid)
        try:
            for _ in range(ops):
                op = rng.random()
                if op < 0.6:
                    keys = rng.sample(range(24), rng.randint(1, max_batch))
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
        patch.object(RawBlockCore, "_prepare_write_payload", prep),
    ):
        threads = [threading.Thread(target=worker, args=(w,)) for w in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(300)
    assert not errors, errors
    return log


@pytest.mark.parametrize("engine", ["io_uring", "posix"])
@pytest.mark.parametrize("seed", [21, 22, 23, 24])
@pytest.mark.parametrize("fail_rate", [0.0, 0.1])
def test_tight_crash_model(tmp_path, engine, seed, fail_rate):
    """Device of 48 blocks for 24 keys of 2-8 blocks: stores run out of
    space and every delete is reused at once."""
    # First Party
    from tests.v1.distributed.raw_block import test_crash_model as cm

    config = _config(tmp_path, engine, 48 * ALIGN)
    versions: dict[int, list[bytes]] = {}
    vcount = itertools.count(seed * 1_000_000)
    core = RawBlockCore(config)
    base = Path(config.device_path).read_bytes()
    log = _tight_workload(
        core, versions, vcount, seed, ops=80, max_batch=6, fail_rate=fail_rate
    )
    core.delete_many([_spec(k).encoded for k in range(24)], force=True)
    after = core.report_status()
    core.close()
    last = max(c["end"] for c in log.calls)
    rng = random.Random(seed)
    bad: list[str] = []
    for t in sorted(rng.sample(range(1, last + 1), min(60, last))):
        for _ in range(3):
            Path(config.device_path).write_bytes(log.image(base, t, rng))
            _, b = cm._check_recovery(config, versions, f"t={t}")
            bad += b
    assert after["allocated_bytes"] == 0, after["allocated_bytes"]
    assert not bad, bad[:10]
