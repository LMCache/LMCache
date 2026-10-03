# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import time

# Third Party
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.ec_engine import ECCacheEngine
from lmcache.v1.metadata import LMCacheMetadata


def _make_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test-ec-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 2, 256, 1, 1),
        role="worker",
    )


def _put_eventually(
    engine: ECCacheEngine,
    mm_hash: str,
    tensor: torch.Tensor,
    timeout: float = 5.0,
) -> bool:
    deadline = time.monotonic() + timeout
    while not engine.put(mm_hash, tensor):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def test_get_falls_back_to_disk_after_cpu_eviction(tmp_path: Path) -> None:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.002,
        local_disk=str(tmp_path),
        max_local_disk_size=1,
        lmcache_instance_id="test-ec-tier-fallback",
    )
    engine = ECCacheEngine(config, _make_metadata(), torch.float16)

    try:
        first = torch.full((750, 800), 1.5, dtype=torch.float16)
        second = torch.full((750, 800), -2.25, dtype=torch.float16)

        assert engine.put("first", first)

        # The pool can hold either tensor but not both. A successful second
        # put therefore means the first entry was evicted from local CPU.
        assert _put_eventually(engine, "second", second)

        deadline = time.monotonic() + 5.0
        while not engine.contains("first") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert engine.contains("first")

        restored = engine.get("first", "cpu")

        assert restored is not None
        assert torch.equal(restored, first)
    finally:
        engine.close()
