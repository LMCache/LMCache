import argparse
import asyncio
import random
import time
from collections import OrderedDict
from lmcache.experimental.storage_backend.storage_manager import KVCacheManager
from lmcache.logging import init_logger
from lmcache.utils import CacheManagerMetadata, CacheEngineKey

SEED = 42
# Based on 0.9766 GB per 8000 tokens
BYTES_PER_TOKEN = 0.9766 * (2**30) / 8000
CHUNK_TOKENS = 1024
CHUNK_BYTES = int(BYTES_PER_TOKEN * CHUNK_TOKENS)
RATE = 1

async def run_context(round_idx: int,
                      manager: KVCacheManager,
                      hot_cache: OrderedDict,
                      lock: asyncio.Lock) -> float:
    # Build a single context's chunks
    num_chunks = random.randint(3, 14)
    to_save_list = OrderedDict()
    for chunk_idx in range(num_chunks):
        metadata = CacheManagerMetadata(
            context_id=[f"context_{round_idx}"],
            method=["kivi"],
            rate=RATE,
            length=CHUNK_BYTES,
            num_tokens=CHUNK_TOKENS * num_chunks,
            score_table=[[(1.0, 1.0), (0.7286, 0.8), (0.4857, 0.6), (0.3714, 0.4), (0.0, 0.2)]],
            emerge_id=[1],
            disk_score_table=[[(1.0, 0.38), (0.7286, 0.36), (0.4857, 0.34), (0.3714, 0.32), (0.0, 0.3)]]
        )
        cache_key = CacheEngineKey(
            fmt="v1",
            model_name="demo_model",
            world_size=1,
            worker_id=0,
            chunk_hash=f"hash_{round_idx}_{chunk_idx}",
            metadata=metadata
        )
        to_save_list[cache_key] = None

    # Measure overhead of inform_new
    start = time.monotonic()
    decisions, update_dict = manager.inform_new(to_save_list)
    duration = time.monotonic() - start

    # Apply cache updates under lock
    async with lock:
        for key, upd in update_dict.items():
            update_rate = upd.compression_rate
            update_device = upd.device
            if (update_device == "cpu" and update_rate == 0) or update_device == "disk":
                hot_cache.pop(key, None)
            else:
                key.metadata.length = int(key.metadata.length * update_rate)
                key.metadata.rate = update_rate
        for key in to_save_list:
            key.metadata.length = int(key.metadata.length * decisions.compression_rate)
            key.metadata.rate = decisions.compression_rate
            if key.metadata.length > 0:
                hot_cache[key] = None

    return duration

async def schedule_contexts(qps: float,
                            total_rounds: int,
                            manager: KVCacheManager,
                            hot_cache: OrderedDict) -> None:
    lock = asyncio.Lock()
    tasks = []
    interval = 1.0 / qps
    for i in range(total_rounds):
        tasks.append(asyncio.create_task(run_context(i, manager, hot_cache, lock)))
        await asyncio.sleep(interval)

    durations = await asyncio.gather(*tasks)
    avg_ms = (sum(durations) / len(durations)) * 1000
    print(f"Completed {total_rounds} contexts at {qps} QPS")
    print(f"Average inform_new duration: {avg_ms:.2f} ms")

def main():
    parser = argparse.ArgumentParser(
        description="Test overhead of KVCacheManager.inform_new with controlled QPS and concurrency."
    )
    parser.add_argument("--qps", type=float, default=10.0,
                        help="Contexts per second to simulate")
    parser.add_argument("--rounds", type=int, default=3000,
                        help="Total number of contexts to run")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    logger = init_logger(__name__)
    hot_cache = OrderedDict()
    manager = KVCacheManager(hot_cache, "ours", RATE)

    asyncio.run(schedule_contexts(args.qps, args.rounds, manager, hot_cache))

if __name__ == "__main__":
    main()
