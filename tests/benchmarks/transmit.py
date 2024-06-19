import torch
import time
from lmcache.cache_engine import LMCacheEngine
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata

if __name__ == "__main__":
    config = LMCacheEngineConfig.from_file("../examples/example.yaml")
    meta = LMCacheEngineMetadata("mistralai/Mistral-7B-Instruct-v0.2", 1, 0, "vllm")
    engine = LMCacheEngine(config, meta)
    hybrid_store = engine.engine_
    remote_store = hybrid_store.remote_store
    keys = remote_store.list()
    for key in keys:
        data = remote_store.connection.get(remote_store._combine_key(key))
    print("Job done")