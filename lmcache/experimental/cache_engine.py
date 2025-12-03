# Compatibility shim for vLLM V0 which expects lmcache.experimental.cache_engine
# Re-export from the actual location
from lmcache.cache_engine import LMCacheEngine, LMCacheEngineBuilder

__all__ = ['LMCacheEngine', 'LMCacheEngineBuilder']

