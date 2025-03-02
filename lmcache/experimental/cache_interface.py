from typing import Optional
import msgspec

class LMCacheModelRequest(
        msgspec.Struct,
        array_like=True,
        omit_defaults=True):
     
    store_cache: bool = True # Whether to store the cache
    ttl: Optional[float] = None # Time to live