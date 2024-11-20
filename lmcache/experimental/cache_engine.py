"""
High-level design

MemoryObj:  -- Done
    raw_array
    metadata

PinBuffer: -- Done
    - Allocate(shape) -> MemoryOb 
    - Free(MemoryObj)

GPUConnector:
    # MemoryObj is flat + shape as metadata
    # Target buffer is paged memory or something else
    - to_gpu(MemoryObj, **kwargs)
    - to_host(dst_MemoryObj, **kwargs) 

TokenDB:
    - process_tokens(tokens, mask) -> List[CacheEngineKey]
    - insert(tokens, mask) -> List[CacheEngineKey]

LMCacheEngine:
    - __init__() # pin buffer, gpu connector, token db, backend manager
    - store_from_paged_memory()
    - retrieve_to_paged_memory()
    - retrieve_layers()
    - prefetch

LMCBackendInterface:
    - put()
    - get()
    - prefetch()

LMCBackendConnector:
    - put_task()
    - get_task()
"""
