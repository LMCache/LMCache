# SPDX-License-Identifier: Apache-2.0
"""
Overview
--------
This server enables KV cache reuse across requests that share token
sub-sequences at *arbitrary positions*, not only at a common prefix.

Workflow (example: chunk_size = 3)
-----------------------------------
1. cb_store_pre_computed([1,2,3,4,5,6])
   Tokens are split into full chunks ([1,2,3] and [4,5,6]).  Each chunk
   is stored in the underlying storage under its normal rolling prefix
   hash, and the chunk fingerprints are registered in
   BlendTokenRangeMatcher for fast sub-sequence lookup.  Because normal
   hashes are used, these chunks are also accessible via the standard
   lookup/retrieve path.

2. cb_lookup_pre_computed([x,y,z, a,b,c, 4,5,6, m,n,p])
   BlendTokenRangeMatcher slides a rolling polynomial hash over the new
   request's tokens and detects that the window at positions [6, 9)
   matches the stored chunk [4,5,6].  A prefetch task is submitted for
   that chunk using its stored hash as the storage key.  Only chunks
   confirmed present in storage are returned as CBMatchResult objects
   (with cur_st/cur_ed pointing to their location in the new request).

3. cb_retrieve_pre_computed(...)
   The (prefetched) KV cache for each matched chunk is copied (CPU→GPU)
   into the correct slot of the new request's KV cache buffer (at
   cur_st + offset), so the LLM can skip recomputing those tokens.

4. cb_store_final([x,y,z, a,b,c, 4,5,6, m,n,p])
   After inference completes on the new request, all its chunks are
   stored under normal prefix hashes.  Future requests sharing
   any prefix of the new request will get standard prefix-cache hits.
   Future requests sharing any prefix of the first request will also
   get hits because cb_store_pre_computed already stored those chunks
   under normal hashes.

This module is now a thin entry point.  The blend engine logic lives in
:mod:`lmcache.v1.multiprocess.modules.blend` (``BlendModule``), assembled
by the unified :func:`lmcache.v1.multiprocess.server.run_cache_server`.
"""

# First Party
from lmcache.v1.distributed.config import parse_args_to_config
from lmcache.v1.mp_observability.config import parse_args_to_observability_config
from lmcache.v1.multiprocess.config import parse_args_to_mp_server_config
from lmcache.v1.multiprocess.server import parse_args, run_cache_server

if __name__ == "__main__":
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    mp_config.engine_type = "blend"
    storage_manager_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
    )
