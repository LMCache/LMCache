# HiSparse with LMCache

## Scope and wiring
- Use current vLLM with `MultiConnector`: `HiSparseConnector` first, then
  `LMCacheMPConnector` from `lmcache.integration.vllm.lmcache_mp_connector`.
- `get_cache_group_ids` registers CPU MLA source and GPU indexer groups.
  Private resident/hot pools are excluded; original group IDs are preserved.
- vLLM owns every original allocation; LMCache never rebinds CPU storage.
- Keys use `##lmcache-hisparse-v1`, separate from earlier indexer-only objects.

## Transport (`transfer_context/mixed.py`)
- `create_transfer_context` selects `MixedTransferContext` for CPU/CUDA caches
  in `auto` or `lmcache_driven` mode.
- Register the original CUDA indexer views and a GPU staging tensor for each
  CPU MLA layer: one LMCache chunk plus a reserved null block.
- CPU MLA uses uncompressed, contiguous blocks
  (rank 3, or rank 4 with one head).
- Split store/retrieve requests at chunk boundaries; retain each group's
  original block IDs, replacing CPU IDs with staging IDs only in the RPC.
- Send both groups through the existing multi-group IPC/storage protocol.
  No server changes or full-host-pool GPU mirror are required.

## Ordering and completion
1. Save: HiSparse's `finish_forward` makes compute wait for host writes;
   LMCache records its producer event, waits on the staging stream, and
   synchronizes that stream before PyTorch can stage CPU reads.
2. Copy CPU MLA to GPU staging, then record the event passed to server STORE.
3. Wait for the server's completion event before reusing staging.
4. Load: RETRIEVE fills GPU indexer and staged MLA; wait for its completion,
   copy MLA to the original CPU pages, and wait for that copy stream.
5. Report load completion only then; vLLM rebuilds HiSparse's private pools.
- Preserve locally computed prefix blocks and null blocks during restore.
- Transfers serialize and block the worker per chunk; this first functional
  path trades throughput for simplicity, without device-wide synchronization.

## Verification
`tests/v1/test_vllm_integration.py` runs a live LMCache server and a small dummy
DeepSeek model on H200: store, evict/overwrite CPU MLA and GPU indexer, restore,
compare both caches byte-for-byte, and compare greedy outputs.
Transfer tests also cover multi-chunk remapping, skipped prefixes, and failures.
