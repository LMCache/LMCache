# KV Bytes Engine Helpers

> Status: v1 helper module for the MP HTTP KV cache API.

## Goal

`lmcache/v1/multiprocess/kv_bytes.py` contains the bytes-level store and
retrieve implementation used by `MPCacheEngine` wrappers. Keeping this
logic outside `server.py` limits the MP server integration diff to model
resolution and thin public methods.

## Boundaries

`MPCacheEngine` owns the registered GPU contexts, token hasher, storage
manager, and chunk size. The helper module receives those dependencies as
arguments and does not import `MPCacheEngine`, avoiding a circular
dependency.

The helper module owns:

- Validating the v1 homogeneous `KV_2LTD` layout contract.
- Splitting streamed store chunks into per-worker `MemoryObj` shards.
- Creating lazy retrieve results that stream one worker shard at a time.
- Releasing storage read/write locks on normal and early-close paths.

The HTTP layer owns frame encoding, protocol-version validation, and
request/response status mapping.
