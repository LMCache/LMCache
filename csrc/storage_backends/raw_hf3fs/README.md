# raw_hf3fs — LMCache native 3FS (RDMA) L2 storage backend

A C++ native L2 backend that talks to a 3FS cluster **in-process** through the
3FS Storage Client API, registered under the type name `raw_hf3fs`. This
document covers the motivation, the design rationale (with measured data),
how to build and configure it, and how to test and evaluate it.

---

## 1. Background and motivation

The upstream `hf3fs` backend reads and writes through a FUSE mount point
(usrbio ring → FUSE daemon → 3FS client). Its write hot path has two
structural limitations:

1. **O(N) metadata**: one file per key, so every batch pays one FUSE-path
   open/close per key; MetaClient RPCs grow linearly with the number of keys;
2. **Serialized writes**: `write_file` submits a single IO and waits for it
   before issuing the next, keeping the whole process at an in-flight depth
   of roughly `num_workers` (4) — RDMA is heavily undersaturated; measured
   Store throughput was only ~480 MB/s.

`raw_hf3fs` removes all three bottlenecks at once (in-process direct
connection, single file + offsets, highly concurrent cross-chain batching).
Measured gains on the same harness, 4MB keys, same configuration:

| op | upstream hf3fs | raw_hf3fs | speedup |
|----|---------------|-----------|---------|
| Store | 456 MB/s | **4071.79 MB/s** | **8.93x** |
| Load | 3767 MB/s | **7050 MB/s** | **1.87x** |
| Lookup (0% hit) | 993K ops/s | 1282K ops/s | 1.29x |

Clean-state baseline (cluster freshly restarted,
`chunk_size=4MB`, `num_workers=4`; three repeated runs within ±5%):

| key size | Store (MB/s) | Load (MB/s) |
|----------|--------------|-------------|
| 32MB/key | 4178 / 4217 | 6923 / 6706 |
| 4MB/key  | 4087 / 4112 | 6106 / 6392 |

Bandwidth accounting (100Gbps NIC, 12.5 GB/s line rate): Store ~33% (the
2-replica copy happens between storage targets server-side and does not
consume client NIC bandwidth), Load ~55-57%. Store is independent of key
shape (32MB vs 4MB differ by <3%), i.e. the bottleneck is the server-side
write aggregation capability, not chunk geometry or client concurrency
(8 workers measured: Store unchanged at 4198 MB/s, Load rose to ~7150).

---

## 2. Architecture overview

```
LMCache Python (NativeConnectorL2Adapter)
        │  pybind11 (pybind.cpp)
        ▼
Hf3fsConnector : ConnectorBase<Hf3fsConnection>   (hf3fs_connector.h/.cc)
        │  worker pool + dedicated per-op lanes (lookup/retrieve/store/delete)
        ▼
DataStore — single file + offset map, metadata persistence, lease, free-list
        │                                   (data_store.h/.cc)
        ▼
3FS client stack: net::Client → MgmtdClientForClient → StorageClient + MetaClient
        │  (RDMA, set_force_use_tcp(false))
        ▼
3FS cluster (mgmtd / meta / storage services)
```

- **The client stack is constructed once and shared by all workers**;
  `StorageClient` runs background threads and is kept alive through a
  shared_ptr keepalive, so it does not die with a connection object.
- **Async pipeline**: Python `submit_batch_*` → C++ enqueue → worker threads
  execute → completion queue + eventfd notification → Python
  `drain_completions` fetches results. The pybind layer keeps a keepalive map
  so Python buffers are not GC'd while workers read/write them asynchronously.
- **Chain-aware grouping**: batches are grouped by (file, chain) and then
  issue concurrent `batchRead`/`batchWrite` calls, saturating the cluster's
  aggregate bandwidth across chains.

### File inventory

| File | Responsibility |
|------|----------------|
| `hf3fs_connector.h/.cc` | Connector core: client initialization, worker loops, batch/single read/write, close/cleanup ordering |
| `data_store.h/.cc` | Single-file + offset management, metadata persistence, lease, free-list, physical release |
| `connector_base.h` | Local fork of the shared `ConnectorBase<T>` (fork rationale in the file header) |
| `pybind.cpp` | Python bindings (module `lmcache.lmcache_raw3fs`) |

### connector_base.h fork deltas

The shared base class is depended on by redis/aerospike and other backends,
so the following generic changes live only in the fork (the file header has
the complete list; use it to reconcile when merging upstream):

- Uniform tiling (ceil-based split) instead of balanced-remainder;
- `per_key_results` preallocated so derived connectors can record per-key
  success/failure;
- **Per-key GET/SET failures propagate to the batch success bit** (a failed
  SET must not masquerade as success, or the upper layer silently loses
  data);
- Batches accept a single contiguous buffer (`bufs.size() == 1`);
- `LMCACHE_DBG=1` debug prints.

---

## 3. Core mechanisms and design rationale

### 3.1 Single file + offsets (O(N) → O(1) metadata)

One `data_${chunk_size}_store` per file index, created/opened once through
the MetaClient at startup. Placement is tracked in an in-memory
`unordered_map<key, (offset, size)>`, so batch-operation metadata is resolved
entirely in memory. Compared to one-file-per-key: MetaClient RPCs drop from
O(N) to O(1), no inode bloat, and stores skip the per-key file-chain
allocation and close-persistence.

### 3.2 Whole-chunk allocation and chunk_size selection

Each key occupies `ceil(len/chunk_size)` **whole, aligned chunks**; no chunk
ever mixes data from two keys. A chunk is 3FS's unit of placement and chain
routing (stripe = all chains), so chunk_size is a parallelism knob:

**Selection guidance (grounded in measurements)**:
- **cs ≤ typical key size, and ideally divides it evenly.** With 32MB keys:
  cs=32M → 1018 MB/s, cs=4M → 2249 MB/s (**2.2x**), cs=1M → 2274 (no further
  gain and metadata capacity shrinks 4x) — a key spanning k chunks gets
  write parallelism across k chains;
- Small keys (≤ cs) are insensitive to cs (single chunk; tail waste is
  equivalent to the old stride semantics);
- cs is unrelated to RDMA wire-level framing (max_sge / post batch / MTU are
  network-layer constants); it only affects the per-RPC payload cap and chain
  fan-out at the storage semantic layer;
- **Default 4MB**: a vLLM 32MB object divides evenly into 8 chunks — zero
  waste, 8-way chain parallelism.

### 3.3 Highly concurrent cross-chain batching (RDMA saturation)

Measured: `max_concurrent_requests_per_server` is the key lever — raising it
from the 3FS default 8 to 64 moved Load from ~4.8 to ~7.2 GB/s (with large
batches, each storage node was being starved). Current compiled defaults:
transport 8 / max_concurrent 256 / per_server 64 / max_batch_bytes 32MB
(128MB measured no difference). This pairs with an RDMA IOBuffer reuse cache
(avoiding repeated alloc_rdma + registerIOBuffer) and parallel memcpy for
large keys.

### 3.4 Metadata persistence (dynamically growing multi-chunk snapshot)

The key→offset map is serialized into chunk 0 of the data file itself;
`load_meta` replays it after a process restart, so offset semantics survive
across processes.

**Chunk 0 layout** (`META_HEADER_SIZE = 4197`):

```
magic(4) ver(1) chunk_size(4) total_entries(4) next_offset(8)
file_size(8) n0(4) nmeta(4) lease(64, @offset 37)
meta_chunk_list u32[1024] (@offset 101)      entries(28 B each)
```

- Fixed-size 28B entries (folded key hi 8 + lo 8 + offset 8 + size 4); any
  wire-format key string (e.g. `model@rank@group@hash[@salt]`) is folded
  deterministically by `DataStore::fold_key` (SpookyHashV2, pinned fixed
  seeds) into the canonical `"<hi>:<lo>"` form before **every** map
  operation, so lookup/store/delete agree on one key identity and the
  snapshot round-trips across restarts (`META_VERSION` = 5; changing the
  hash or seeds requires a version bump, which discards older snapshots);
- Chunk 0 capacity = (cs − 4197) / 28 — **149,646 entries** at cs=4M;
  overflow goes to **extension meta chunks** allocated on demand from the
  bump/free-list (tracked in `meta_chunk_list`, capped at 1024 chunks →
  total capacity ~149M entries; 128TB space cap at cs=4M);
- **Full-snapshot semantics**: `save_meta` rewrites the whole table every
  time (single registered buffer, chain-grouped batchWrite, parallel
  collectAllRange), so crash recovery needs no journal/replay log — the cost
  is write amplification linear in the total key count, an explicitly
  accepted trade-off (simplicity-first ruling); the meta_saver background
  thread persists every 5s and refreshes the lease heartbeat;
- Validation: magic + version + chunk_size are each verified; a mismatch
  discards the whole table (defense in depth; in practice the
  `data_${CS}_store` naming already isolates different cs values — changing
  cs means a new file, and old files are left for ops cleanup);
- A failed load starts with an empty table, equivalent to a legacy new file.

**Verified**: 151,000-key smoke run — chunk 0 filled to its exact capacity of
149,646 with 1,354 entries overflowing into an extension chunk; after
restart, replay hit 151,000/151,000, and byte-level spot checks across 200
keys all matched.

### 3.5 Multi-instance protection (owner lease)

Two instances sharing one `data_${CS}_store` silently destroy each other
(bump-pointer allocation hands out the same chunk → cross-key corruption;
a full meta overwrite drops the other instance's entire key set), while
sharing has zero read benefit (the offset map is purely in memory) — the
only legitimate pattern is sequential hand-off.

A fixed 64B lease region in chunk 0 (random_device 128-bit owner id +
hostname + heartbeat timestamp) provides mutual exclusion:

- On init, the lease is read: **if a foreign owner is present with a
  heartbeat < 60s (TTL), startup is refused** — the connector fail-fasts
  (all ops error out) with a loud ERROR suggesting a separate `base_dir`,
  waiting out the TTL, or deleting the file;
- The owner refreshes the heartbeat every 5s; `close()` ordering =
  stop_meta_saver (which includes a final flush) → release_lease (zeroed) →
  the next instance takes over immediately (verified: 16 sequential
  create/close cycles, zero wait);
- Crash residue expires via the 60s TTL.

### 3.6 Physical deletion + free-list

`delete()` does more than remove the map entry: each chunk of the deleted
key is physically punched (`createRemoveOp` per chunk, same as the FUSE
PUNCH_HOLE pattern), releasing server-side space.

Ordering invariant: `remove_offset` only moves chunks into
`pending_release_`; the background thread moves them into `free_ranges_`
(sorted + coalesced) only after the batched remove succeeds — this prevents
"chunk reused, then mistakenly deleted". Allocation is **best-fit over
contiguous ranges first, bump fallback**: a key's chunks must be a
contiguous index range (the write path walks chunkIds by increasing offset,
and the 28B entry only stores offset+size); contiguity also naturally
guarantees no repeated chunk→chain mapping (`idx % stripe`).

### 3.7 Lazy initialization on the read path

Store creation/open (including lease validation + load_meta) is triggered by
the **first operation of any kind** (SET/GET/EXISTS/DELETE all work) — so a
restarted instance whose first op is a LOOKUP/GET still hits keys persisted
by the previous process instead of missing everything for lack of a prior
SET.

### 3.8 Token authentication (optional)

The `token` constructor argument is injected into
`UserInfo{uid, gid, token}` and carried automatically on all RPCs;
`file://<path>` loads it from a file (mirroring fuse token.txt usage). The
current cluster's meta server runs with `authenticate=false` (the 3FS
default), so the token path is ready but dormant; re-verify end-to-end once
the server enables it.

---

## 4. Build and configuration

### 4.1 Prerequisite: build 3FS from source

**Check out source code.** Clone the 3FS repository from GitHub and check out
its submodules:

```bash
git clone https://github.com/deepseek-ai/3fs
cd 3fs
git submodule update --init --recursive
./patches/apply.sh
```

**Install dependencies:**

```bash
# for Ubuntu 20.04.
apt install cmake libuv1-dev liblz4-dev liblzma-dev libdouble-conversion-dev libdwarf-dev libunwind-dev \
  libaio-dev libgflags-dev libgoogle-glog-dev libgtest-dev libgmock-dev clang-format-14 clang-14 clang-tidy-14 lld-14 \
  libgoogle-perftools-dev google-perftools libssl-dev libclang-rt-14-dev gcc-10 g++-10 libboost1.71-all-dev build-essential

# for Ubuntu 22.04.
apt install cmake libuv1-dev liblz4-dev liblzma-dev libdouble-conversion-dev libdwarf-dev libunwind-dev \
  libaio-dev libgflags-dev libgoogle-glog-dev libgtest-dev libgmock-dev clang-format-14 clang-14 clang-tidy-14 lld-14 \
  libgoogle-perftools-dev google-perftools libssl-dev gcc-12 g++-12 libboost-all-dev build-essential

# for Ubuntu 24.04.
apt install cmake libuv1-dev liblz4-dev liblzma-dev libdouble-conversion-dev libdwarf-dev libunwind-dev \
  libaio-dev libgflags-dev libgoogle-glog-dev libgtest-dev libgmock-dev clang-format-14 clang-14 clang-tidy-14 lld-14 \
  libgoogle-perftools-dev google-perftools libssl-dev gcc-12 g++-12 libboost-all-dev build-essential
```

**Build 3FS** in the `build` folder:

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_C_COMPILER=clang-14 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j 32
```

Ubuntu 24.04 requires additional compiler flags:

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_C_COMPILER=clang-14 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_FLAGS="--gcc-toolchain=/usr -isystem /usr/include/c++/12 -isystem /usr/include/x86_64-linux-gnu/c++/12" \
  -DCMAKE_CXX_FLAGS="--gcc-toolchain=/usr -isystem /usr/include/c++/12 -isystem /usr/include/x86_64-linux-gnu/c++/12" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/12"
cmake --build build -j 32
```

### 4.2 Building the LMCache backend

```bash
HF3FS_3FS_DIR=/path/to/3FS BUILD_WITH_RAW_HF3FS=1 pip install -e . --no-build-isolation
```

- The 3FS source tree is referenced read-only (headers + prebuilt shared
  libraries); 3FS is not modified;
- Build entry: `setup_extensions/storage_backend_profiles/raw_hf3fs.py`;
- Fully independent of, and coexists with, the usrbio `BUILD_HF3FS` build;
- Artifact: `lmcache/lmcache_raw3fs.cpython-*.so`.

### 4.3 Runtime configuration (adapter JSON)

```bash
lmcache bench l2 --l2-adapter '{"type":"raw_hf3fs",
  "num_workers":4,
  "meta_addr":"RDMA://192.168.1.100:8000",
  "mgmtd_addr":"RDMA://192.168.1.100:8000"}'
```

| Key | Default | Notes |
|-----|---------|-------|
| `type` | — | `"raw_hf3fs"` (auto-discovered plugin type) |
| `num_workers` | 8 | C++ worker thread count |
| `chunk_size` | 4MB | Forced ≥4MB and a multiple of 4MB (`normalize_cs` clamps); should be ≤ typical key size and divide it evenly |
| `meta_addr` / `mgmtd_addr` | RDMA://192.168.1.100:8000 | 3FS meta / mgmtd addresses |
| `base_dir` | `"lmcache"` | Directory prefix on 3FS; **parallel instances must use distinct base_dirs** (see lease) |
| `cluster_id` / `chain_table_id` | `"stage"` / 1 | 3FS cluster identity |
| `transport_pool_max_connections` | 8 | Connections per IO worker |
| `max_batch_bytes` | 32MB | Per-batchRead/batchWrite RPC cap |
| `max_concurrent_requests` | 256 | StorageClient concurrency cap |
| `max_concurrent_requests_per_server` | 64 | Per-storage-node concurrency (**the key lever**; 0 = 3FS default 8) |
| `per_op_workers` | `{}` | Dedicated worker counts per op lane (lookup/retrieve/store/delete), isolating reads from writes |
| `max_capacity_gb` | 0 | Capacity tracking/eviction (0 = off) |
| `token` | `""` | 3FS auth token; supports `file://<path>` |
| `config_file` | `$LMCACHE_3FS_CONFIG` or `./lmcache_3fs.json` | Cluster config JSON; explicit keys override it |

---

## 5. Testing and evaluation

### 5.1 Integration tests (10 cases, `pytest -m integration`)

```bash
RUN_HF3FS_INTEGRATION=1 pytest -m integration \
  tests/v1/distributed/test_hf3fs_l2_adapter_v2.py -v
```

Coverage: basic round-trip (tensor), delete-then-recreate, multi-key,
nonexistent key, **restart persistence** (a new instance GETs a key stored by
the previous process; also for adapter-serialized `@`-format wire keys,
including a salted variant — guards the 128-bit fold), **deletes survive
restarts** (no resurrection), **lease rejects a second instance**,
**free-list reuse**, and **multi-chunk large-key round-trip**.

### 5.2 vLLM data integrity (16 cases)

```bash
python tests/v1/distributed/test_vllm_kv_cache_integrity.py
```

End-to-end pipeline `vLLM tensor → serialize → 3FS → deserialize →
byte-exact compare`, 4 shapes (llama8b MLA / llama70b bf16 / gemma9b / tiny,
8KB-2MB scale) × 4 modes (batch / single sequential / single concurrent /
batch of 64 concurrent), with both batch and single-key paths.

### 5.3 Performance benchmark protocol

```bash
# MUST clean before every run (bench itself does not clean; accumulated
# store degrades Store up to 5x). Either via a FUSE mount, if one exists:
rm -f /3fs/stage/lmcache/data_*_store
# ... or mount-free, through a short-lived connector:
python -c "import lmcache.lmcache_raw3fs as C; \
  c = C.HF3fsConnector(num_workers=1, meta_addr='RDMA://192.168.1.100:8000', \
  mgmtd_addr='RDMA://192.168.1.100:8000'); c.cleanup_data_files(); c.close()"

# 4MB-key protocol
lmcache bench l2 --data-size-kb 4096 --num-keys 320 --in-flight 16 \
  --rounds 2 --warmup-rounds 1 --lookup-max-hit-rate 0.0 \
  --l2-adapter '{"type":"raw_hf3fs","num_workers":4,"meta_addr":"RDMA://192.168.1.100:8000","mgmtd_addr":"RDMA://192.168.1.100:8000"}'

# 32MB-key protocol (num-keys 80: 320 would need a 160GB bench buffer,
# exhausting the 440GB host, triggering kswapd → spurious RDMA disconnects;
# such data is untrustworthy)
lmcache bench l2 --data-size-kb 32768 --num-keys 80 ... (otherwise identical)
```

**Interpretation discipline**:
- **The write path is sensitive to cluster state**: accumulated server-side
  state degrades Store by ~50% (daily band 1904-2249 vs freshly-restarted
  4178-4217, same protocol); comparisons must pin the server state (freshly
  restarted = upper bound, long-running = daily band); increasing latency
  across rounds within a run is the typical fingerprint;
- The read path is decoupled from cluster state (6.1-7.2 GB/s constant);
- The Store ceiling of ~4.1-4.2 GB/s is a server-side write aggregation
  bottleneck (unchanged at 8 workers), while Load climbs gently with worker
  count (4→8: 6.4→7.15 GB/s, 57% of line rate).

### 5.4 Large-scale smoke test (metadata extension path)

The dynamic metadata growth beyond chunk-0 capacity (at cs=4M, >149,646
keys) is out of reach of the integration tests and needs a manual smoke run:
store 151K small keys → restart → EXISTS hits all → byte-level spot checks →
no resurrection across three restarts after deletion (verified this round).

---

## 6. Known limitations and caveats

- **Multiple instances**: the lease guarantees mutual exclusion with
  sequential-hand-off semantics; parallel instances must each use a distinct
  `base_dir`;
- **Changing chunk_size**: the cs is embedded in the store file name
  (`data_${CS}_store`), so changing cs transparently switches to a new file
  and never misreads the old layout's metadata; old-cs files become orphans
  for ops cleanup (the magic+cs checks in the metadata remain as defense in
  depth);
- **Meta is not self-describing**: keys are persisted only as their 128-bit
  deterministic fold, so the original key strings cannot be recovered from
  the snapshot — no offline listing, audit, or migration by meta alone. A
  fold collision would alias two keys; birthday-bound probability is ~10⁻²¹
  at 10⁹ keys, and keys are always rewritten by the same fold that read
  them, so aliasing never happens at feasible key counts;
- **Exit safety**: normal runs shut down cleanly (bench and the integration
  suite exit with rc=0). The historical exit-time segfault was root-caused to
  3FS INFO-level folly logging corrupting the heap during interpreter teardown
  (ASan bisect) and fixed by holding the 3FS log level at ERR; opting into
  deep-debug logging (`LMCACHE_DBG=1` or `LMCACHE_3FS_LOGLEVEL=INFO/WARN`)
  may still abort at exit, after all results have been printed;
- **close() contract**: in-flight completions must be drained before
  `close()` (all in-tree callers do); closing with undispatched async
  operations races teardown in the worker threads and can segfault;
- **3FS session mismatch** (ExtendClientSessionMismatch): a known 3FS
  server-side issue when a stale session exists; the client has a
  reset_connector fallback;
- meta_saver snapshot write amplification is linear in the total key count
  (currently accepted; >149K keys means every full snapshot starts at
  ~4.2MB).
