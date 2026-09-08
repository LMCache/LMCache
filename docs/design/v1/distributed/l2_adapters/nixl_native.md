# Native NIXL L2 Adapter Design

## Purpose

`nixl_native` is a built-in C++ implementation of the MP
`L2AdapterInterface`. It is parallel to `fs_native`: both use
`NativeConnectorL2Adapter` for task IDs, event notification, locking, byte
accounting, and completion demultiplexing. The difference is the data path:
`fs_native` performs system calls directly, while `nixl_native` expresses each
tile as a NIXL descriptor-list transfer.

The native connector is separate from the Python `nixl_store` adapters so it
can register the exact L1 arena once, avoid Python work in the transfer path,
and share one connector runtime across FILE and OBJECT storage semantics. It
does not change the existing adapters or their persistent formats.

## Component boundary

```text
L2AdapterInterface
  -> NativeConnectorL2Adapter
       -> LMCacheNixlClient (pybind)
            -> NixlConnector / ConnectorBase worker pool
                 -> NixlStorageStrategy
                      -> NixlFileStorage   (FILE_SEG)
                      -> NixlObjectStorage (OBJ_SEG)
```

`NixlConnector` owns request validation, workers, NIXL agents, L1
registrations, transfers, and LMCache completions. A strategy owns persistent
identity mapping, storage descriptor construction, query semantics,
publication/rollback, and deletion capability. Connector code never opens a
file or interprets an object key.

Plugin discovery is the sole strategy selector. Each worker requires
`DRAM_SEG` and exactly one of `FILE_SEG` or `OBJ_SEG`; the advertised storage
segment creates `NixlFileStorage` or `NixlObjectStorage`, respectively. A
backend advertising neither or both is rejected, so configuration cannot
contradict the backend's actual capabilities.

## Worker and registration lifetime

The initial design uses one `nixlAgent`, backend handle, and complete L1
registration per worker. NIXL 1.3 does not document a general contract for
driving one agent concurrently from unrelated application threads, so worker
ownership provides a simple synchronization boundary.

Construction prepares every worker context before starting `ConnectorBase`
threads. If agent creation, plugin discovery, backend creation, or L1
registration fails, ordinary C++ ownership unwinds all earlier contexts and no
worker exists. Each worker then takes one prepared context. Shutdown is
idempotent: `ConnectorBase` stops and joins workers before their contexts,
registrations, agents, and backend handles are destroyed.

The registered descriptor is exactly
`[l1_base, l1_base + l1_size)`. Before a request reaches NIXL, the connector
checks, without overflowing, that every nonempty buffer lies inside that range.
Arena containment is the connector's only buffer validation. The filesystem direct-I/O alignment is evaluated per buffer by
the FILE strategy (see "Direct-I/O eligibility and buffered fallback" below).
Arbitrary process memory is never registered on demand.

When direct I/O is in effect (FILE storage with `use_direct_io: "true"`, as
reported by the connector's `supports_direct_io` capability), the Python
adapter automatically submits each object's full physical L1 slot — logical
bytes plus the allocator's alignment padding up to `--l1-align-bytes` —
instead of just the logical bytes. This padding is an adapter-level
optimization, not a connector validation requirement: a padded slot keeps both
its address and its length as alignment multiples, so it stays eligible for
direct I/O. The padding is part of the object's own allocation, so the range
still lies inside the registered arena; the connector needs no change because
it transfers exactly the pointer/length pair it is given. Buffered transfers
(direct I/O off) and OBJECT storage are never padded.

## Batch transfer lifecycle

One `ConnectorBase` tile becomes one local descriptor list, one storage
descriptor list, and one NIXL transfer request. A normal tile is not reduced to
one request per key.

```text
validate buffers
  -> create/register storage descriptors
  -> prepare local and storage descriptor lists
  -> create and post one transfer request
  -> progress to a terminal status
  -> release request and prepared-list handles
  -> deregister per-operation storage
  -> publish or report per-key load results
```

RAII objects retain storage registrations, prepared-list handles, transfer
handles, and unpublished temporary paths through completion. NIXL owns any
file descriptors opened by a path-mode registration and closes them when the
registration is deregistered. An exception unwinds these resources in reverse
order. FILE loads do not preflight individual paths: the caller queries first,
supplies the correct destination lengths, and the complete tile succeeds or
fails as one batch.

## FILE strategy

`NixlFileStorage` is the reference `FILE_SEG` strategy, verified with the NIXL
POSIX plugin.

FILE transfers use NIXL path-mode registration. Every registration descriptor
has address zero, the buffer length, and a process-wide, thread-safe synthetic
`devId` that is unique for its registration lifetime. The matching transfer
descriptor has the same fields. Distinct IDs are required because a NIXL file
backend rejects simultaneously registered paths that reuse one `devId`.

Store deliberately performs no existence query. For each key it chooses an
unused temporary path in the final directory and registers this metadata:

```text
rw,create,sync:/data/lmcache/l2/model@0x00000000@0@00000001.data.tmp.42
rw,create,sync,direct:/data/lmcache/l2/model@0x00000000@0@00000002.data.tmp.43
```

NIXL opens each path during `registerMem`, owns the descriptor for the batched
WRITE, and closes it during `deregisterMem`. The `sync` flag keeps the durable
store-completion boundary inside NIXL ownership; LMCache neither reopens the
file nor calls `fsync` on a NIXL-owned descriptor. Transfer requests and
prepared descriptor lists are released first, then the storage registration is
deregistered, and only then does LMCache publish the completed temporary
paths. Publication uses a same-filesystem hard-link operation, which atomically
fails if the final name already exists; the temporary link is then removed.
This implements no-replace publication without exposing partial bytes.

The connector does not scan a batch for duplicate serialized keys. Such input
violates the upstream submission contract and must not be generated by callers.

If a racing writer already published the expected length, store accepts the
existing file. A different length is a collision error. If later publication
in the same batch fails, finals published by this batch are removed and RAII
removes all remaining temporary files. This policy assumes the L2 query path
was used before store, while still protecting against races between query and
publication. Temporary-name allocation and this race policy support concurrent
worker threads within one LMCache process. A `file_path` namespace must not be
written by multiple LMCache processes.

Lookup uses NIXL `queryMem` with full deterministic paths, without a path-mode
prefix. Load trusts that lookup has established existence and that the caller
provided the correct buffer length. It registers every final path as
`ro:<absolute-path>` in one batched READ; registration or transfer failure
fails the complete load batch. LMCache performs no `open`, existence check,
`fstat`, or size filtering before registration. Delete remains a filesystem
removal and is reported as supported.

### Direct-I/O eligibility and buffered fallback

Direct I/O is a strategy capability: `supports_direct_io` continues to mean
that aligned FILE transfers can use direct I/O. `use_direct_io: "true"`
requests direct I/O for eligible FILE buffers; it neither guarantees nor
requires it for every buffer.

Eligibility is decided per buffer before registration. The strategy reads the
target filesystem's direct-I/O alignment from `statvfs(file_path).f_bsize`,
and a buffer is eligible only when both its address and its byte length are
exact multiples of that alignment. Eligible buffers are registered with NIXL's
path-mode `direct` flag; an address- or length-misaligned buffer automatically
falls back to buffered I/O for that descriptor instead of raising an alignment
error. One batched request may therefore mix `direct` and buffered FILE
descriptors:

```text
rw,create,sync,direct:<tmp>    (eligible store buffer)
rw,create,sync:<tmp>           (misaligned store buffer, buffered fallback)
ro,direct:<path>               (eligible load buffer)
ro:<path>                      (misaligned load buffer, buffered fallback)
```

The fallback exists because unaligned buffers previously failed the operation;
selecting the I/O mode per descriptor preserves the direct-I/O benefit for
every eligible buffer instead of rejecting the whole batch. The alignment
checked here is a property of the target filesystem and is enforced by the
FILE strategy; it is distinct from the L1 allocator alignment
(`--l1-align-bytes`), which the allocator and adapter own and which the
connector never validates.

Fallback covers this deterministic preflight alignment decision only. NIXL
registration, `open()`, and transfer failures still fail the operation exactly
as before — there is no retry or fallback for them. The separate FS connector
(`csrc/storage_backends/fs/`) keeps its own behavior; no parity is claimed.

## OBJECT strategy

`NixlObjectStorage` is the reference `OBJ_SEG` strategy, verified at build time
with the NIXL 1.3 OBJ API and through a live S3-compatible round trip. It
registers one object descriptor per key and sends the entire selected tile in
one request.

Store is an unconditional whole-object operation. It does not query first and
does not attempt a conditional create because the NIXL 1.3 OBJ API exposes
neither condition. Lookup uses `queryMem`. Load first queries to build a
hit-only descriptor list; a transfer failure turns those selected keys into
misses without corrupting unrelated buffers.

NIXL 1.3 OBJ query returns existence but no content length. Exact remote length
therefore cannot be validated before its ranged GET. The adapter always uses
offset zero and the complete LMCache destination size, so it cannot request a
partial write.

NIXL 1.3 exposes no object-deletion operation. The strategy reports deletion
as unsupported, its delete entry point returns a clear error, status contains
`supports_delete: false`, and Python configuration rejects OBJECT eviction.

## Persistent identity

The native wire key is:

```text
model@kv-rank-8hex@object-group-hex@chunk-hash-hex[@cache-salt]
```

Persistent names use the shared native key serializer: replace model `/` with
`-SEP-`, separate every identity field with `@`, render the rank as
`0x` plus eight hex digits, append a nonempty salt as another field, then
append `.data`. Optional sharding prefixes the first four chunk-hash hex
digits as two directories.

```text
org/model@0000002a@7@00112233@tenant
  -> org-SEP-model@0x0000002a@7@00112233@tenant.data
  -> 00/11/org-SEP-model@0x0000002a@7@00112233@tenant.data  (sharded)
```

The mapping retains model, rank, object group, hash, and salt. It cannot escape
the configured namespace because model separators are removed and `ObjectKey`
rejects separators in salt. Names longer than the filesystem component limit
are rejected. Compatibility with `fs_native` is intentionally not provided;
that connector retains its established `.data` identity.

## Optional dependency boundary

`setup_extensions/storage_backend_profiles/nixl.py` discovers headers and
`libnixl` only for the isolated `lmcache.lmcache_nixl` C++20 extension.
Ordinary builds do not import NIXL or acquire a linker dependency. The Python
factory imports the extension only after `nixl_native` is selected and returns
an actionable build error when it is absent.

Runtime status exposes only the backend name, inferred storage type, workers,
FILE path, direct-I/O choice, and inferred strategy capabilities. The opaque
backend parameter map is never reported because it can contain credentials.

## Deferred changes

Sharing agents, registering out-of-arena buffers, registration caches, native
object deletion, and a custom asynchronous progress engine require separate
correctness tests and benchmark evidence. They are not part of the initial
lifetime model.
