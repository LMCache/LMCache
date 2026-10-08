# Valkey RDMA L2 Adapter Design

This document describes the built-in `valkey_rdma` L2 adapter for LMCache
multiprocess mode. The adapter stores KV cache chunks in a Valkey instance
running the [valkey-large-object][module] module and moves every chunk by RDMA:
the server reads a chunk out of, or writes it into, the L1 arena itself. It
is the [`valkey` adapter](valkey.md) with the transport replaced, and uses
the RDMA API of the official `valkey-glide` Python sync client.

Read the [`valkey` design doc](valkey.md) first. Everything there that is not
about how bytes move — threading model, batch state, partial-failure
accounting, cluster vs standalone, authentication, capacity and eviction,
lock semantics — applies to this adapter unchanged.

[module]: https://github.com/valkey-io/valkey-large-object

## Goals

- Move multi-megabyte KV chunks with **no copy on either end**: no RESP
  payload, no glide-core buffer, no Python staging buffer.
- Change nothing above the transport. To the L2 controllers this is the
  `valkey` adapter: same submit/event-fd/query contract, same wire keys,
  same eviction accounting.
- Fail at construction, with a message naming the missing piece, on a
  machine that cannot do RDMA.
- Never let a stored value larger than its destination chunk touch memory
  outside that chunk.

## Why a subclass of the `valkey` adapter

The `valkey` adapter already has the shape RDMA needs: N worker threads,
each holding its own sync glide client, with every operation expressed
through the worker pool's four `submit_*` methods. Two facts of glide's
RDMA API make that per-thread client the natural unit of registration:

- An `RdmaRegion` is bound to the client that registered it.
- A region carries **one transfer at a time**.

So one registration per worker is what lets `num_workers` transfers run
concurrently, and the pool is the only layer that needs to change. The
adapter subclass overrides pool construction and `report_status()` and
nothing else.

## Dependencies

- `valkey-glide-sync >= 2.6.0` built with RDMA support. Published wheels
  include it; a build from source needs `GLIDE_SYNC_RDMA=1`. The RDMA API
  (`RdmaConfiguration`, `register_rdma_region`, `rdma_get`, `rdma_set`)
  lives on the sync client only.
- **libfabric** on the machine, loaded by glide at runtime. On AWS that is
  the EFA installer's libfabric; for development the tcp provider of any
  libfabric `>= 2.4` works.
- A Valkey node with the **valkey-large-object module** loaded and a fabric
  provider configured (`fabric-provider EfaDirect` or `Emulated`).

All three are checked when the first worker builds its client, and a
missing one fails construction:

```
The installed valkey-glide-sync was built without RDMA support. ...
valkey-glide-sync has RDMA support but libfabric is not available on this machine. ...
```

The import is lazy, as in the `valkey` adapter, so installations that do
not use this adapter never need the dependency.

## Components

```text
ValkeyL2Adapter                        ValkeyWorkerPool
  │ _create_pool(config)                 │ _client_config_extras(glide_sync)
  │ _pool_kwargs(config)                 │ _do_set / _do_get_into
  ▼                                      ▼
ValkeyRdmaL2Adapter  ── builds ──►  ValkeyRdmaWorkerPool
  report_status()                      rdma=RdmaConfiguration(...)
                                       one RdmaRegion per worker thread
                                       BLOB.SET / BLOB.GET windows
```

| File | Owns |
|------|------|
| `lmcache/v1/distributed/l2_adapters/valkey_rdma_l2_adapter.py` | `ValkeyRdmaL2AdapterConfig`, `ValkeyRdmaL2Adapter`, registration under `valkey_rdma` |
| `lmcache/v1/storage_backend/valkey/rdma_worker_pool.py` | `ValkeyRdmaWorkerPool`: client configuration, arena registration, the two transfers, region teardown |

The two parent hooks were added for this adapter: `_create_pool` /
`_pool_kwargs` on the adapter, and `_client_config_extras` on the pool.
Each defaults to the parent's existing behavior.

## L1 arena registration

The factory receives the storage manager's `L1MemoryDesc` (arena base
pointer, size, alignment) and refuses to build without one, or with an
empty one — the same rule the Mooncake and NIXL adapters apply to their
RDMA paths.

```text
factory(config, l1_memory_desc)
  -> ValkeyRdmaWorkerPool(l1_base, l1_size, ...)
       -> warm-up: each worker thread
            -> GlideClient.create(rdma=RdmaConfiguration(provider))
            -> client.register_rdma_region(<ctypes view of the arena>)
```

The arena must be fully allocated before that: run the MP server with
`--no-l1-use-lazy`, since the lazy L1 allocator only reserves the range and
populates it in steps, and registering unpopulated pages is untested (the
L1 manager carries a TODO to that effect).

Each worker registers the **whole arena, once**, with its own client's
fabric. The registration is a `ctypes` overlay over the descriptor's
pointer that satisfies the buffer protocol; nothing is copied or owned. A
worker thread that was not exercised during warm-up registers on its first
operation, exactly as it builds its client.

The arena is pinned once per worker. In time that is cheap: on a
c8gn.16xlarge, registering a populated 32 GiB arena takes about 50 ms with
transparent huge pages (which torch's large allocations get by default) and
about 0.5 s with 4 KiB pages, so eight workers cost 60 ms or 1.5 s at
start-up, and the kernel pins shared pages by reference rather than copying
them.

In device capacity it is the one hard constraint of this design: the EFA
device caps **total** registered memory at its `max_mr_size`, and every
worker's registration counts against it. Measured: 384 GiB on c8gn.16xlarge
(8 × 32 GiB fits), 96 GiB on g6.8xlarge (8 × 16 GiB does not; the sixth
16 GiB registration fails with ENOMEM). libfabric reports the overflow only
as `fi_mr_reg: Cannot allocate memory`, so the pool wraps a failed
registration in a `RuntimeError` that states `num_workers × arena` and what
to shrink. `num_workers × l1_size_gb` must be sized against the device.

## Window derivation

`ValkeyL2Adapter` hands the pool `obj.byte_array`, a `memoryview` of the
object's logical bytes at `obj.data_ptr`, for both SET and GET. The RDMA
pool locates that view inside the arena:

```text
offset = address(view) - l1_base
window = region.window(offset, len(view))
```

Deriving the window from the buffer rather than from the `MemoryObj` keeps
the pool's submit API identical to the plain pool's, which is what lets the
adapter subclass override nothing per operation.

A buffer outside `[l1_base, l1_base + l1_size)` has no registration to
transfer through. The pool raises, the adapter records a per-key failure
(failed store, or load miss) and logs it. There is deliberately **no
fallback path**: in MP mode the storage manager allocates every object
from the arena, so an out-of-arena buffer is a bug to surface, not a case
to serve.

## Transfers

| Operation | Command sent | Server action | Reply |
|-----------|--------------|---------------|-------|
| store | `BLOB.SET key len rkey addr len` | `fi_read`s `len` bytes out of the window, stores them | `OK` |
| store, with `ttl_seconds` | `EXPIRE key ttl` after the above | sets the key's TTL | `1` |
| load | `BLOB.GET key rkey addr len` | refuses if the value is larger than `len`; else `fi_write`s it into the window | `[bytes, crc32c]` |
| lookup | `EXISTS key` (inherited) | native command, any key type | count |
| delete | `DEL key` (inherited) | the module's free callback releases the object | count |

The keys are the module's own data type. A plain `GET` or `SET` cannot read
or replace them, so a node holds either adapter's keys under a given
`key_prefix`, never both.

`ttl_seconds` costs one extra round trip per store, because `BLOB.SET` has no
expiry argument.

## Size validation on load

`BLOB.GET` carries the window's length, so a stored value **larger** than the
destination chunk is refused by the server before any write is posted. This
matters more here than in the `valkey` adapter: there, an oversized value
lands in glide's scratch buffer and is dropped; here, every worker's region
is the whole arena, so an unchecked write would run past the chunk into the
L1 slab that follows it. The client can only detect that afterwards, from
the receipt's byte count, which is too late.

A **smaller** value lands and is reported as a miss, because fixed-size
chunks must round-trip exactly — the same rule `_do_get_into` applies in the
plain pool.

glide verifies the receipt's `crc32c` against the landed bytes before
returning, so a load that reports a hit has also been checksummed.

## Error model

| Failure | Per-key bit | Logged | Notes |
|---------|-------------|--------|-------|
| Buffer outside the arena | `False` | `WARNING` | No transfer attempted. Other keys in the batch unaffected. |
| Value larger than the window | `False` | `WARNING` | Refused by the server; nothing written. |
| Value smaller than the window | `False` | `WARNING` | Landed, but rejected as a miss. |
| Server error reply (`ERR ...`) | `False` | `WARNING` | Region stays usable. |
| Transfer fails without a reply (connection drop, fabric error) | `False` | `WARNING` | glide **revokes** the region; see below. |
| Next transfer through a revoked region | retried once | `WARNING` | Worker deregisters, registers the arena again, retries. |
| Client closed mid-transfer | `False` | `WARNING` | `ClosingError`; only during `close()`. |
| Missing RDMA support / libfabric / module | construction fails | (raises) | Actionable message naming the piece. |

**Revoked regions.** glide revokes a region when a transfer fails for any
reason other than a reply from the server, because the server may still be
using the memory. The next transfer through that region is refused as
revoked, and only then does the worker register the arena again and retry
the operation once. A server's own error reply leaves the region usable, so
it never costs a registration.

**Cancellation.** glide applies no timeout to a transfer, because the
server's remote write cannot be called off once posted; `request_timeout`
still bounds every other command. Closing the client cancels every transfer
in flight, so `close()` closes each worker's client first and its region
second, which is the order glide requires.

**Direct writes into L1.** The plain pool stages a GET through a scratch
buffer because glide writes with the GIL released and a slab could be
recycled mid-write (issue #6215). RDMA writes into the slab by design, so
this adapter relies on the locks the controllers hold on an object for the
duration of a load — the same assumption the NIXL and Mooncake adapters
make.

## Configuration

JSON schema (CLI `--l2-adapter`); every `valkey` field is accepted, plus:

```json
{
  "type": "valkey_rdma",
  "startup_nodes": "kv.internal:6379",
  "num_workers": 8,
  "ttl_seconds": 3600,
  "rdma_provider": "efa-direct",
  "rdma_interface": "efa0"
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `rdma_provider` | `"efa-direct"` | `"efa-direct"` opens EFA hardware; `"tcp"` opens libfabric's software provider, for development and tests only. |
| `rdma_interface` | unset | Fabric domain to pin to on a host with more than one card. |

## Non-goals

- **A TCP fallback for out-of-arena buffers**: nothing legitimate lives
  there in MP mode, and a copy path nothing exercises would rot.
- **Per-chunk registration**: one registration per operation would let the
  fabric reject an overrun, but costs a registration per transfer; the
  `BLOB.GET` length gives the same guarantee for one integer on the wire.
- **Interoperating with `valkey`-adapter keys**: the module's data type and
  a plain string are different key types; use a different `key_prefix`.
- **Serde with zero copy**: the `serde` wrapper still works but stages each
  chunk through a second L1 buffer and a transform pass.
- **Replica reads**: transfers always run against a primary, as in glide.

## Testing

- `tests/v1/distributed/test_valkey_rdma_l2_adapter.py` — unit tests
  against an in-process fake of glide's RDMA API. The fake records every
  registration and every transfer window, so the tests assert that each
  worker registers the whole arena, that each window is the object's own
  offset and size, and that bytes land in and are read from the arena
  tensor directly.
- `tests/v1/distributed/test_valkey_rdma_l2_adapter_integration.py` —
  real transfers against a valkey-server running the module with the
  `Emulated` fabric provider and `valkey-glide-sync` built with RDMA, using
  the `tcp` provider. Skipped unless `LMCACHE_VALKEY_RDMA_SERVER` names a
  server.
