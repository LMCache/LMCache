# Shared-Pool CXL Borrowing Peer Adapter

`lmcache/v1/distributed/l2_adapters/cxl_peer_l2_adapter.py`,
`BorrowingPeerL2AdapterBase` in `lmcache/v1/distributed/l2_adapters/p2p_l2_adapter.py`,
`CxlPeerMapping` in `lmcache/v1/memory_allocators/devdax_memory_allocator.py`

**Experimental**: validated against a file-backed mock pool, not real
multi-host CXL.

Nodes that map one shared CXL pool each own a disjoint slab of it (their
Device-DAX L1). The CXL peer adapter **borrows** a peer's object, as the GDR
adapter does ([`gdr_peer_l2_adapter.md`](gdr_peer_l2_adapter.md)): the owner
keeps its read lock and a shadow L1 indexes the object. The difference is the
bytes: this host maps the owner's slab, so a borrowed object is a view of the
owner's bytes and no transfer happens between hosts.

```
--p2p-load-mode gdr (borrow over RDMA)              --p2p-transfer-engine cxl (borrow in place)
lookup:   owner read-locks; shadow indexes a handle lookup:   owner read-locks; shadow indexes a view
retrieve: owner DRAM ─RDMA─► GPU slot ─► paged KV    retrieve: owner slab (mapped here) ─PCIe─► GPU
unlock:   after the last local reader               unlock:   after the last local reader
```

Enable with `--p2p-transfer-engine cxl --cxl-pool-id <id>` on a Device-DAX L1
(`--l1-devdax-path`, `--l1-devdax-offset-gb` placing this node's slab) and a
coordinator; no advertise URL. The server adds the CXL shadow L1
(`shadow.cxl`, [`../l1_manager/l1-manager-types.md`](../l1_manager/l1-manager-types.md))
when none is configured. `cxl` excludes `--enable-p2p-write` and
`--p2p-load-mode gdr`, and requires the slab to be the only L1 that owns
memory (peers can read nothing else).

## Classes

`CxlPeerL2Adapter` extends `BorrowingPeerL2AdapterBase`; reservations, the
load that binds the shadow L1's placeholders, the lease TTL and the unlocks
are the base's and are described in the GDR doc. This class adds only:

| Hook | What it does |
| --- | --- |
| `_register_peer_region` | `CxlPeerMapping`: map the peer's slab from the local device path, check the identity header, GPU-register (pin) the mapping. |
| `_process_lookup_result` | Refuse a hit whose `cxl_arena` is not the mapped incarnation of the peer's slab and return its owner lock at once; the rest become reservations. |
| `_bind_borrowed_object` | `obj.bind(lease, local_view=mapping.view(offset, size), ...)`. The lease additionally requires the mapped header to still be current. |
| `_unregister_peer_region` | Return unborrowed reservations, then unpin and unmap. Raises while a lease is live (`RuntimeError`) or a reader still references a view (`BufferError`); the P2P controller retries the removal every discovery cycle. |

`read_window_s` is 0: a read from the mapping is a host/GPU copy queued at
retrieve, and the base's TTL margin (`min(1 s, TTL / 10)`) is its time to
finish. As with GDR, **every node must use the same
`--l1-read-ttl-seconds`**; the borrower bounds a lease with its own.

## Local views

`BorrowedMemoryObj.bind` takes exactly one of `read_into` (GDR: bytes are
read into a device region at retrieve) and `local_view` (CXL: a `uint8`
tensor over the owner's bytes mapped into this process). With a local view:

- `raw_tensor`, `tensor`, `get_tensor`, `byte_array` and `data_ptr` return the
  view, so the object is a host-memory object to every copy path.
- `is_local` is True and `select_engine` picks `GPU_DMA`, not `NIC_DMA`: the
  direct copy, native plan and staged loop all apply, and no staging buffer
  is registered with a transfer channel.
- `release()` drops the view, so the mapping can be closed once no reader
  holds a tensor of it.

## Slab identity and addressing

`CxlArenaDescriptor` (`lmcache/v1/common/transfer_channel/api.py`) holds
`pool_id`, `offset` (slab header offset in the device), `size` (payload
bytes), `alignment` (also the header length) and `session_id` (fresh each
time the owner opens the slab). The owner:

- writes the descriptor's SHA-256 fingerprint into the slab header at startup
  and zeroes it on close;
- advertises the descriptor as JSON under coordinator metadata
  `lmcache.cxl.arena` (`CXL_METADATA_KEY`);
- answers lookups with `TransferChannelAddress(offset, size, cxl_arena)`,
  `offset` relative to the slab payload. Its virtual address never crosses
  the wire:

```text
local_pointer = local_mapping_of_peer_slab + peer.alignment + address.offset
```

The borrower accepts a peer in the same pool with a disjoint slab, checks the
mapped fingerprint before serving it, and re-checks it on every lease
validity check. A restarted owner therefore invalidates every view of its
previous incarnation at once and discovery replaces the adapter (the
descriptor changed). The fingerprint detects wrong mappings and restarts; it
is not authentication.

Only the owned slab of the one CXL L1 is exported: the P2P controller serves
`(CXL, LOCAL)` under this engine, so a borrowed view (`(CXL, SHADOW)`) is
never re-exported.

## Flow

```mermaid
sequenceDiagram
    participant PC  as A PrefetchController
    participant CXL as A CxlPeerL2Adapter
    participant OWN as B P2PController
    participant L1  as A Shadow L1 (CXL)
    participant GPU as A GPU

    PC  ->> CXL: submit_lookup_and_lock_task(keys)
    CXL ->> OWN: p2p_lookup_and_lock(keys)
    Note over OWN: read-lock found keys in the CXL L1
    PC  ->> CXL: query_lookup_and_lock_result(task_id)
    OWN -->> CXL: slab-relative addresses + arena
    Note over CXL: one reservation per hit in the mapped slab
    CXL -->> PC: found_bitmap
    PC  ->> L1:  reserve_write(found_keys)
    L1 -->> PC:  BorrowedMemoryObj placeholders  [no bytes]
    PC  ->> CXL: submit_load_task(found_keys, placeholders)
    Note over CXL: bind each placeholder to its lease and a view of B's slab
    PC  ->> L1:  finish_write_and_reserve_read(keys)
    Note over PC,GPU: Retrieve: host-memory run, GPU DMA engine
    GPU ->> L1:  read B's bytes through A's mapping
    PC  ->> L1:  finish_read(keys)
    Note over L1: last reader frees the object, its lease ends
    L1  ->> CXL: lease on_release
    CXL ->> OWN: p2p_unlock_objects(keys)
    PC  ->> CXL: submit_unlock(keys) for hits it did not borrow
```

## Deployment

Reserve `alignment + payload_size` bytes per slab: two 2 GiB payloads with
2 MiB headers start at 0 and 2149580800 bytes (`--l1-devdax-offset-gb
2.001953125`) and need 4299161600 shared bytes. `--l1-align-bytes` is the
header length and must be a multiple of the device alignment (2 MiB on most
Device-DAX devices). Slabs must not overlap, and a slab must not be
reinitialized under active readers. Device paths may differ between nodes.
User guide: [P2P](../../../../source/mp/p2p.rst).

## Known limitations

- No fencing: an owner that dies during a GPU copy is not detected until the
  header check of the next lease validity test.
- A view released after its owner restarted still sends its (key-only)
  unlock to the new incarnation, as GDR does; it is harmless unless another
  borrower locked the same key there within the TTL.
- A node with a shorter read TTL than its borrowers lets them read reused
  bytes; nothing checks TTL equality (shared with GDR).
- Cross-host CXL coherency, DMA performance and failure recovery still need
  hardware validation.
