# Transfer Channel Throughput Benchmark

Measures read and write throughput (GB/s) of the LMCache **transfer channel**
(`lmcache/v1/distributed/transfer_channel/`) for batched peer-to-peer
transfers.

Unlike a raw-tensor microbenchmark, this tool uses LMCache's `L1MemoryManager`
to initialize the registered memory region and to **allocate the transferred
objects** on both sides, so it exercises the same memory path production uses.

## How it works

The benchmark runs as **two separate processes**:

- **server** — uses an `L1MemoryManager` to allocate a registered L1 buffer and
  a pool of source memory objects, registers the buffer with the transfer
  channel, and publishes the source object catalog (`offset`/`size` per object)
  over a small ZMQ side-channel.
- **client** — fetches the catalog, allocates its own objects via an
  `L1MemoryManager`, connects the transfer channel, and repeatedly transfers a
  random `--num-objects` subset of the source objects, then reports throughput.
  `--op read` (default) pulls the server's objects into the client's buffer;
  `--op write` pushes the client's objects into the server's buffer.

A side-channel is needed because the transfer channel handshake only exchanges
the whole-buffer registration, not per-object offsets.

> Only the `nixl` transfer channel type is registered today. The tool is generic
> over `--transfer-channel-type`; an unknown type raises a clear error.

## Usage

Run via the module or the `lmcache` CLI. Start the server first.

### `python -m`

```bash
# terminal 1 — server
python -m lmcache.tools.transfer_channel_benchmark \
  --role server --transfer-channel-type nixl \
  --url 0.0.0.0:7600 --control-url 0.0.0.0:7610 \
  --buffer-size 2GB --page-size 512KB --object-size 10MB

# terminal 2 — client
python -m lmcache.tools.transfer_channel_benchmark \
  --role client --transfer-channel-type nixl \
  --url 127.0.0.1:7600 --control-url 127.0.0.1:7610 \
  --listen-url 0.0.0.0:7601 \
  --page-size 512KB --object-size 10MB \
  --num-objects 10 --iters 3 --warmup 1
```

### `lmcache tool`

```bash
lmcache tool transfer-channel-benchmark --role server  --url 0.0.0.0:7600 \
  --control-url 0.0.0.0:7610 --buffer-size 2GB --object-size 10MB
lmcache tool transfer-channel-benchmark --role client --url 127.0.0.1:7600 \
  --control-url 127.0.0.1:7610 --object-size 10MB --num-objects 10
```

The server always writes a deterministic per-object byte pattern (object index
mod 256) into its source objects, so adding `--verify` on the **client** checks
the transferred bytes against that pattern (no `--verify` needed on the server).

### Write mode

```bash
# terminal 2 — client pushes its objects into the server's buffer
python -m lmcache.tools.transfer_channel_benchmark \
  --role client --op write --verify \
  --url 127.0.0.1:7600 --control-url 127.0.0.1:7610 \
  --listen-url 0.0.0.0:7601 \
  --page-size 512KB --object-size 10MB \
  --num-objects 10 --iters 3 --warmup 1
```

Before every write iteration the client fills its objects with a pattern that
differs from the server's initial one (`(index + iteration + 1) mod 256`).
With `--verify`, after the last iteration the client sends the expected byte
per written object over the control side-channel and the **server** checks its
own buffer, so the verdict comes from the memory that was written, not from a
read-back. A server process can serve read and write clients in any order; a
later `--op read --verify` run fails for objects a write run has overwritten,
so restart the server between a write run and a verified read run.

## Key options

| Option | Role | Meaning |
| --- | --- | --- |
| `--role {server,client}` | both | Which side to run (required). |
| `--op {read,write}` | client | Pull the server's objects (read) or push into them (write). Default `read`. |
| `--transfer-channel-type` | both | Implementation to benchmark (default `nixl`). |
| `--nixl-backend` | both | nixl backend, e.g. `UCX` (nixl-specific). |
| `--url` | both | Server binds its transfer-channel server here; client dials it. |
| `--listen-url` | client | Client's own (mandatory) transfer-channel server bind. |
| `--control-url` | both | Catalog side-channel: server binds, client connects. |
| `--buffer-size` | server | Registered L1 source buffer size (e.g. `8GB`). |
| `--page-size` | both | Page / alignment size; **must match** on both sides. |
| `--object-size` | both | Per-object size; multiple of `--page-size`. |
| `--num-objects` | client | Objects transferred per iteration. |
| `--num-source-objects` | server | Source pool size (default `5 * --num-objects`). |
| `--iters` / `--warmup` | client | Measured / warmup iterations. |
| `--seed` | client | RNG seed for the transferred subset. |
| `--verify` | client | Verify transferred bytes: reads in the client's buffer, writes in the server's buffer. |
| `--use-lazy` | both | Use the lazy L1 allocator (experimental for registration). |
| `--server-timeout` | server | Seconds to serve catalog requests before exiting. |

## Notes

- `--page-size` must be identical on server and client (enforced via the catalog
  handshake) so remote page-index math lines up.
- With the default (non-lazy) allocator the whole `--buffer-size` is allocated up
  front and may be CUDA-pinned; keep it within available host memory.
- Requires a working transfer channel runtime (for `nixl`, a UCX backend).

## Performance: NUMA placement

On a multi-NUMA host with NICs spread across nodes (e.g. an 8-NIC, 2-socket
box), run **both** the server and client under `numactl --interleave=all`:

```bash
numactl --interleave=all \
python -m lmcache.tools.transfer_channel_benchmark --role server ...
```

Without it the registered buffer is allocated on a single NUMA node, so only the
rails local to that node reach full bandwidth and the rest are throttled by the
cross-socket link — roughly halving throughput. In testing on a 2-NUMA / 8-rail
host this was the difference between **~110 GB/s** (no interleave) and
**~210 GB/s** (`--interleave=all`), with everything else identical. Page size
only matters in the small regime (descriptor-bound below ~64KB); from ~128KB up
the transfer is bandwidth-bound and flat.

## Cross-node runs

Run the server on one host and the client on another; both hosts need the same
LMCache build and a NIXL runtime. Point the client's `--url` and
`--control-url` at the server host's address, and make the client's
`--listen-url` reachable from the server (the handshake dials back to it).
On hosts with several RDMA devices set the UCX multi-rail variables on both
sides so every rail is used:

```bash
export UCX_TLS=rc,sm,self,cuda_copy          # cuda_copy is required by nixl >= 1.5 on GPU hosts
export UCX_NET_DEVICES=mlx5_bond_0:1,mlx5_bond_1:1,...   # every RDMA device
export UCX_MAX_RMA_RAILS=16                  # paths, two per LAG bond
export UCX_MULTI_LANE_MAX_RATIO=1000
```

```bash
# host B (server)
numactl --interleave=all python -m lmcache.tools.transfer_channel_benchmark \
  --role server --url 0.0.0.0:7600 --control-url 0.0.0.0:7610 \
  --buffer-size 8GB --page-size 512KB --object-size 10MB --num-source-objects 500

# host A (client): write, verified on host B, then read
numactl --interleave=all python -m lmcache.tools.transfer_channel_benchmark \
  --role client --op write --verify --url <B_IP>:7600 --control-url <B_IP>:7610 \
  --listen-url 0.0.0.0:7601 --page-size 512KB --object-size 10MB \
  --num-objects 100 --iters 5 --warmup 1
```

### Measured cross-node results

Two H200 hosts (`ali-us-h200-02` -> `ali-us-h200-03`, 8 RoCE LAG bonds each,
UCX multi-rail as above, nixl 1.5.2, `numactl --interleave=all` on both sides,
512 KiB pages, about 1 GB per iteration, median of 5 iterations after 1
warmup, 2026-11-06):

| object size x count | write | read |
| --- | --- | --- |
| 1 MiB x 1000 | 10.38 ms, 101.0 GB/s | 9.48 ms, 110.6 GB/s |
| 10 MiB x 100 | 7.10 ms, 147.8 GB/s | 5.65 ms, 185.7 GB/s |
| 64 MiB x 16 | 7.30 ms, 147.1 GB/s | 5.68 ms, 188.9 GB/s |

Every write run was verified on the server side (`--verify`), in both
directions (client on either host). During the 64 MiB write run each of the
eight bonds on the client transmitted the same 0.82 GB (6 transfers x 1.07 GB
spread over 8 rails), and the Ethernet counters did not move, so the writes
went over RDMA on every rail.

## Troubleshooting

- **Client hangs after "connected" or during connect, then raises
  `TimeoutError` after ~60s.** The transfer-channel handshake could not reach the
  server. Check that `--url`/`--control-url` on the client point at the *server's*
  reachable address (a common mistake is a typo such as a trailing dot:
  `10.0.0.5.:7600`), that the server is running, and that the ports are open
  between hosts:
  ```bash
  nc -vz <server-host> 7600   # transfer channel
  nc -vz <server-host> 7610   # catalog side-channel
  ```
  The handshake has a fixed 60s timeout, so a misconfiguration fails with a clear
  error instead of hanging forever.
- **`page_size`/`object_size` mismatch error.** The client validates these against
  the server's catalog; pass the same `--page-size` and `--object-size` on both.
- **Allocation `RuntimeError` on the server.** The source pool
  (`--num-source-objects` × `--object-size`) must fit in `--buffer-size`; increase
  the buffer or reduce the pool/object size.
