# Examples vLLM + LMCache w. local backends
LMCache should be able to reduce the generation time of the second and following calls.
## CPU offloading
- `python offload.py -v v0` - CPU offloading implementation for vLLM v0
- `python offload.py -v v1` - CPU offloading implementation for vLLM v1
## Disk offloading
- `python offload.py -v v0 --use-disk` - Disk offloading implementation for vLLM v0
- `python offload.py -v v1 --use-disk` - Disk offloading implementation for vLLM v1

## Multi-device disk offloading

`LMCACHE_LOCAL_DISK` accepts a comma-separated list of directories, one per
storage device. Paths are assigned to workers by the `by_gpu` strategy
(`LMCACHE_LOCAL_DISK_PATH_SHARDING`, the default), keyed by each worker's
local rank:

- `#paths == #workers`: one path per worker.
- `#paths > #workers`: each worker is assigned a contiguous *subset* of paths
  so no disk sits idle (e.g. TP=2 with 4 disks -> rank 0 owns disks 0-1,
  rank 1 owns disks 2-3). Within a subset, files are routed by a stable hash
  of the chunk hash.
- `#workers > #paths`: workers share paths.

The path count and local worker count must be equal or an exact multiple of
each other, otherwise startup fails with a clear error.

```bash
# TP=2, 4 disks -> 2 disks per rank
python offload.py -v v1 --use-disk --local-disk /mnt/disk0,/mnt/disk1,/mnt/disk2,/mnt/disk3
```

## RUST raw block based Disk offloading

   # WARNING: This will erase the content of target device.
- `python rust_backend_offload.py --disk_path=/dev/nvme0n1` - posix disk offloading
- `python rust_backend_offload.py --disk_path=/dev/nvme0n1 --use_uring` - io_uring disk offloading
