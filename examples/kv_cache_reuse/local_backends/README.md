# Examples vLLM + LMCache w. local backends

> **Legacy in-process examples:** These scripts construct `LMCacheEngine`
> directly. Prefer MP for new deployments where the required feature is
> supported; see the [migration guide](https://docs.lmcache.ai/legacy/migration_to_mp.html).

LMCache should be able to reduce the generation time of the second and following calls.
## CPU offloading
- `python offload.py -v v0` - CPU offloading implementation for vLLM v0
- `python offload.py -v v1` - CPU offloading implementation for vLLM v1
## Disk offloading
- `python offload.py -v v0 --use-disk` - Disk offloading implementation for vLLM v0
- `python offload.py -v v1 --use-disk` - Disk offloading implementation for vLLM v1

## RUST raw block based Disk offloading

   # WARNING: This will erase the content of target device.
- `python rust_backend_offload.py --disk_path=/dev/nvme0n1` - posix disk offloading
- `python rust_backend_offload.py --disk_path=/dev/nvme0n1 --use_uring` - io_uring disk offloading
