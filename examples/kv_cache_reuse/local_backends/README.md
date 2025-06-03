# Examples vLLM + LMCache w. local backends
LMCache should be able to reduce the generation time of the second and following calls.
## CPU offloading
- `python offload.py -v v0` - CPU offloading implementation for vLLM v0
- `python offload.py -v v1` - CPU offloading implementation for vLLM v1
## Disk offloading
- `python offload.py -v v0 --use-disk` - Disk offloading implementation for vLLM v0
- `python offload.py -v v1 --use-disk` - Disk offloading implementation for vLLM v1
