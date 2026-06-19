# Installation

## Download Wheel

**lmcache-0.4.8rc2.dev15-cp312-cp312-linux_x86_64.whl**

SHA256: `ebbf8be0239b47b6b6b2c77a18a4fc763dfb54b99b49240835fdc64fbb08de5c`

## Quick Install

```bash
# Download the wheel from this release
pip install lmcache-0.4.8rc2.dev15-cp312-cp312-linux_x86_64.whl --force-reinstall --no-deps
```

## Build from Source

If you need a different Python version or platform, build from source:

```bash
git clone https://github.com/efschu/LMCache.git
cd LMCache
git checkout dev

# Build wheel (requires GPU)
bash scripts/build_manylinux_gpu.sh

# Install
pip install wheelhouse/*.whl --force-reinstall --no-deps
```

See [docs/BUILD.md](docs/BUILD.md) for detailed build instructions.

---

## What's New

This pre-release implements **engine-driven multi-group KV cache transfer** for hybrid models like Qwen3.6-27B.

### Key Features

- **CPU-only LMCache Server**: No GPU memory allocation (was ~666 MB/GPU)
- **Multi-Group Support**: Works with hybrid models (Attention + GDN/Mamba groups)
- **Prefetch Fix**: Properly awaits all group handles before retrieval

### Changes

- `lmcache/v1/multiprocess/custom_types.py`: Added `GroupLayoutInfo` for per-group metadata
- `lmcache/v1/multiprocess/transfer_context/base.py`: Multi-group gather/scatter functions
- `lmcache/v1/multiprocess/transfer_context/worker_transfer.py`: Worker-side multi-group support
- `lmcache/v1/multiprocess/modules/engine_driven_transfer.py`: Server-side multi-group support
- `lmcache/v1/multiprocess/modules/lookup.py`: Fixed multi-group prefetch handling

---

## Requirements

- Python 3.12
- CUDA 12.x
- PyTorch 2.5+
- NVIDIA GPU (RTX 3080+, A100, H100, etc.)

---

## Documentation

- [README.md](README.md) - Overview and quick start
- [docs/BUILD.md](docs/BUILD.md) - Build instructions
- [Lmcache_engine_driven_multigroup.md](Lmcache_engine_driven_multigroup.md) - Full design document
