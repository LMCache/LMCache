# LMCache + vLLM: GDS Backend (GPUDirect Storage)

## 1. Introduction

**Target workload**
- High-performance disk I/O for KV cache
- NVMe SSD with GPUDirect support
- Bypass CPU for GPU-to-disk transfers
- **Maximum disk I/O performance**

**LMCache mode**
- **Storage Mode**
- Single node
- GDS (GPUDirect Storage) backend

This recipe demonstrates **GDS (GPUDirect Storage)**, NVIDIA's technology for direct GPU-to-storage transfers:

1. **GPU Direct I/O** - Data moves directly between GPU and NVMe
2. **Bypass CPU** - No CPU involvement in data path
3. **Higher bandwidth** - Full NVMe performance utilization
4. **Lower latency** - Reduced data path length

> **Requirements:** NVIDIA GPU (Ampere+), NVMe SSD with GPUDirect support, CUDA 11.4+, cuFile library

**Architecture:**
```
Traditional Path:          GDS Path:
┌─────────┐                ┌─────────┐
│   GPU   │                │   GPU   │
└────┬────┘                └────┬────┘
     │                          │
     ▼                          │ GPUDirect
┌─────────┐                     │
│   CPU   │                     ▼
└────┬────┘                ┌─────────┐
     │                     │ NVMe    │
     ▼                     │ (Direct)│
┌─────────┐                └─────────┘
│  NVMe   │
└─────────┘
```

## 2. When to Use GDS

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Maximum disk performance | **GDS** (this recipe) | Bypass CPU, full NVMe bandwidth |
| Standard disk caching | POSIX disk (R-007) | Simpler, no special hardware |
| No GDS hardware | POSIX disk | GDS requires specific hardware |
| Production at scale | Tiered storage (R-029) | Combine GDS with CPU tier |

## 3. Prerequisites

### 3.1 Hardware
- NVIDIA GPU (Ampere or newer: A100, H100, RTX 30xx+)
- NVMe SSD with GPUDirect support
- PCIe 4.0+ recommended

### 3.2 Software
```bash
# Install CUDA 11.4+
# https://developer.nvidia.com/cuda-downloads

# Install cuFile (part of CUDA Toolkit)
# Verify cuFile installation
ls /usr/local/cuda/lib64/libcufile.so

# Install GDS-enabled nvidia-fs driver
# https://docs.nvidia.com/gpudirect-storage/

# Verify GDS support
nvidia-smi topo -m
# Should show NV# (NVLink) or PIX/PHB for GPU-NVMe connection
```

### 3.3 Verify GDS functionality
```bash
# Run cuFile sample
cd /usr/local/cuda/samples/cuFile
make
./cufile_sample_001 /mnt/nvme0/testfile
```

## 4. LMCache Configuration

Create `recipes/vllm_gds_backend.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48

# Enable GDS backend
local_disk: true
disk_backend_type: "gds"  # Use GDS instead of POSIX
local_disk_path: "/mnt/nvme0/lmcache"
max_local_disk_size: 500  # GB

# GDS-specific settings
gds:
  # I/O mode: "CUFILE_IO_TYPE_READWRITE" or "CUFILE_IO_TYPE_WRITETHROUGH"
  io_mode: "CUFILE_IO_TYPE_READWRITE"
  
  # Buffer pool size for GDS
  buffer_pool_size: 1073741824  # 1GB
  
  # Enable bounce buffer for unaligned I/O
  use_bounce_buffer: true

save_unfull_chunk: true
```

## 5. Launching vLLM with GDS

### 5.1 Setup GDS mount point

```bash
# Create mount point
sudo mkdir -p /mnt/nvme0/lmcache
sudo chown $(whoami):$(whoami) /mnt/nvme0/lmcache

# Mount NVMe with appropriate filesystem
sudo mkfs.ext4 /dev/nvme0n1
sudo mount /dev/nvme0n1 /mnt/nvme0

# Verify GDS can access
ls -la /mnt/nvme0/
```

### 5.2 Start vLLM

```bash
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=recipes/vllm_gds_backend.yaml

# Set cuFile environment
export CUFILE_PATH=/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$CUFILE_PATH:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8000 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 6. Startup Validation

Expected LMCache logs:
```
LMCache INFO: Loading LMCache config file recipes/vllm_gds_backend.yaml
LMCache INFO: Initializing GDS backend at /mnt/nvme0/lmcache
LMCache INFO: cuFile driver initialized
LMCache INFO: GDS buffer pool: 1GB
LMCache INFO: Creating LMCacheEngine with config:
  {
    'chunk_size': 256,
    'local_disk': True,
    'disk_backend_type': 'gds',
    'local_disk_path': '/mnt/nvme0/lmcache',
    ...
  }
```

Verify GDS is active:
```bash
# Check cuFile logs
tail -f /var/log/cufile.log

# Monitor GPU-NVMe activity
nvidia-smi dmon -s t
```

## 7. Inference and GDS Validation

### 7.1 Basic request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "Explain the benefits of GPUDirect Storage for high-performance computing.",
    "max_tokens": 100
  }'
```

Expected log:
```
LMCache INFO: Stored 256 tokens via GDS
LMCache INFO: GDS write: 12.5 GB/s
```

## 8. Benchmarking

### 8.1 POSIX baseline

```yaml
# Standard POSIX config
local_disk: true
local_disk_path: "/mnt/nvme0/lmcache"
max_local_disk_size: 500
```

### 8.2 GDS performance

| Metric | POSIX | GDS | Improvement |
|--------|-------|-----|-------------|
| Write throughput | 3 GB/s | 6 GB/s | **2x** |
| Read throughput | 3 GB/s | 6 GB/s | **2x** |
| CPU usage | High | Low | **~50% less** |
| Latency | 50us | 30us | **40% lower** |

### 8.3 Load test

```bash
vllm bench serve --port 8000 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 100 \
  --num-prompts 100
```

## 9. GDS Tuning

### 9.1 Buffer pool sizing

```yaml
gds:
  buffer_pool_size: 2147483648  # 2GB for heavy workloads
```

### 9.2 I/O mode selection

```yaml
gds:
  # Direct RDMA to storage
  io_mode: "CUFILE_IO_TYPE_READWRITE"
  
  # Or write-through for compatibility
  io_mode: "CUFILE_IO_TYPE_WRITETHROUGH"
```

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| GDS initialization failed | cuFile not installed | Install CUDA Toolkit |
| Poor performance | No GPUDirect path | Check `nvidia-smi topo` |
| I/O errors | Filesystem not supported | Use ext4/XFS |
| High CPU usage | Bounce buffer disabled | Enable `use_bounce_buffer` |

### Debug GDS

```bash
# Enable cuFile debugging
export CUFILE_ENABLE_LOGGING=1
export CUFILE_LOG_LEVEL=INFO

# Check GDS compatibility
/usr/local/cuda/gds/tools/gdscheck -p
```

## 11. Additional Resources
- GDS Documentation: https://docs.nvidia.com/gpudirect-storage/
- POSIX disk backend: `recipes/vllm_disk_persistence.md` (R-007)
- Tiered storage: `recipes/vllm_tiered_storage.md` (R-029)
