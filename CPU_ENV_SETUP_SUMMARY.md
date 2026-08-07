# LMCache CPU 环境搭建 + vLLM MP Connector 验证全记录

> 时间：2026-08-06 | 环境：macOS arm64 (Apple Silicon)，无 GPU | 用户自称"龙哥"

---

## 1. 环境准备

### 1.1 前置依赖

| 依赖 | 来源 | 备注 |
|------|------|------|
| Python 3.12 | `brew install python@3.12` | 系统 python3 是 3.9.6，过旧不可用 |
| cmake | `brew install cmake` | 系统缺 cmake |
| uv | pip | 用于虚拟环境和包管理 |

### 1.2 虚拟环境

```bash
# 创建
uv venv --python 3.12 ~/.venv-lmcache

# 激活方式（每次操作前）
source ~/.venv-lmcache/bin/activate
```

### 1.3 关键环境变量（macOS 启动 vLLM 时必须）

```bash
export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")
export HF_HOME=/Users/mbl/.cache/huggingface
export VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_OMP_THREADS_BIND=nobind   # macOS OpenMP 必须
export OMP_NUM_THREADS=4                   # macOS OpenMP 必须
```

> **注意**：仅启动 vLLM 需要 OpenMP 变量；LMCache server/bench 不需要。

---

## 2. vLLM 编译安装（CPU 版）

```bash
# 源码路径
cd /Users/mbl/projects/vllm

# CPU 版编译安装
VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation

# 安装版本
# vllm 0.26.1rc1.dev369+g66b3c0e61.cpu
```

---

## 3. LMCache 编译安装（CPU 版）

```bash
# 项目路径
cd /Users/mbl/projects/LMCache

# 必须用 NO_GPU_EXT=1，不能用 NO_NATIVE_EXT=1
# 原因：lmcache bench server 依赖完整 lmcache.v1.* 下的 C++ 扩展
NO_GPU_EXT=1 pip install --no-build-isolation -e .

# 安装版本
# lmcache 0.5.3rc3.dev3 (g3b8093cf)

# 手动补装 sortedcontainers（NO_GPU_EXT=1 不会自动装）
pip install sortedcontainers
```

---

## 4. LMCache Server + Bench 基线验证

### 4.1 启动 LMCache Server

```bash
lmcache server \
  --port 5555 \
  --http-port 8080 \
  --l1-size-gb 1 \
  --eviction-policy LRU
```

- ZMQ 监听端口：`tcp://localhost:5555`
- HTTP 健康检查：`http://localhost:8080` → `{"status": "healthy"}`
- 日志文件：`/tmp/lmcache_server.log`

### 4.2 Bench 验证

```bash
lmcache bench server \
  --mode cpu \
  --transfer-mode lmcache_driven \
  --num-tokens 512 \
  --end 3
```

| 指标 | 结果 |
|------|------|
| Checksum OK | 3 |
| Checksum FAIL | 0 |
| Pass rate | 100% |

---

## 5. vLLM + LMCache MP Connector 集成

### 5.1 错误尝试：engine_driven 模式（已被纠正）

第一次启动 vLLM 时，由于 `LMCACHE_MP_TRANSFER_MODE` 未设置（默认 `auto`），CPU 环境自动选择了 `engine_driven` 模式。

**问题**：
- Server 侧日志：`Registered non-GPU context (engine_driven_transfer.py:376)` —— 走的是 gather/scatter 拷贝路径
- 龙哥指出：要的是 `lmcache_driven` 模式，不是 `engine_driven`

### 5.2 正确方案：lmcache_driven 模式

通过环境变量强制指定：

```bash
export LMCACHE_MP_TRANSFER_MODE=lmcache_driven
```

`create_transfer_context` 检查此环境变量来决定传输模式。

### 5.3 完整启动命令

```bash
source ~/.venv-lmcache/bin/activate
export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")
export HF_HOME=/Users/mbl/.cache/huggingface
export VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_OMP_THREADS_BIND=nobind
export OMP_NUM_THREADS=4
export LMCACHE_MP_TRANSFER_MODE=lmcache_driven

MODEL=/Users/mbl/.cache/hf_models/opt-125m

nohup vllm serve $MODEL \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
  --no-enable-prefix-caching \
  --enforce-eager \
  --port 8100 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.4 \
  > /tmp/vllm_lmcdriven.log 2>&1 &
```

### 5.4 lmcache_driven 启动成功日志（关键证据）

**Worker 侧日志** (`/tmp/vllm_lmcdriven.log`)：

```
[02:03:06] Creating transfer context (device_type=cpu, mode=lmcache_driven)
           (worker_transfer.py:879)

[02:03:06] Migrated CPU KV cache tensor (nbytes=2202796032) to SHM /lmcache_kv_14959_0
           (shm.py:297)
           ... (共 12 个 layer，全部迁移到 SHM)
```

**Server 侧日志** (`/tmp/lmcache_server.log`，line 95-96)：

```
[02:03:06] CPUCacheContext: 12 layers, 5602 blocks, dtype=torch.float16 (shm-backed)
           (cache_context.py:186)

[02:03:06] Registered KV cache for GPU ID 3039025113800915941 with 12 layers
           (lmcache_driven_transfer.py:921)
```

### 5.5 两种模式对比

| | engine_driven | lmcache_driven |
|------|------|------|
| Server 注册日志 | `Registered non-GPU context (engine_driven_transfer.py:376)` | `Registered KV cache for GPU ID ... (lmcache_driven_transfer.py:921)` |
| KV cache 内存 | 普通 CPU 内存 | SHM 共享内存（`shm-backed`） |
| 传输方式 | worker 侧 gather/scatter CPU 拷贝 | SHM 零拷贝，server 直接读写 worker 的 SHM tensor |
| 性能 | 有额外拷贝开销 | 零拷贝，更高效 |

---

## 6. 端到端 Store / Retrieve 验证

### 6.1 问题：短 prompt 不触发 Store

LMCache 按 **chunk** 存储 KV cache，默认 `chunk_size = 256` tokens（`lmcache_tokens_per_chunk`）。

第一次测试用了短 prompt `"Hello, my name is"`（仅 6 tokens），`6 // 256 = 0`，没有生成任何 STORE metadata，因此不会触发 Store 操作。

Server 日志只显示：
```
[02:03:53] Session cmpl-b39fc3e259c5746b-0-9612ae86 not found, skipping touch
```

### 6.2 解决方案：构造 > 256 tokens 长 prompt

```python
# 构造约 352 tokens 的 prompt
text = "The quick brown fox jumps over the lazy dog. " * 35
```

### 6.3 第一轮：触发 STORE

```bash
curl -s --max-time 300 http://127.0.0.1:8100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/Users/mbl/.cache/hf_models/opt-125m",
       "prompt":"The quick brown fox jumps over the lazy dog. "*35,
       "max_tokens":4}'
```

**Server 日志验证**：

```
[02:10:28] Stored 256 tokens in 0.005 seconds
           (lmcache_driven_transfer.py:1150)
```

### 6.4 第二轮：触发 RETRIEVE

用同样的 prompt 再发一次：

```
[02:10:37] Prefetch request completed (L1+L2): 1/1 retained keys (1 L1, 0 L2) in 0.4 ms
           (storage_manager.py:716)

[02:10:37] Retrieved 256 tokens in 0.003 seconds
           (lmcache_driven_transfer.py:1341)
```

### 6.5 第三轮：再次命中

```
[02:11:48] Retrieved 256 tokens in 0.003 seconds
           (lmcache_driven_transfer.py:1341)
```

### 6.6 完整链路总结

```
第1次请求（352 tokens prompt）
  → vLLM prefill 生成 KV cache
  → LMCacheMPConnector 检测到 256 token chunk
  → lmcache_driven_transfer.store() → SHM 零拷贝写入 LMCache server
  → Server: "Stored 256 tokens in 0.005 seconds"

第2次请求（同样 prompt）
  → vLLM 查询 LMCacheMPConnector
  → lmcache_driven_transfer.retrieve() → SHM 零拷贝读回
  → Server: "Retrieved 256 tokens in 0.003 seconds"
  → vLLM 跳过 prefill，直接 decode

第3次请求（同样 prompt）
  → 再次命中 cache
  → Server: "Retrieved 256 tokens in 0.003 seconds"
```

---

## 7. 关键代码路径

### 7.1 传输模式选择

- 入口：`lmcache/v1/multiprocess/transfer_context/worker_transfer.py`
- 函数：`create_transfer_context`
- 控制变量：`LMCACHE_MP_TRANSFER_MODE`（可选值：`auto`、`engine_driven`、`lmcache_driven`）

### 7.2 lmcache_driven 传输实现

- Worker 注册：`lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`
  - `register` 方法（line 921 附近）
  - `store` 方法（line 1150 附近）
  - `retrieve` 方法（line 1341 附近）
- CPU SHM 支持：`lmcache/v1/platform/cpu/shm.py`
  - `Migrated CPU KV cache tensor to SHM` 日志来自此处

### 7.3 vLLM 集成

- Connector：`lmcache/integration/vllm/lmcache_mp_connector.py`
- Adapter：`lmcache/integration/vllm/vllm_multi_process_adapter.py`

### 7.4 KV Cache 存储触发条件

- Chunk 大小：`LMCacheEngineConfig.lmcache_tokens_per_chunk`（默认 256）
- 触发逻辑：`GetStoreMetadata` 在 `lmcache_mp_connector.py` 中，prompt tokens 按 chunk 切分，只有 >= 256 tokens 才会产生 STORE metadata

---

## 8. 踩坑清单

| 编号 | 问题 | 原因 | 解决 |
|------|------|------|------|
| 1 | `sortedcontainers` 缺失 | `NO_GPU_EXT=1` 不会自动装此依赖 | 手动 `pip install sortedcontainers` |
| 2 | `lmcache bench server` 失败 | 用了 `NO_NATIVE_EXT=1` 而非 `NO_GPU_EXT=1`，C++ 扩展缺失 | 改用 `NO_GPU_EXT=1` |
| 3 | 默认走了 engine_driven 模式 | CPU 上 `auto` 默认选 engine_driven | 设置 `LMCACHE_MP_TRANSFER_MODE=lmcache_driven` |
| 4 | 短 prompt 不触发 Store | chunk_size = 256，6 token prompt 凑不满一个 chunk | 构造 > 256 token 的长 prompt |

---

## 9. 运行中的进程

| 进程 | 启动方式 | 端口 | 日志 |
|------|------|------|------|
| LMCache Server | `lmcache server ...` | ZMQ 5555, HTTP 8080 | `/tmp/lmcache_server.log` |
| vLLM (lmcache_driven) | `vllm serve ...` | HTTP 8100 | `/tmp/vllm_lmcdriven.log` |

---

## 10. 快速复用命令

```bash
# 激活环境
source ~/.venv-lmcache/bin/activate

# 设置环境变量
export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")
export HF_HOME=/Users/mbl/.cache/huggingface
export VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_OMP_THREADS_BIND=nobind
export OMP_NUM_THREADS=4
export LMCACHE_MP_TRANSFER_MODE=lmcache_driven

# 启动 LMCache Server
lmcache server --port 5555 --http-port 8080 --l1-size-gb 1 --eviction-policy LRU

# 启动 vLLM
MODEL=/Users/mbl/.cache/hf_models/opt-125m
vllm serve $MODEL \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
  --no-enable-prefix-caching --enforce-eager \
  --port 8100 --max-model-len 1024 --gpu-memory-utilization 0.4

# 发送测试请求（需要 > 256 tokens 的 prompt 才能触发 Store）
curl -s http://127.0.0.1:8100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/Users/mbl/.cache/hf_models/opt-125m",
       "prompt":"The quick brown fox jumps over the lazy dog. " (重复 35 次),
       "max_tokens":4}'

# 查看 Server 日志确认 Store/Retrieve
grep -i "Stored\|Retrieved" /tmp/lmcache_server.log | tail -5
```
