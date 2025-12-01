# 使用 nsys 进行 NVTX Profile 指南

## 概述

`_lmcache_nvtx_annotate` 装饰器会在函数执行时添加 NVTX (NVIDIA Tools Extension) 标记，这些标记可以被 NVIDIA Nsight Systems (nsys) 捕获并可视化。

## 前置要求

1. **安装 NVIDIA Nsight Systems**
   ```bash
   # 从 NVIDIA 官网下载并安装
   # 或使用 conda
   conda install -c nvidia nsight-systems
   ```

2. **确保安装了 nvtx Python 包**
   ```bash
   pip install nvtx
   ```

## 基本用法

### 1. 使用 nsys 运行你的程序

```bash
# 基本用法
nsys profile --output=profile.nsys-rep python your_script.py

# 指定 GPU 设备
nsys profile --gpu-metrics-device=0 --output=profile.nsys-rep python your_script.py

# 捕获特定 domain 的 NVTX 标记（lmcache domain）
nsys profile --trace=nvtx --output=profile.nsys-rep python your_script.py
```

### 2. 常用参数说明

```bash
nsys profile \
    --output=profile.nsys-rep \           # 输出文件名
    --trace=cuda,nvtx,osrt \              # 跟踪类型：cuda, nvtx, osrt (OS runtime)
    --gpu-metrics-device=0 \              # 指定 GPU 设备
    --cuda-memory-usage=true \            # 跟踪 CUDA 内存使用
    --duration=30 \                       # 只捕获前 30 秒
    --force-overwrite=true \              # 覆盖已存在的输出文件
    python your_script.py
```

### 3. 针对 LMCache 的完整示例

```bash
# 捕获 LMCache 相关的性能数据
nsys profile \
    --output=lmcache_profile.nsys-rep \
    --trace=cuda,nvtx,osrt \
    --gpu-metrics-device=0 \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    python -m lmcache.v1.multiprocess.server \
        --bind-url tcp://*:5555 \
        --cpu-buffer-size 10.0
```

## 查看和分析结果

### 1. 使用 Nsight Systems GUI

```bash
# 打开 GUI 查看
nsys-ui profile.nsys-rep
```

在 GUI 中：
- **Timeline 视图**：可以看到所有被 `_lmcache_nvtx_annotate` 装饰的函数执行时间
- **NVTX 标记**：在 timeline 上会显示为不同颜色的区域，对应不同的函数
- **Domain 过滤**：可以过滤显示 "lmcache" domain 的标记（详见下方"利用 domain 进行过滤和分析"部分）

### 2. 使用命令行导出报告

```bash
# 导出统计报告
nsys stats --report gputrace --report cudaapis --report nvtx profile.nsys-rep

# 导出 CSV 格式
nsys export --type=csv --output=profile.csv profile.nsys-rep
```

### 3. 查看 NVTX 标记统计

```bash
# 查看 NVTX 标记的统计信息
nsys stats --report nvtx profile.nsys-rep

# 查看特定 domain 的统计（如果支持）
nsys stats --report nvtx profile.nsys-rep | grep lmcache
```

## 利用 domain="lmcache" 进行过滤和分析

`domain="lmcache"` 参数的主要作用是**组织和过滤 NVTX 标记**，让你可以专注于分析 LMCache 相关的代码。

### 1. 在 nsys-ui 中使用 domain 过滤

打开 profile 文件后：

1. **查看 NVTX 标记列表**
   - 在左侧面板找到 "NVTX" 或 "Markers" 部分
   - 展开可以看到按 domain 分组的标记

2. **过滤特定 domain**
   - 在搜索框或过滤器中输入 `lmcache`
   - 或者在 NVTX 标记列表中只勾选 "lmcache" domain
   - Timeline 视图将只显示 LMCache 相关的标记

3. **对比不同 domain**
   - 如果你的代码还有其他 domain（如 PyTorch、CUDA 等）
   - 可以同时显示多个 domain 来对比分析
   - 例如：同时显示 "lmcache" 和 "cuda" domain 来查看 LMCache 操作与 CUDA 操作的关系

### 2. 在命令行中利用 domain

```bash
# 导出 NVTX 数据并过滤 lmcache domain
nsys export --type=csv --output=lmcache_only.csv profile.nsys-rep
# 然后使用 grep 或 Python 脚本过滤 domain="lmcache" 的行

# 查看统计信息时关注 lmcache domain
nsys stats --report nvtx profile.nsys-rep | grep -A 10 "lmcache"
```

### 3. 使用 domain 的优势

**优势 1: 模块化分析**
- 当你的应用包含多个库时（如 LMCache + PyTorch + vLLM）
- 可以只关注 LMCache 相关的性能问题
- 不会被其他库的标记干扰

**优势 2: 清晰的代码组织**
- 所有 LMCache 的函数都标记为 `domain="lmcache"`
- 在 timeline 上可以快速识别哪些是 LMCache 的操作
- 便于理解代码的执行流程

**优势 3: 性能对比**
- 可以对比不同 domain 的执行时间
- 例如：比较 LMCache 操作 vs CUDA kernel 的执行时间比例

### 4. 实际使用示例

**场景 1: 分析 reserve 方法的性能**

```bash
# 运行 profiling
nsys profile --output=reserve_profile.nsys-rep python your_script.py
```

在 nsys-ui 中：
1. 打开 `reserve_profile.nsys-rep`
2. 在 NVTX 标记列表中，找到 domain="lmcache" 的标记
3. 展开可以看到所有 LMCache 函数，包括 `MPStorageManager.reserve`
4. 点击该标记，在 timeline 上高亮显示所有 reserve 调用
5. 可以查看每次调用的时间、频率、以及与其他操作的关系

**场景 2: 对比不同模块的性能**

如果你的代码同时使用了 LMCache 和其他库：

```python
# 其他库可能使用不同的 domain
with nvtx.annotate(message="pytorch_operation", domain="pytorch"):
    # PyTorch 操作
    pass

# LMCache 使用 domain="lmcache"
@_lmcache_nvtx_annotate  # 自动使用 domain="lmcache"
def reserve(...):
    pass
```

在 nsys-ui 中：
- 可以同时显示两个 domain
- 对比 LMCache 操作和 PyTorch 操作的时间分布
- 识别性能瓶颈在哪个模块

### 5. 自定义 domain（高级用法）

如果需要更细粒度的分类，可以创建子 domain：

```python
from nvtx import annotate

# 在 LMCache 内部使用不同的子 domain
with annotate(message="memory_allocation", domain="lmcache.memory"):
    # 内存分配相关代码
    pass

with annotate(message="cache_operation", domain="lmcache.cache"):
    # 缓存操作相关代码
    pass
```

这样可以在 nsys-ui 中进一步细分 LMCache 的不同功能模块。

## 高级用法

### 1. 限制捕获时间

```bash
# 只捕获前 30 秒的数据
nsys profile \
    --output=profile.nsys-rep \
    --duration=30 \
    python your_script.py
```

**注意**: nsys 不支持 `--start-time` 参数。如果需要延迟开始捕获，可以使用以下方法：

**方法 1: 在代码中使用 NVTX 标记控制**
```python
import time
from nvtx import annotate

# 等待 10 秒后再开始需要 profile 的操作
time.sleep(10)

with annotate(message="profile_this_section", domain="lmcache"):
    # 需要 profile 的代码
    pass
```

**方法 2: 使用环境变量控制**
```bash
# 在脚本中检查环境变量，延迟执行关键代码
PROFILE_START_DELAY=10 python your_script.py
```

### 2. 多进程/多 GPU 分析

```bash
# 捕获多个 GPU
nsys profile \
    --output=profile.nsys-rep \
    --gpu-metrics-device=all \
    python your_script.py
```

### 3. 结合其他 profiling 工具

```bash
# 同时使用 nsys 和 PyTorch profiler
python -m torch.profiler \
    --type=nsys \
    --output=torch_profile.nsys-rep \
    your_script.py
```

## 在代码中添加更多标记

如果需要手动添加 NVTX 标记：

```python
from nvtx import annotate

# 作为装饰器
@annotate(message="my_function", domain="lmcache", color="green")
def my_function():
    pass

# 作为上下文管理器
with annotate(message="critical_section", domain="lmcache"):
    # 你的代码
    pass
```

## 常见问题

### 1. 看不到 NVTX 标记

- 确保安装了 `nvtx` Python 包
- 检查 `_lmcache_nvtx_annotate` 装饰器是否正确应用
- 在 nsys 中确保启用了 `--trace=nvtx`

### 2. 性能开销

NVTX 标记的开销很小，但在生产环境中可以考虑：
- 使用环境变量控制是否启用
- 只在需要 profiling 时启用

### 3. 文件大小

nsys 输出文件可能很大，可以：
- 限制捕获时间（`--duration`）
- 只捕获关键部分
- 在代码中使用 NVTX 标记来只标记需要分析的部分
- 使用 `--trace` 参数只启用必要的跟踪类型（如只使用 `--trace=nvtx` 而不是 `--trace=cuda,nvtx,osrt`）

## 示例：分析 MPStorageManager.reserve 方法

```bash
# 运行并捕获 reserve 方法的性能
nsys profile \
    --output=reserve_profile.nsys-rep \
    --trace=cuda,nvtx,osrt \
    --gpu-metrics-device=0 \
    --cuda-memory-usage=true \
    python -m pytest tests/v1/multiprocess/test_mp_storage_manager.py::TestThreadSafety::test_concurrent_reserves
```

在 nsys-ui 中，你可以：
1. 找到 "MPStorageManager.reserve" 的 NVTX 标记
2. 查看该方法的执行时间和调用频率
3. 分析锁竞争情况（通过时间线重叠）
4. 查看相关的 CUDA 操作

## 参考资源

- [NVIDIA Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)
- [NVTX Python 包文档](https://github.com/NVIDIA/NVTX-Python)
- [PyTorch Profiler 文档](https://pytorch.org/docs/stable/profiler.html)

