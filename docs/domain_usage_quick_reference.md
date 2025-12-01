# domain="lmcache" 快速使用指南

## 什么是 domain？

`domain` 是 NVTX 标记的一个属性，用于**组织和分类**不同的代码模块。在 LMCache 中，所有使用 `_lmcache_nvtx_annotate` 装饰器的函数都会自动标记为 `domain="lmcache"`。

## 为什么使用 domain？

### 1. **模块化分析**
当你的应用包含多个库时（LMCache + PyTorch + vLLM 等），可以：
- ✅ 只关注 LMCache 相关的性能问题
- ✅ 排除其他库的标记干扰
- ✅ 快速定位 LMCache 的性能瓶颈

### 2. **清晰的代码组织**
- 所有 LMCache 函数统一标记为 `domain="lmcache"`
- 在 timeline 上可以快速识别 LMCache 操作
- 便于理解代码执行流程

### 3. **性能对比**
- 对比不同模块的执行时间
- 分析 LMCache 操作与其他操作的时间比例

## 在 nsys-ui 中使用 domain

### 步骤 1: 打开 profile 文件
```bash
nsys-ui profile.nsys-rep
```

### 步骤 2: 找到 NVTX 标记
1. 在左侧面板找到 **"NVTX"** 或 **"Markers"** 部分
2. 展开可以看到按 domain 分组的标记
3. 找到 `domain="lmcache"` 的标记组

### 步骤 3: 过滤显示
**方法 A: 使用标记列表**
- 在 NVTX 标记列表中，只勾选 `lmcache` domain
- Timeline 视图将只显示 LMCache 相关的标记

**方法 B: 使用搜索**
- 在搜索框中输入 `lmcache`
- 或者输入函数名，如 `MPStorageManager.reserve`

### 步骤 4: 分析特定函数
1. 在标记列表中找到 `MPStorageManager.reserve`
2. 点击该标记，在 timeline 上高亮显示所有 reserve 调用
3. 查看：
   - 每次调用的执行时间
   - 调用频率
   - 与其他操作的关系（如 CUDA kernel）

## 实际使用场景

### 场景 1: 分析 reserve 方法的性能瓶颈

```bash
# 1. 运行 profiling
nsys profile --output=reserve_profile.nsys-rep python your_script.py

# 2. 打开 GUI
nsys-ui reserve_profile.nsys-rep
```

在 GUI 中：
1. 过滤显示 `domain="lmcache"`
2. 找到 `MPStorageManager.reserve` 标记
3. 查看每次调用的时间线
4. 分析是否有锁竞争（多个 reserve 调用重叠）

### 场景 2: 对比 LMCache 与其他模块的性能

如果你的代码同时使用多个库：

```python
# PyTorch 操作（domain="pytorch"）
with nvtx.annotate(message="pytorch_op", domain="pytorch"):
    tensor = torch.randn(100, 100)

# LMCache 操作（domain="lmcache"）
@_lmcache_nvtx_annotate  # 自动使用 domain="lmcache"
def reserve(...):
    pass
```

在 nsys-ui 中：
- 同时勾选 `lmcache` 和 `pytorch` domain
- 对比两个模块的时间分布
- 识别性能瓶颈在哪个模块

### 场景 3: 只关注 LMCache 的性能

当你的应用很复杂，包含很多库时：
- 只勾选 `domain="lmcache"`
- 排除所有其他库的干扰
- 专注于分析 LMCache 的性能问题

## 命令行中使用 domain

### 查看统计信息
```bash
# 查看所有 NVTX 标记的统计
nsys stats --report nvtx profile.nsys-rep

# 过滤 lmcache domain（使用 grep）
nsys stats --report nvtx profile.nsys-rep | grep -i lmcache
```

### 导出数据
```bash
# 导出 CSV，然后可以用 Python 脚本过滤 domain="lmcache"
nsys export --type=csv --output=profile.csv profile.nsys-rep

# 使用 Python 过滤
python -c "
import pandas as pd
df = pd.read_csv('profile.csv')
lmcache_df = df[df['domain'] == 'lmcache']
print(lmcache_df)
"
```

## 代码示例

### 基本用法（自动）
```python
# 使用装饰器，自动设置 domain="lmcache"
@_lmcache_nvtx_annotate
def reserve(self, keys, ...):
    # 这个函数会自动标记为 domain="lmcache"
    pass
```

### 手动添加标记
```python
from nvtx import annotate

# 在代码块中使用
with annotate(message="critical_section", domain="lmcache"):
    # LMCache 相关代码
    pass
```

### 创建子 domain（高级）
```python
# 更细粒度的分类
with annotate(message="memory_alloc", domain="lmcache.memory"):
    # 内存分配相关
    pass

with annotate(message="cache_op", domain="lmcache.cache"):
    # 缓存操作相关
    pass
```

## 快速检查清单

使用 domain 进行分析时：

- [ ] 确认 `_lmcache_nvtx_annotate` 装饰器已应用到目标函数
- [ ] 在 nsys-ui 中找到了 NVTX 标记列表
- [ ] 能够过滤显示 `domain="lmcache"` 的标记
- [ ] 能够识别和选择特定的 LMCache 函数（如 `reserve`）
- [ ] 能够查看函数的时间线和统计信息
- [ ] 能够对比不同 domain 的性能

## 常见问题

**Q: 在 nsys-ui 中看不到 domain 分组？**
A: 确保使用了 `--trace=nvtx` 参数运行 nsys，并且代码中确实使用了 NVTX 标记。

**Q: 如何知道某个函数使用了哪个 domain？**
A: 查看函数定义，如果使用了 `@_lmcache_nvtx_annotate`，则使用 `domain="lmcache"`。

**Q: 可以同时显示多个 domain 吗？**
A: 可以，在 nsys-ui 的标记列表中，可以同时勾选多个 domain。

**Q: domain 会影响性能吗？**
A: 不会，domain 只是标记的一个属性，对性能没有影响。

## 参考

- 详细文档：`docs/nsys_profiling_guide.md`
- 示例代码：`examples/analyze_domain_example.py`
- NVTX 文档：https://github.com/NVIDIA/NVTX-Python

