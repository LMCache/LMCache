#!/usr/bin/env python3
"""
演示如何利用 domain="lmcache" 进行性能分析

这个示例展示了：
1. 如何使用 domain 来区分不同模块的标记
2. 如何在 nsys-ui 中利用 domain 进行过滤和分析
"""

import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from nvtx import annotate
    from lmcache.v1.multiprocess.mp_storage_manager import MPStorageManager
    from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
    from lmcache.v1.memory_management import MemoryFormat
    import torch
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装 LMCache 和 nvtx")
    sys.exit(1)


def simulate_other_library_operations():
    """模拟其他库的操作（使用不同的 domain）"""
    # 模拟 PyTorch 操作
    with annotate(message="pytorch_tensor_creation", domain="pytorch", color="red"):
        time.sleep(0.01)  # 模拟操作
        tensor = torch.randn(100, 100)
    
    # 模拟 CUDA 操作
    with annotate(message="cuda_kernel_launch", domain="cuda", color="yellow"):
        time.sleep(0.005)
        tensor = tensor.cuda()
    
    return tensor


def example_mixed_operations():
    """
    示例：混合操作，展示如何利用 domain 区分不同模块
    
    在 nsys-ui 中：
    1. 可以只显示 domain="lmcache" 的标记，专注于 LMCache 性能
    2. 可以同时显示多个 domain，对比不同模块的性能
    3. 可以过滤特定 domain，排除干扰
    """
    print("=" * 60)
    print("混合操作示例：展示 domain 的使用")
    print("=" * 60)
    
    # 初始化 LMCache
    storage_manager = MPStorageManager(cpu_buffer_size=1.0)
    
    # 创建测试 keys
    keys = [
        IPCCacheEngineKey.from_int_hash("test_model", 1, 0, i)
        for i in range(5)
    ]
    shape = (2, 10, 16, 64)
    dtype = torch.float16
    fmt = MemoryFormat.KV_2LTD
    
    print("\n[步骤 1] 其他库的操作（domain='pytorch', 'cuda'）")
    # 这些操作使用不同的 domain，在 nsys-ui 中可以区分
    tensor = simulate_other_library_operations()
    
    print("\n[步骤 2] LMCache 操作（domain='lmcache'）")
    # 这个操作会自动使用 domain="lmcache"
    # 在 nsys-ui 中可以过滤只显示这个 domain
    handle, reserved_dict = storage_manager.reserve(keys, shape, dtype, fmt)
    print(f"  Reserve 完成: {len(reserved_dict)} keys")
    
    print("\n[步骤 3] 更多其他库操作")
    tensor = simulate_other_library_operations()
    
    print("\n[步骤 4] 更多 LMCache 操作")
    storage_manager.commit(handle)
    found = storage_manager.lookup(keys)
    print(f"  Lookup 完成: 找到 {found} keys")
    
    print("\n[步骤 5] 混合操作")
    with annotate(message="custom_processing", domain="custom", color="purple"):
        # 自定义处理逻辑
        time.sleep(0.002)
    
    # 更多 LMCache 操作
    with storage_manager.retrieve(keys) as objs:
        print(f"  Retrieve 完成: {len(objs)} 对象")
    
    storage_manager.on_retrieve_finished(keys)
    storage_manager.close()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("\n在 nsys-ui 中查看 profile 文件时：")
    print("1. 在 NVTX 标记列表中，可以看到按 domain 分组的标记")
    print("2. 只勾选 'lmcache' domain，可以只关注 LMCache 的性能")
    print("3. 同时勾选多个 domain，可以对比不同模块的性能")
    print("4. 在 timeline 上，不同 domain 的标记会有不同的颜色")


def example_domain_filtering_demo():
    """
    演示如何使用 domain 进行过滤分析
    
    这个函数展示了如何在实际分析中利用 domain
    """
    print("\n" + "=" * 60)
    print("Domain 过滤分析示例")
    print("=" * 60)
    
    storage_manager = MPStorageManager(cpu_buffer_size=1.0)
    
    # 创建多个批次的 keys
    all_keys = []
    for batch_id in range(3):
        keys = [
            IPCCacheEngineKey.from_int_hash("model", 1, 0, batch_id * 10 + i)
            for i in range(3)
        ]
        all_keys.extend(keys)
        
        shape = (2, 10, 16, 64)
        dtype = torch.float16
        fmt = MemoryFormat.KV_2LTD
        
        # 每个 reserve 调用都会被标记为 domain="lmcache"
        handle, reserved_dict = storage_manager.reserve(keys, shape, dtype, fmt)
        storage_manager.commit(handle)
        
        print(f"批次 {batch_id}: 处理了 {len(reserved_dict)} keys")
    
    # 查找所有 keys
    found = storage_manager.lookup(all_keys)
    print(f"\n总共找到 {found} keys")
    
    storage_manager.close()
    
    print("\n分析提示：")
    print("- 在 nsys-ui 中，可以搜索 'MPStorageManager.reserve'")
    print("- 只显示 domain='lmcache' 的标记，查看所有 LMCache 操作")
    print("- 统计每个函数的调用次数和执行时间")


if __name__ == "__main__":
    print("Domain 使用示例")
    print("运行方式: nsys profile --output=domain_example.nsys-rep python analyze_domain_example.py")
    print()
    
    # 运行示例
    example_mixed_operations()
    example_domain_filtering_demo()
    
    print("\n" + "=" * 60)
    print("查看结果:")
    print("  nsys-ui domain_example.nsys-rep")
    print("=" * 60)

