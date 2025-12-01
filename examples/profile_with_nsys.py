#!/usr/bin/env python3
"""
使用 nsys 进行 LMCache 性能分析的示例

运行方式:
1. 直接运行（需要先启动 nsys）:
   nsys profile --output=profile.nsys-rep python profile_with_nsys.py

2. 或者使用提供的脚本:
   ./nsys_profile_example.sh python profile_with_nsys.py
"""

import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from lmcache.v1.multiprocess.mp_storage_manager import MPStorageManager
    from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
    from lmcache.v1.memory_management import MemoryFormat
    import torch
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装 LMCache 并设置正确的 Python 路径")
    sys.exit(1)


def example_reserve_operations():
    """示例：测试 reserve 操作的性能"""
    print("初始化 MPStorageManager...")
    storage_manager = MPStorageManager(cpu_buffer_size=1.0)  # 1GB buffer
    
    # 创建一些测试 keys
    keys = [
        IPCCacheEngineKey.from_int_hash("test_model", 1, 0, i)
        for i in range(10)
    ]
    
    shape = (2, 10, 16, 64)
    dtype = torch.float16
    fmt = MemoryFormat.KV_2LTD
    
    print(f"执行 reserve 操作 ({len(keys)} keys)...")
    # 这个操作会被 _lmcache_nvtx_annotate 装饰器标记
    handle, reserved_dict = storage_manager.reserve(keys, shape, dtype, fmt)
    
    print(f"Reserve 完成: handle={handle}, reserved={len(reserved_dict)} keys")
    
    # Commit
    print("执行 commit 操作...")
    storage_manager.commit(handle)
    
    # Lookup
    print("执行 lookup 操作...")
    found_count = storage_manager.lookup(keys)
    print(f"Lookup 完成: 找到 {found_count} keys")
    
    # Retrieve
    print("执行 retrieve 操作...")
    with storage_manager.retrieve(keys) as objs:
        print(f"Retrieve 完成: 获取 {len(objs)} 对象")
    
    storage_manager.on_retrieve_finished(keys)
    
    print("清理...")
    storage_manager.close()


def example_concurrent_reserve():
    """示例：并发 reserve 操作（模拟多线程场景）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print("初始化 MPStorageManager...")
    storage_manager = MPStorageManager(cpu_buffer_size=2.0)  # 2GB buffer
    
    def reserve_keys(thread_id, num_keys=5):
        """每个线程执行的 reserve 操作"""
        keys = [
            IPCCacheEngineKey.from_int_hash("test_model", 1, 0, thread_id * 100 + i)
            for i in range(num_keys)
        ]
        shape = (2, 10, 16, 64)
        dtype = torch.float16
        fmt = MemoryFormat.KV_2LTD
        
        # 这个操作会被 NVTX 标记，可以在 nsys 中看到
        handle, reserved_dict = storage_manager.reserve(keys, shape, dtype, fmt)
        storage_manager.commit(handle)
        return thread_id, len(reserved_dict)
    
    num_threads = 4
    print(f"启动 {num_threads} 个线程并发执行 reserve...")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(reserve_keys, i) for i in range(num_threads)]
        results = [f.result() for f in as_completed(futures)]
    
    print("并发 reserve 完成:")
    for thread_id, num_reserved in results:
        print(f"  Thread {thread_id}: {num_reserved} keys reserved")
    
    storage_manager.close()


if __name__ == "__main__":
    print("=" * 60)
    print("LMCache nsys Profiling 示例")
    print("=" * 60)
    print()
    print("注意: 这个脚本应该通过 nsys 运行:")
    print("  nsys profile --output=profile.nsys-rep python profile_with_nsys.py")
    print()
    
    # 检查是否在 nsys 环境中运行
    if "NSYS_PROFILE" not in os.environ:
        print("提示: 未检测到 nsys 环境变量，但可以继续运行")
        print("建议使用: nsys profile --output=profile.nsys-rep python profile_with_nsys.py")
        print()
    
    # 运行示例
    print("\n[示例 1] 基本 reserve 操作")
    print("-" * 60)
    example_reserve_operations()
    
    print("\n[示例 2] 并发 reserve 操作")
    print("-" * 60)
    example_concurrent_reserve()
    
    print("\n" + "=" * 60)
    print("完成! 使用 nsys-ui 查看 profile.nsys-rep 文件")
    print("=" * 60)

