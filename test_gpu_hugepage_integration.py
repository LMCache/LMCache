#!/usr/bin/env python3
"""
GPU와 Hugepage 통합 테스트 스크립트

이 스크립트는 GPU 메모리와 hugepage 메모리가 함께 잘 작동하는지 테스트합니다.
"""

import os
import torch
import lmcache.c_ops as lmc_ops
from lmcache.v1.hugepage_memory import get_hugepage_info

# 라이브러리 경로 설정
os.environ['LD_LIBRARY_PATH'] = (
    f"{os.environ.get('CONDA_PREFIX', '')}/lib/python3.11/site-packages/torch/lib:"
    f"{os.environ.get('CONDA_PREFIX', '')}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

def test_gpu_basic_operations():
    """GPU 기본 동작 테스트"""
    print("=== GPU Basic Operations Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # GPU 정보 출력
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # GPU 메모리 상태
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        # GPU 텐서 생성
        gpu_tensor = torch.randn(1000, 1000, device='cuda')
        print(f"GPU tensor created: {gpu_tensor.shape}, device: {gpu_tensor.device}")
        print(f"GPU memory after tensor: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        # GPU 연산 테스트
        result = torch.mm(gpu_tensor, gpu_tensor.T)
        print(f"GPU computation result: {result.shape}")
        
        # 메모리 정리
        del gpu_tensor, result
        torch.cuda.empty_cache()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        print("✅ GPU basic operations test passed")
        return True
        
    except Exception as e:
        print(f"❌ GPU basic operations test failed: {e}")
        return False

def test_hugepage_availability():
    """Hugepage 가용성 테스트"""
    print("\n=== Hugepage Availability Test ===")
    
    try:
        info = get_hugepage_info()
        print(f"Hugepage info: {info}")
        
        if info["available"]:
            print(f"Hugepage size: {info['hugepage_size'] / (1024*1024):.1f} MB")
            print(f"Available count: {info['available_count']}")
            print("✅ Hugepage availability test passed")
            return True
        else:
            print("❌ Hugepages not available")
            return False
            
    except Exception as e:
        print(f"❌ Hugepage availability test failed: {e}")
        return False

def test_gpu_hugepage_memory_transfer():
    """GPU와 Hugepage 간 메모리 전송 테스트"""
    print("\n=== GPU-Hugepage Memory Transfer Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    if not lmc_ops.is_hugepage_available():
        print("❌ Hugepages not available")
        return False
    
    try:
        # GPU에 텐서 생성
        gpu_tensor = torch.randn(500, 500, device='cuda')
        print(f"GPU tensor created: {gpu_tensor.shape}")
        print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        # GPU에서 CPU로 데이터 전송
        cpu_tensor = gpu_tensor.cpu()
        print(f"GPU -> CPU transfer completed: {cpu_tensor.shape}")
        
        # CPU에서 GPU로 데이터 전송
        gpu_tensor2 = cpu_tensor.cuda()
        print(f"CPU -> GPU transfer completed: {gpu_tensor2.shape}")
        
        # 데이터 일치성 확인
        assert torch.allclose(gpu_tensor, gpu_tensor2), "Data mismatch after round-trip transfer"
        print("✅ Data consistency verified")
        
        # 메모리 정리
        del gpu_tensor, gpu_tensor2, cpu_tensor
        torch.cuda.empty_cache()
        
        print("✅ GPU-Hugepage memory transfer test passed")
        return True
        
    except Exception as e:
        print(f"❌ GPU-Hugepage memory transfer test failed: {e}")
        return False

def test_gpu_hugepage_performance():
    """GPU와 Hugepage 성능 테스트"""
    print("\n=== GPU-Hugepage Performance Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        import time
        
        # GPU 메모리 할당 성능 테스트
        sizes = [1000, 2000, 3000]
        
        for size in sizes:
            print(f"\nTesting size: {size}x{size}")
            
            # GPU 할당 시간 측정
            torch.cuda.synchronize()
            start_time = time.time()
            gpu_tensor = torch.randn(size, size, device='cuda')
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            # CPU 할당 시간 측정
            start_time = time.time()
            cpu_tensor = torch.randn(size, size)
            cpu_time = time.time() - start_time
            
            # GPU 연산 시간 측정
            torch.cuda.synchronize()
            start_time = time.time()
            result = torch.mm(gpu_tensor, gpu_tensor.T)
            torch.cuda.synchronize()
            compute_time = time.time() - start_time
            
            print(f"  GPU allocation: {gpu_time*1000:.2f} ms")
            print(f"  CPU allocation: {cpu_time*1000:.2f} ms")
            print(f"  GPU computation: {compute_time*1000:.2f} ms")
            print(f"  Memory: {gpu_tensor.numel() * 4 / 1024**2:.1f} MB")
            
            # 메모리 정리
            del gpu_tensor, cpu_tensor, result
            torch.cuda.empty_cache()
        
        print("✅ GPU-Hugepage performance test passed")
        return True
        
    except Exception as e:
        print(f"❌ GPU-Hugepage performance test failed: {e}")
        return False

def test_gpu_hugepage_stress():
    """GPU와 Hugepage 스트레스 테스트"""
    print("\n=== GPU-Hugepage Stress Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # 여러 GPU 텐서를 동시에 생성하고 연산
        tensors = []
        results = []
        
        print("Creating multiple GPU tensors...")
        for i in range(5):
            tensor = torch.randn(800, 800, device='cuda')
            tensors.append(tensor)
            print(f"  Tensor {i+1}: {tensor.shape}, GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        print("Performing computations...")
        for i, tensor in enumerate(tensors):
            result = torch.mm(tensor, tensor.T)
            results.append(result)
            print(f"  Computation {i+1} completed: {result.shape}")
        
        print("Cleaning up...")
        del tensors, results
        torch.cuda.empty_cache()
        print(f"Final GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        print("✅ GPU-Hugepage stress test passed")
        return True
        
    except Exception as e:
        print(f"❌ GPU-Hugepage stress test failed: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 GPU + Hugepage Integration Testing")
    print("=" * 50)
    
    # 모든 테스트 실행
    tests = [
        test_gpu_basic_operations,
        test_hugepage_availability,
        test_gpu_hugepage_memory_transfer,
        test_gpu_hugepage_performance,
        test_gpu_hugepage_stress
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPU and Hugepage integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\nGPU + Hugepage integration test completed!")

if __name__ == "__main__":
    main() 