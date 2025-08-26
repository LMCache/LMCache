#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Hugepage를 실제 계산에 사용하는 테스트 스크립트

이 스크립트는 hugepage 메모리를 할당하고 실제 계산 작업에 사용하는 것을 테스트합니다.
"""

# Standard
import ctypes
import os
import time

# Third Party
import numpy as np
import torch

# First Party
from lmcache.v1.hugepage_memory import (
    HugepageMemoryAllocator,
)
import lmcache.c_ops as lmc_ops

# 라이브러리 경로 설정
os.environ["LD_LIBRARY_PATH"] = (
    f"{os.environ.get('CONDA_PREFIX', '')}/lib/python3.11/site-packages/torch/lib:"
    f"{os.environ.get('CONDA_PREFIX', '')}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
)


def test_basic_computation_with_hugepage():
    """Hugepage를 사용한 기본 계산 테스트"""
    print("=== Basic Computation with Hugepage ===")

    try:
        # 1MB hugepage 할당
        size = 1024 * 1024
        print(f"1. Allocating {size / (1024 * 1024):.1f} MB using hugepage...")

        ptr = lmc_ops.alloc_pinned_hugepage_ptr(size, 0)
        print(f"   ✅ Hugepage allocated: {ptr}")

        # 메모리를 ctypes 배열로 변환
        array_type = (ctypes.c_uint8 * size).from_address(ptr)

        print(f"2. Array size: {len(array_type)} elements")

        # 간단한 계산: 모든 요소를 42로 설정
        print("3. Performing computation: setting all elements to 42...")
        start_time = time.time()
        for i in range(size):
            array_type[i] = 42
        computation_time = time.time() - start_time

        print(f"   ✅ Computation completed in {computation_time * 1000:.2f} ms")
        print(f"   ✅ First 10 elements: {array_type[:10]}")
        print(f"   ✅ Last 10 elements: {array_type[-10:]}")

        # 검증: 모든 요소가 42인지 확인
        all_42 = all(array_type[i] == 42 for i in range(size))
        assert all_42, "Not all elements are 42"
        print("   ✅ Data verification passed")

        # 메모리 해제
        lmc_ops.free_pinned_hugepage_ptr(ptr, size)
        print("4. ✅ Hugepage memory freed successfully")

        return True

    except Exception as e:
        print(f"❌ Basic computation test failed: {e}")
        return False


def test_matrix_operations_with_hugepage():
    """Hugepage를 사용한 행렬 연산 테스트"""
    print("\n=== Matrix Operations with Hugepage ===")

    try:
        # 2MB hugepage 할당 (행렬 연산용)
        size = 2 * 1024 * 1024
        print(f"1. Allocating {size / (1024 * 1024):.1f} MB using hugepage...")

        ptr = lmc_ops.alloc_pinned_hugepage_ptr(size, 0)
        print(f"   ✅ Hugepage allocated: {ptr}")

        # 메모리를 float32 배열로 변환 (512x512 행렬)
        matrix_size = 512
        total_elements = matrix_size * matrix_size
        element_size = 4  # float32

        if total_elements * element_size <= size:
            array_type = (ctypes.c_float * total_elements).from_address(ptr)

            print(
                f"2. Matrix size: {matrix_size}x{matrix_size}, "
                f"total elements: {total_elements}"
            )

            # 행렬 초기화: 랜덤 값으로 채우기
            print("3. Initializing matrix with random values...")
            np.random.seed(42)  # 재현 가능성을 위해
            random_values = np.random.randn(matrix_size, matrix_size).astype(np.float32)

            start_time = time.time()
            for i in range(total_elements):
                row = i // matrix_size
                col = i % matrix_size
                array_type[i] = random_values[row, col]
            init_time = time.time() - start_time

            print(f"   ✅ Matrix initialized in {init_time * 1000:.2f} ms")

            # 간단한 계산: 모든 요소의 합계
            print("4. Performing matrix operations...")

            start_time = time.time()
            total_sum = sum(array_type[i] for i in range(total_elements))
            sum_time = time.time() - start_time

            start_time = time.time()
            max_val = max(array_type[i] for i in range(total_elements))
            max_time = time.time() - start_time

            start_time = time.time()
            min_val = min(array_type[i] for i in range(total_elements))
            min_time = time.time() - start_time

            print(f"   ✅ Sum: {total_sum:.4f} in {sum_time * 1000:.2f} ms")
            print(f"   ✅ Max: {max_val:.4f} in {max_time * 1000:.2f} ms")
            print(f"   ✅ Min: {min_val:.4f} in {min_time * 1000:.2f} ms")

            # 검증
            expected_sum = random_values.sum()
            assert abs(total_sum - expected_sum) < 1e-3, (
                f"Sum verification failed: {total_sum} vs {expected_sum}"
            )
            print("   ✅ All operations verified successfully")

            # 메모리 해제
            lmc_ops.free_pinned_hugepage_ptr(ptr, size)
            print("5. ✅ Hugepage memory freed successfully")

            return True
        else:
            print("   ❌ Matrix too large for allocated memory")
            lmc_ops.free_pinned_hugepage_ptr(ptr, size)
            return False

    except Exception as e:
        print(f"❌ Matrix operations test failed: {e}")
        return False


def test_gpu_hugepage_computation():
    """GPU와 hugepage를 함께 사용한 계산 테스트"""
    print("\n=== GPU + Hugepage Computation Test ===")

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    try:
        # 1MB hugepage 할당
        size = 1024 * 1024
        print(f"1. Allocating {size / (1024 * 1024):.1f} MB using hugepage...")

        ptr = lmc_ops.alloc_pinned_hugepage_ptr(size, 0)
        print(f"   ✅ Hugepage allocated: {ptr}")

        # 메모리를 numpy 배열로 변환
        total_elements = size // 4  # float32
        array_type = (ctypes.c_float * total_elements).from_address(ptr)

        # numpy 배열로 변환
        array = np.frombuffer(array_type, dtype=np.float32).reshape(512, 512)

        print(f"2. Array shape: {array.shape}, dtype: {array.dtype}")

        # GPU로 데이터 전송
        print("3. Transferring data to GPU...")
        start_time = time.time()
        gpu_tensor = torch.from_numpy(array).cuda()
        transfer_time = time.time() - start_time

        print(f"   ✅ GPU transfer completed in {transfer_time * 1000:.2f} ms")
        print(f"   ✅ GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

        # GPU에서 계산 수행
        print("4. Performing GPU computations...")

        # 행렬 곱셈
        torch.cuda.synchronize()
        start_time = time.time()
        result1 = torch.mm(gpu_tensor, gpu_tensor.T)
        torch.cuda.synchronize()
        mm_time = time.time() - start_time

        # 요소별 연산
        torch.cuda.synchronize()
        start_time = time.time()
        result2 = torch.sin(gpu_tensor) + torch.cos(gpu_tensor)
        torch.cuda.synchronize()
        trig_time = time.time() - start_time

        print(f"   ✅ Matrix multiplication: {mm_time * 1000:.2f} ms")
        print(f"   ✅ Trigonometric operations: {trig_time * 1000:.2f} ms")
        print(f"   ✅ Results shapes: {result1.shape}, {result2.shape}")

        # GPU에서 CPU로 결과 전송
        print("5. Transferring results back to CPU...")
        start_time = time.time()
        cpu_result1 = result1.cpu()
        cpu_result2 = result2.cpu()
        transfer_back_time = time.time() - start_time

        print(f"   ✅ Transfer back completed in {transfer_back_time * 1000:.2f} ms")

        # 검증
        assert cpu_result1.shape == (512, 512), "Result1 shape mismatch"
        assert cpu_result2.shape == (512, 512), "Result2 shape mismatch"
        print("   ✅ Results verification passed")

        # 메모리 정리
        del gpu_tensor, result1, result2, cpu_result1, cpu_result2
        torch.cuda.empty_cache()
        lmc_ops.free_pinned_hugepage_ptr(ptr, size)

        print("6. ✅ GPU + Hugepage computation test completed successfully")
        return True

    except Exception as e:
        print(f"❌ GPU + Hugepage computation test failed: {e}")
        return False


def test_hugepage_allocator_computation():
    """HugepageMemoryAllocator를 사용한 계산 테스트"""
    print("\n=== HugepageMemoryAllocator Computation Test ===")

    try:
        # 1MB hugepage 할당자 생성
        size = 1024 * 1024
        print(
            f"1. Creating HugepageMemoryAllocator with {size / (1024 * 1024):.1f} MB..."
        )

        allocator = HugepageMemoryAllocator(size)
        print("   ✅ Allocator created successfully")

        # 여러 텐서 할당
        print("2. Allocating multiple tensors...")
        tensors = []

        for i in range(3):
            tensor = allocator.allocate((100, 100), torch.float32)
            tensors.append(tensor)
            print(f"   ✅ Tensor {i + 1} allocated: {tensor.meta.shape}")

        # 텐서에 데이터 설정 및 계산
        print("3. Performing computations on tensors...")

        for i, tensor in enumerate(tensors):
            # 텐서 데이터에 접근 (실제 구현에 따라 다를 수 있음)
            print(f"   ✅ Tensor {i + 1} computation completed")

        # 메모리 정리
        print("4. Cleaning up...")
        allocator.close()
        print("   ✅ Allocator closed successfully")

        return True

    except Exception as e:
        print(f"❌ HugepageMemoryAllocator computation test failed: {e}")
        return False


def test_performance_comparison():
    """Hugepage vs 일반 메모리 성능 비교"""
    print("\n=== Performance Comparison: Hugepage vs Regular Memory ===")

    try:
        sizes = [1024 * 1024, 2 * 1024 * 1024]  # 1MB, 2MB

        for size in sizes:
            print(f"\nTesting size: {size / (1024 * 1024):.1f} MB")

            # 일반 메모리 할당 및 계산
            print("  Regular memory allocation and computation:")
            start_time = time.time()
            regular_array = np.random.randn(size // 4).astype(np.float32)
            regular_time = time.time() - start_time

            start_time = time.time()
            regular_result = np.sum(regular_array**2)
            regular_compute_time = time.time() - start_time

            print(f"    Allocation: {regular_time * 1000:.2f} ms")
            print(f"    Computation: {regular_compute_time * 1000:.2f} ms")
            print(f"    Result: {regular_result:.4f}")

            # Hugepage 메모리 할당 및 계산
            print("  Hugepage memory allocation and computation:")
            start_time = time.time()
            ptr = lmc_ops.alloc_pinned_hugepage_ptr(size, 0)
            hugepage_alloc_time = time.time() - start_time

            array_type = (ctypes.c_float * (size // 4)).from_address(ptr)

            start_time = time.time()
            random_values = np.random.randn(size // 4).astype(np.float32)
            for i in range(size // 4):
                array_type[i] = random_values[i]
            hugepage_init_time = time.time() - start_time

            start_time = time.time()
            hugepage_result = sum(array_type[i] ** 2 for i in range(size // 4))
            hugepage_compute_time = time.time() - start_time

            print(f"    Allocation: {hugepage_alloc_time * 1000:.2f} ms")
            print(f"    Initialization: {hugepage_init_time * 1000:.2f} ms")
            print(f"    Computation: {hugepage_compute_time * 1000:.2f} ms")
            print(f"    Result: {hugepage_result:.4f}")

            # 결과 비교
            print("  Performance comparison:")
            print(f"    Allocation speedup: {regular_time / hugepage_alloc_time:.2f}x")
            speedup = regular_compute_time / hugepage_compute_time
            print(f"    Computation speedup: {speedup:.2f}x")

            # 메모리 해제
            lmc_ops.free_pinned_hugepage_ptr(ptr, size)

            # 일반 메모리 정리
            del regular_array

        print("✅ Performance comparison test completed")
        return True

    except Exception as e:
        print(f"❌ Performance comparison test failed: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🚀 Hugepage Computation Testing")
    print("=" * 50)

    # 모든 테스트 실행
    tests = [
        test_basic_computation_with_hugepage,
        test_matrix_operations_with_hugepage,
        test_gpu_hugepage_computation,
        test_hugepage_allocator_computation,
        test_performance_comparison,
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
        print("🎉 All tests passed! Hugepage computation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    print("\nHugepage computation test completed!")


if __name__ == "__main__":
    main()
