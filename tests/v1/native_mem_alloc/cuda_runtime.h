// SPDX-License-Identifier: Apache-2.0
#pragma once

// Minimal CUDA runtime surface for CPU-only allocator regression tests.
#include <cstddef>

enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
constexpr unsigned int cudaHostAllocPortable = 1;
constexpr unsigned int cudaHostAllocMapped = 2;
constexpr unsigned int cudaHostAllocWriteCombined = 4;
constexpr unsigned int cudaHostRegisterPortable = 1;
constexpr unsigned int cudaHostRegisterMapped = 2;

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void* ptr);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t error);
