// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include "mem_kernels.cuh"
#include "cachegen_kernels.cuh"
#include "pos_kernels.cuh"
#include "mem_alloc.h"
#include "utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <chrono>
#include <algorithm>

namespace {

inline void check_cuda(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
  }
}

}  // namespace

struct Chunk {
  uint64_t src;
  size_t len;
};

struct ChunkEx {
  uint64_t src;
  size_t len;
  uint64_t dst;
  size_t dst_off;
};

void memcpy_h2d_async(uint64_t dst_dev_ptr, uint64_t src_host_ptr,
                      size_t nbytes) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  cudaStream_t s = stream.stream();

  void* dst = reinterpret_cast<void*>(dst_dev_ptr);
  const void* src = reinterpret_cast<const void*>(src_host_ptr);

  auto err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, s);
  check_cuda(err, "cudaMemcpyAsync(H2D) failed");
}

void pack_and_h2d(uint64_t dst_dev_ptr, uint64_t staging_host_ptr,
                  const std::vector<Chunk>& chunks, size_t total_bytes) {
  char* dst_h = reinterpret_cast<char*>(staging_host_ptr);
  size_t off = 0;
  for (const auto& c : chunks) {
    const char* src = reinterpret_cast<const char*>(c.src);
    std::memcpy(dst_h + off, src, c.len);
    off += c.len;
  }
  auto stream = c10::cuda::getCurrentCUDAStream();
  check_cuda(
      cudaMemcpyAsync(reinterpret_cast<void*>(dst_dev_ptr),
                      reinterpret_cast<void*>(staging_host_ptr), total_bytes,
                      cudaMemcpyHostToDevice, stream.stream()),
      "cudaMemcpyAsync(H2D)");
}

struct StagingBuf {
  void* host = nullptr;
  size_t bytes = 0;
  cudaEvent_t done = nullptr;
  bool in_flight = false;
};

struct StreamStaging {
  std::vector<StagingBuf> ring;
  size_t idx = 0;
  size_t bytes = 0;
};

static std::unordered_map<uint64_t, StreamStaging> g_staging;
static std::mutex g_staging_mu;
static int g_staging_spin_us = 0;

static StagingBuf* acquire_buf(cudaStream_t s) {
  uint64_t key = reinterpret_cast<uint64_t>(s);
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::microseconds(g_staging_spin_us);

  for (;;) {
    {
      std::lock_guard<std::mutex> lock(g_staging_mu);
      auto it = g_staging.find(key);
      if (it == g_staging.end() || it->second.ring.empty()) {
        throw std::runtime_error("staging not initialized for stream");
      }
      auto& ss = it->second;
      size_t n = ss.ring.size();
      for (size_t tries = 0; tries < n; ++tries) {
        auto& b = ss.ring[ss.idx];
        if (!b.in_flight) {
          b.in_flight = true;
          return &b;
        }
        if (cudaEventQuery(b.done) == cudaSuccess) {
          b.in_flight = false;
          b.in_flight = true;
          return &b;
        }
        ss.idx = (ss.idx + 1) % n;
      }
    }
    if (g_staging_spin_us > 0 && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(g_staging_mu);
      auto& ss = g_staging.at(key);
      auto& b = ss.ring[ss.idx];
      check_cuda(cudaEventSynchronize(b.done), "cudaEventSynchronize(staging)");
      b.in_flight = false;
      b.in_flight = true;
      return &b;
    }
  }
}

static void mark_inflight(StagingBuf* b, cudaStream_t s) {
  check_cuda(cudaEventRecord(b->done, s), "cudaEventRecord(staging.done)");
}

void init_stream_staging(size_t bytes, int N) {
  if (N <= 0) N = 1;
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  uint64_t key = reinterpret_cast<uint64_t>(s);
  std::lock_guard<std::mutex> lock(g_staging_mu);
  auto& ss = g_staging[key];
  if (!ss.ring.empty()) return;
  ss.bytes = bytes;
  ss.ring.resize(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    void* p = nullptr;
    check_cuda(cudaHostAlloc(&p, bytes, cudaHostAllocPortable), "cudaHostAlloc");
    ss.ring[i].host = p;
    ss.ring[i].bytes = bytes;
    check_cuda(
        cudaEventCreateWithFlags(&ss.ring[i].done, cudaEventDisableTiming),
        "cudaEventCreateWithFlags");
    ss.ring[i].in_flight = false;
  }
}

void set_staging_spin_us(int spin_us) {
  if (spin_us < 0) spin_us = 0;
  g_staging_spin_us = spin_us;
}

void pack_and_h2d_group(uint64_t dst_dev_ptr, const std::vector<Chunk>& chunks,
                        size_t total_bytes) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  StagingBuf* b = acquire_buf(stream);
  if (b->bytes < total_bytes) {
    throw std::runtime_error("staging buffer too small for group");
  }
  char* dst_h = reinterpret_cast<char*>(b->host);
  size_t off = 0;
  for (const auto& c : chunks) {
    const char* src = reinterpret_cast<const char*>(c.src);
    std::memcpy(dst_h + off, src, c.len);
    off += c.len;
  }
  check_cuda(cudaMemcpyAsync(reinterpret_cast<void*>(dst_dev_ptr), b->host,
                             total_bytes, cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(H2D)");
  mark_inflight(b, stream);
}

void build_pack_and_h2d_batches(uint64_t dst_base,
                                const std::vector<ChunkEx>& in, size_t gran_min,
                                size_t gran_max, int lookahead) {
  cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
  std::vector<ChunkEx> v = in;
  std::sort(v.begin(), v.end(), [](const ChunkEx& a, const ChunkEx& b) {
    return a.dst_off < b.dst_off;
  });

  size_t i = 0;
  while (i < v.size()) {
    if (v[i].len >= gran_min) {
      const void* src = reinterpret_cast<const void*>(v[i].src);
      void* dst = reinterpret_cast<void*>(dst_base + v[i].dst_off);
      check_cuda(cudaMemcpyAsync(dst, src, v[i].len, cudaMemcpyHostToDevice, s),
                 "cudaMemcpyAsync(H2D,bypass)");
      ++i;
      continue;
    }

    size_t start = i;
    size_t group_bytes = 0;
    size_t j = i;
    while (j < v.size() && (int)(j - i) < lookahead) {
      if (j == i || (v[j - 1].dst_off + v[j - 1].len == v[j].dst_off)) {
        if (group_bytes + v[j].len <= gran_max) {
          group_bytes += v[j].len;
          ++j;
        } else {
          break;
        }
      } else {
        break;
      }
    }
    size_t end = j;
    if (end == start) {
      end = start + 1;
      group_bytes = v[start].len;
    }

    StagingBuf* b = acquire_buf(s);
    if (b->bytes < group_bytes) {
      throw std::runtime_error("staging buffer too small for batch");
    }
    char* dst_h = reinterpret_cast<char*>(b->host);
    size_t off = 0;
    for (size_t k = start; k < end; ++k) {
      const char* src = reinterpret_cast<const char*>(v[k].src);
      std::memcpy(dst_h + off, src, v[k].len);
      off += v[k].len;
    }
    void* dst_dev = reinterpret_cast<void*>(dst_base + v[start].dst_off);
    check_cuda(cudaMemcpyAsync(dst_dev, b->host, group_bytes,
                               cudaMemcpyHostToDevice, s),
               "cudaMemcpyAsync(H2D,batch)");
    mark_inflight(b, s);

    i = end;
  }
}

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("encode_fast_new", &encode_cuda_new);
  m.def("decode_fast_new", &decode_cuda_new);
  m.def("decode_fast_prefsum", &decode_cuda_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("get_gpu_pci_bus_id", &get_gpu_pci_bus_id);

  // Added: CUDA IO bindings (thin coalescing helpers)
  py::class_<Chunk>(m, "Chunk")
      .def(py::init<>())
      .def_readwrite("src", &Chunk::src)
      .def_readwrite("len", &Chunk::len);
  py::class_<ChunkEx>(m, "ChunkEx")
      .def(py::init<>())
      .def_readwrite("src", &ChunkEx::src)
      .def_readwrite("len", &ChunkEx::len)
      .def_readwrite("dst", &ChunkEx::dst)
      .def_readwrite("dst_off", &ChunkEx::dst_off);

  m.def("memcpy_h2d_async", &memcpy_h2d_async,
        "Single cudaMemcpyAsync H2D on current torch CUDA stream");
  m.def("pack_and_h2d", &pack_and_h2d,
        "Pack host chunks into staging and issue single H2D on current stream");
  m.def("init_stream_staging", &init_stream_staging,
        "Initialize per-stream staging ring with pinned host buffers");
  m.def("set_staging_spin_us", &set_staging_spin_us,
        "Set optional spin (microseconds) for acquiring staging buffers");
  m.def("pack_and_h2d_group", &pack_and_h2d_group,
        "Acquire ring buffer, pack chunks, and issue one H2D with event");
  m.def("build_pack_and_h2d_batches", &build_pack_and_h2d_batches,
        "Build groups and launch pack+H2D per group with bypass for big chunks");
}
