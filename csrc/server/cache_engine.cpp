// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — CacheEngine orchestrator implementation
//
// Mirrors Python MPCacheEngine from server.py: wires GPUContext, L1Store,
// TokenHasher, SessionManager together with the async prefetch pool.

#include "cache_engine.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <future>
#include <queue>
#include <stdexcept>
#include <thread>

#include <cuda_runtime_api.h>

#include "tensor_bridge.h"

// Kernel entry points (mp_mem_kernels.cuh includes mem_kernels.cuh)
#include "mp_mem_kernels.cuh"

// ATen CUDA guards for device/stream scoping
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAEvent.h>

namespace lmcache {
namespace server {

// ============================================================================
// Simple thread pool for async prefetch
// ============================================================================

struct CacheEngine::ThreadPool {
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex mutex;
  std::condition_variable cv;
  bool stop = false;

  explicit ThreadPool(int num_workers) {
    for (int i = 0; i < num_workers; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty()) return;
            task = std::move(tasks.front());
            tasks.pop();
          }
          task();
        }
      });
    }
  }

  template <typename F>
  std::future<int> submit(F&& f) {
    auto task = std::make_shared<std::packaged_task<int()>>(std::forward<F>(f));
    auto fut = task->get_future();
    {
      std::lock_guard<std::mutex> lk(mutex);
      tasks.push([task]() { (*task)(); });
    }
    cv.notify_one();
    return fut;
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mutex);
      stop = true;
    }
    cv.notify_all();
    for (auto& w : workers) {
      if (w.joinable()) w.join();
    }
  }
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

CacheEngine::CacheEngine(int chunk_size, const L1StoreConfig& l1_config,
                         std::unique_ptr<L2Adapter> l2_adapter)
    : chunk_size_(chunk_size),
      l1_store_(L1Store::create(l1_config)),
      l2_adapter_(std::move(l2_adapter)),
      token_hasher_(chunk_size),
      session_manager_(token_hasher_),
      prefetch_pool_(std::make_unique<ThreadPool>(8)) {
  std::fprintf(stderr,
               "CacheEngine: initialized (chunk_size=%d, L1=%.1f GiB)\n",
               chunk_size_,
               static_cast<double>(l1_config.capacity_bytes) /
                   (1024.0 * 1024.0 * 1024.0));
}

CacheEngine::~CacheEngine() { close(); }

// ============================================================================
// GPU registration
// ============================================================================

void CacheEngine::register_kv_cache(
    int instance_id, const std::vector<CudaIpcTensorDesc>& kv_caches,
    const std::string& model_name, int world_size) {
  auto ctx = std::make_unique<GPUContext>(kv_caches, chunk_size_);
  gpu_contexts_[instance_id] = std::move(ctx);
  gpu_context_meta_[instance_id] = {model_name, world_size};
  std::fprintf(stderr,
               "CacheEngine: registered KV cache for instance %d "
               "(%d layers, model=%s, ws=%d)\n",
               instance_id, gpu_contexts_[instance_id]->num_layers(),
               model_name.c_str(), world_size);
}

void CacheEngine::unregister_kv_cache(int instance_id) {
  auto it = gpu_contexts_.find(instance_id);
  if (it != gpu_contexts_.end()) {
    gpu_contexts_.erase(it);
    gpu_context_meta_.erase(instance_id);
    std::fprintf(stderr, "CacheEngine: unregistered KV cache for instance %d\n",
                 instance_id);
  } else {
    std::fprintf(stderr, "WARNING: CacheEngine: no KV cache for instance %d\n",
                 instance_id);
  }
}

// ============================================================================
// Layout descriptor helper
// ============================================================================

MemoryLayoutDesc CacheEngine::find_layout_desc(const std::string& model_name,
                                               int world_size) const {
  for (const auto& [gpu_id, meta] : gpu_context_meta_) {
    if (meta.first == model_name && meta.second == world_size) {
      auto it = gpu_contexts_.find(gpu_id);
      if (it != gpu_contexts_.end()) {
        auto shape = it->second->get_kv_buffer_shape(chunk_size_);
        DType dt = it->second->dtype();
        return MemoryLayoutDesc{{ShapeDesc{shape}}, {dt}};
      }
    }
  }
  return {};  // empty = not found
}

// ============================================================================
// Store
// ============================================================================

std::pair<std::vector<uint8_t>, bool> CacheEngine::store(
    const IPCCacheEngineKey& key, int instance_id,
    const std::vector<int32_t>& gpu_block_ids,
    const std::vector<uint8_t>& event_ipc_handle) {
  // Get session and compute hashes
  auto session = session_manager_.get_or_create(key.request_id);
  session->set_tokens(key.token_ids);
  auto chunk_hashes = session->get_hashes(key.start, key.end);

  auto obj_keys = ipc_key_to_object_keys(key.model_name, key.world_size,
                                         key.worker_id, chunk_hashes);

  // Find GPU context
  auto ctx_it = gpu_contexts_.find(instance_id);
  if (ctx_it == gpu_contexts_.end()) {
    std::fprintf(stderr, "ERROR: KV cache not registered for instance %d\n",
                 instance_id);
    cudaEvent_t ev;
    auto ev_bytes = create_ipc_event(ev);
    cudaEventDestroy(ev);
    return {ev_bytes, false};
  }
  auto& gpu_ctx = ctx_it->second;

  int blocks_per_chunk = chunk_size_ / gpu_ctx->block_size();

  // Set device and stream guards
  c10::cuda::CUDAGuard device_guard(gpu_ctx->device_index());
  at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromExternal(
      gpu_ctx->stream(), gpu_ctx->device_index());
  at::cuda::CUDAStreamGuard stream_guard(torch_stream);

  // Stage all block_ids to GPU once
  at::Tensor all_block_ids_gpu = gpu_ctx->stage_block_ids(gpu_block_ids);

  // Wait for vLLM to finish writing
  cudaEvent_t vllm_event = open_ipc_event(event_ipc_handle.data());
  cudaStreamWaitEvent(gpu_ctx->stream(), vllm_event, 0);
  cudaEventDestroy(vllm_event);

  // Get layout and reserve L1 write slots
  auto layout = find_layout_desc(key.model_name, key.world_size);
  auto reserved = l1_store_->reserve_write(obj_keys, layout, "new");

  // Transfer: GPU → tmp_buffer → L1 slab (via the block-level CUDA kernel)
  // NOTE: Store is not batched because some obj_keys may be skipped
  // (not in reserved_dict), making block_ids non-contiguous.
  {
    std::lock_guard<std::mutex> lk(gpu_ctx->transfer_lock());

    for (size_t idx = 0; idx < obj_keys.size(); ++idx) {
      auto it = reserved.find(obj_keys[idx]);
      if (it == reserved.end()) continue;

      // Slice block_ids for this chunk
      int start_block = static_cast<int>(idx) * blocks_per_chunk;
      int end_block = start_block + blocks_per_chunk;
      at::Tensor chunk_block_ids_gpu =
          all_block_ids_gpu.slice(0, start_block, end_block);

      at::Tensor tmp_buf = gpu_ctx->get_tmp_gpu_buffer(chunk_size_);

      // GPU KV cache → tmp_buffer (D2H staging) via block-level kernel
      multi_layer_block_kv_transfer(
          gpu_ctx->kv_pointers(),
          {reinterpret_cast<int64_t>(tmp_buf.data_ptr())}, chunk_block_ids_gpu,
          at::Device(at::kCUDA, gpu_ctx->device_index()),
          TransferDirection::D2H, gpu_ctx->shape_desc(), chunk_size_,
          static_cast<GPUKVFormat>(gpu_ctx->gpu_kv_format()), 0);

      // tmp_buffer → L1 slab (async memcpy D2H)
      auto& slab_ref = it->second;
      cudaMemcpyAsync(slab_ref.data, tmp_buf.data_ptr(), slab_ref.size_bytes,
                      cudaMemcpyDeviceToHost, gpu_ctx->stream());
    }
  }

  // Record completion event
  cudaEvent_t done_event;
  auto done_bytes = create_ipc_event(done_event);
  cudaEventRecord(done_event, gpu_ctx->stream());

  // Synchronize and finish write
  cudaStreamSynchronize(gpu_ctx->stream());
  std::vector<ObjectKey> written_keys;
  for (const auto& [k, _] : reserved) {
    written_keys.push_back(k);
  }
  l1_store_->finish_write(written_keys);

  cudaEventDestroy(done_event);

  if (!reserved.empty()) {
    std::fprintf(stderr, "CacheEngine: stored %zu chunks (%d tokens)\n",
                 reserved.size(),
                 static_cast<int>(reserved.size()) * chunk_size_);
  }

  return {done_bytes, true};
}

// ============================================================================
// Retrieve
// ============================================================================

std::pair<std::vector<uint8_t>, bool> CacheEngine::retrieve(
    const IPCCacheEngineKey& key, int instance_id,
    const std::vector<int32_t>& gpu_block_ids,
    const std::vector<uint8_t>& event_ipc_handle, int skip_first_n_tokens) {
  auto session = session_manager_.get_or_create(key.request_id);
  session->set_tokens(key.token_ids);
  auto chunk_hashes = session->get_hashes(key.start, key.end);

  auto obj_keys = ipc_key_to_object_keys(key.model_name, key.world_size,
                                         key.worker_id, chunk_hashes);

  auto ctx_it = gpu_contexts_.find(instance_id);
  if (ctx_it == gpu_contexts_.end()) {
    std::fprintf(stderr, "ERROR: KV cache not registered for instance %d\n",
                 instance_id);
    cudaEvent_t ev;
    auto ev_bytes = create_ipc_event(ev);
    cudaEventDestroy(ev);
    return {ev_bytes, false};
  }
  auto& gpu_ctx = ctx_it->second;

  int blocks_per_chunk = chunk_size_ / gpu_ctx->block_size();
  static constexpr int kBatchSize = 4;

  // Set device and stream guards — use high-priority stream for retrieve
  c10::cuda::CUDAGuard device_guard(gpu_ctx->device_index());
  at::cuda::CUDAStream torch_hp_stream = at::cuda::getStreamFromExternal(
      gpu_ctx->high_priority_stream(), gpu_ctx->device_index());
  at::cuda::CUDAStreamGuard stream_guard(torch_hp_stream);

  // Stage all block_ids to GPU once
  at::Tensor all_block_ids_gpu = gpu_ctx->stage_block_ids(gpu_block_ids);

  // Wait for vLLM
  cudaEvent_t vllm_event = open_ipc_event(event_ipc_handle.data());
  cudaStreamWaitEvent(gpu_ctx->high_priority_stream(), vllm_event, 0);
  cudaEventDestroy(vllm_event);

  // Read from L1
  auto read_refs = l1_store_->reserve_read(obj_keys);
  if (read_refs.empty()) {
    std::fprintf(stderr, "ERROR: retrieve: no keys found in L1\n");
    cudaEvent_t ev;
    auto ev_bytes = create_ipc_event(ev);
    cudaEventRecord(ev, gpu_ctx->high_priority_stream());
    cudaEventDestroy(ev);
    return {ev_bytes, false};
  }

  // Transfer: L1 slab → tmp_buffer → GPU KV cache (batched, block-level)
  {
    std::lock_guard<std::mutex> lk(gpu_ctx->transfer_lock());

    // Process in batches of kBatchSize chunks
    for (size_t batch_start = 0; batch_start < obj_keys.size();
         batch_start += kBatchSize) {
      size_t batch_end = std::min(batch_start + kBatchSize, obj_keys.size());
      int actual_batch_size = static_cast<int>(batch_end - batch_start);

      int chunk_start_tok = static_cast<int>(batch_start) * chunk_size_;
      int chunk_end_tok = static_cast<int>(batch_end) * chunk_size_;

      int effective_start = std::max(chunk_start_tok, skip_first_n_tokens);
      if (effective_start >= chunk_end_tok) continue;

      int skip_tokens_in_batch =
          std::max(0, std::min(effective_start - chunk_start_tok,
                               chunk_size_ * kBatchSize - 1));
      int skip_blocks_in_batch = skip_tokens_in_batch / gpu_ctx->block_size();

      // Slice block_ids for this batch
      int block_start = static_cast<int>(batch_start) * blocks_per_chunk;
      int block_end = static_cast<int>(batch_end) * blocks_per_chunk;
      at::Tensor batch_block_ids_gpu =
          all_block_ids_gpu.slice(0, block_start, block_end);

      // Get batched tmp buffers
      auto tmp_bufs =
          gpu_ctx->get_tmp_gpu_buffer_batched(chunk_size_, actual_batch_size);

      // H2D memcpy for each chunk in the batch
      for (int bi = 0; bi < actual_batch_size; ++bi) {
        size_t idx = batch_start + bi;
        auto it = read_refs.find(obj_keys[idx]);
        if (it == read_refs.end()) continue;

        auto& slab_ref = it->second;
        cudaMemcpyAsync(tmp_bufs[bi].data_ptr(), slab_ref.data,
                        slab_ref.size_bytes, cudaMemcpyHostToDevice,
                        gpu_ctx->high_priority_stream());
      }

      // Build lmcache_objects_ptrs for the block-level kernel
      std::vector<int64_t> lmcache_ptrs;
      lmcache_ptrs.reserve(actual_batch_size);
      for (int bi = 0; bi < actual_batch_size; ++bi) {
        lmcache_ptrs.push_back(
            reinterpret_cast<int64_t>(tmp_bufs[bi].data_ptr()));
      }

      // tmp_buffers → GPU KV cache via block-level kernel
      multi_layer_block_kv_transfer(
          gpu_ctx->kv_pointers(), lmcache_ptrs, batch_block_ids_gpu,
          at::Device(at::kCUDA, gpu_ctx->device_index()),
          TransferDirection::H2D, gpu_ctx->shape_desc(), chunk_size_,
          static_cast<GPUKVFormat>(gpu_ctx->gpu_kv_format()),
          skip_blocks_in_batch);
    }
  }

  // Record completion
  cudaEvent_t done_event;
  auto done_bytes = create_ipc_event(done_event);
  cudaEventRecord(done_event, gpu_ctx->high_priority_stream());

  // Schedule finish_read after GPU completes
  cudaStreamSynchronize(gpu_ctx->high_priority_stream());
  l1_store_->finish_read(obj_keys);

  cudaEventDestroy(done_event);

  std::fprintf(stderr, "CacheEngine: retrieved %zu chunks (%d tokens)\n",
               obj_keys.size(),
               static_cast<int>(obj_keys.size()) * chunk_size_);

  return {done_bytes, true};
}

// ============================================================================
// Lookup (async two-phase)
// ============================================================================

int CacheEngine::lookup(const IPCCacheEngineKey& key, int tp_size) {
  auto layout = find_layout_desc(key.model_name, key.world_size);
  if (layout.shapes.empty()) {
    std::fprintf(stderr, "ERROR: lookup: no GPU context for model=%s ws=%d\n",
                 key.model_name.c_str(), key.world_size);
    std::lock_guard<std::mutex> lk(prefetch_job_lock_);
    int job_id = next_prefetch_job_id_++;
    prefetch_jobs_[job_id] = PrefetchJob{
        PrefetchHandle{-1, key.request_id, 0, 0, 0.0}, 1, key.request_id};
    return job_id;
  }

  auto chunk_hashes = token_hasher_.compute_chunk_hashes(key.token_ids);
  if (chunk_hashes.empty()) {
    std::lock_guard<std::mutex> lk(prefetch_job_lock_);
    int job_id = next_prefetch_job_id_++;
    prefetch_jobs_[job_id] = PrefetchJob{
        PrefetchHandle{-1, key.request_id, 0, 0, 0.0}, 1, key.request_id};
    return job_id;
  }

  auto obj_keys = ipc_key_to_object_keys(key.model_name, key.world_size,
                                         key.worker_id, chunk_hashes);

  // L1 prefix lookup
  int64_t l1_hits = l1_store_->prefix_lookup(obj_keys);

  // Register prefetch job
  std::lock_guard<std::mutex> lk(prefetch_job_lock_);
  int job_id = next_prefetch_job_id_++;
  prefetch_jobs_[job_id] =
      PrefetchJob{PrefetchHandle{-1, key.request_id, l1_hits,
                                 static_cast<int64_t>(obj_keys.size()), 0.0},
                  key.world_size, key.request_id};
  return job_id;
}

// ============================================================================
// Query prefetch lookup hits
// ============================================================================

int CacheEngine::query_prefetch_lookup_hits(int prefetch_job_id) {
  std::lock_guard<std::mutex> lk(prefetch_job_lock_);
  auto it = prefetch_jobs_.find(prefetch_job_id);
  if (it == prefetch_jobs_.end()) {
    return -1;  // not found or already consumed
  }

  // For L1-only mode, hits are known immediately
  int found_count = static_cast<int>(it->second.handle.l1_prefix_hit_count) /
                    it->second.world_size;
  return found_count;
}

// ============================================================================
// Query prefetch status
// ============================================================================

int CacheEngine::query_prefetch_status(int prefetch_job_id) {
  std::lock_guard<std::mutex> lk(prefetch_job_lock_);
  auto it = prefetch_jobs_.find(prefetch_job_id);
  if (it == prefetch_jobs_.end()) {
    return -1;  // not found
  }

  // For L1-only mode, prefetch is always immediately complete
  int found_count = static_cast<int>(it->second.handle.l1_prefix_hit_count) /
                    it->second.world_size;
  prefetch_jobs_.erase(it);
  return found_count;
}

// ============================================================================
// Free lookup locks
// ============================================================================

void CacheEngine::free_lookup_locks(const IPCCacheEngineKey& key, int tp_size) {
  // Release L1 read locks
  auto chunk_hashes =
      token_hasher_.compute_chunk_hashes(key.token_ids, key.start, key.end);
  if (chunk_hashes.empty()) return;

  auto obj_keys = ipc_key_to_object_keys(key.model_name, key.world_size,
                                         key.worker_id, chunk_hashes);

  int extra_count = compute_extra_count(tp_size, key.world_size);
  l1_store_->finish_read(obj_keys, extra_count);
}

// ============================================================================
// Utility
// ============================================================================

void CacheEngine::end_session(const std::string& request_id) {
  session_manager_.remove(request_id);
}

void CacheEngine::clear() {
  std::lock_guard<std::mutex> lk(lock_);
  l1_store_->memcheck();
  l1_store_->clear(true);
  l1_store_->memcheck();
}

void CacheEngine::close() {
  if (l2_adapter_) {
    l2_adapter_->close();
  }
  prefetch_pool_.reset();
  gpu_contexts_.clear();
  std::fprintf(stderr, "CacheEngine: closed\n");
}

// ============================================================================
// Prefetch load helper (for L2, used by prefetch pool)
// ============================================================================

int CacheEngine::run_prefetch_load(
    const std::string& request_id, const std::vector<ObjectKey>& remaining_keys,
    const MemoryLayoutDesc& layout_desc,
    std::unordered_map<int, storage_manager::Bitmap*> l2_lookup_results,
    int extra_count) {
  // L2 integration TBD — this is the async L2→L1 transfer path
  // For now, return 0 (no L2 loads)
  (void)request_id;
  (void)remaining_keys;
  (void)layout_desc;
  (void)l2_lookup_results;
  (void)extra_count;
  return 0;
}

}  // namespace server
}  // namespace lmcache
