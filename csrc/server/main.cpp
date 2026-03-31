// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — Main entry point
//
// Wires CacheEngine + MessageQueueServer with concrete request handlers.
// Mirrors Python run_cache_server() from server.py.

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <execinfo.h>
#include <unistd.h>

#include "types.h"
#include "wire_protocol.h"
#include "cache_engine.h"
#include "mq_server.h"

#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>

using namespace lmcache::server;

static std::atomic<bool> g_shutdown{false};

static void signal_handler(int signum) {
  (void)signum;
  g_shutdown.store(true, std::memory_order_release);
}

static void crash_handler(int signum) {
  std::fprintf(stderr, "\n=== CRASH: signal %d (%s) ===\n", signum,
               strsignal(signum));
  void* frames[64];
  int n = backtrace(frames, 64);
  backtrace_symbols_fd(frames, n, STDERR_FILENO);
  std::fprintf(stderr, "=== END BACKTRACE ===\n");
  _exit(128 + signum);
}

// ============================================================================
// Concrete request handlers — bridge IRequestHandler to CacheEngine methods
// ============================================================================

// --- REGISTER_KV_CACHE: SYNC, no response ---
class RegisterHandler : public IRequestHandler {
 public:
  explicit RegisterHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::SYNC; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>& payloads) override {
    auto p = dec_.decode_register_payload(payloads);
    engine_.register_kv_cache(p.instance_id, p.kv_caches, p.model_name,
                              p.world_size);
    return enc_.encode_none_response();
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }

 private:
  CacheEngine& engine_;
  Decoder dec_;
  Encoder enc_;
};

// --- UNREGISTER_KV_CACHE: SYNC, no response ---
class UnregisterHandler : public IRequestHandler {
 public:
  explicit UnregisterHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::SYNC; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>& payloads) override {
    if (payloads.empty()) return enc_.encode_none_response();
    int32_t instance_id =
        dec_.decode_int_payload(payloads[0].data(), payloads[0].size());
    engine_.unregister_kv_cache(instance_id);
    return enc_.encode_none_response();
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }

 private:
  CacheEngine& engine_;
  Decoder dec_;
  Encoder enc_;
};

// --- STORE: BLOCKING, returns tuple[bytes, bool] ---
class StoreHandler : public IRequestHandler {
 public:
  explicit StoreHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::BLOCKING; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>& payloads) override {
    auto p = dec_.decode_store_payload(payloads);
    auto [ev, ok] = engine_.store(p.key, p.instance_id, p.gpu_block_ids,
                                  p.event_ipc_handle);
    return enc_.encode_store_response(ev, ok);
  }

 private:
  CacheEngine& engine_;
  Decoder dec_;
  Encoder enc_;
};

// --- RETRIEVE: BLOCKING, returns tuple[bytes, bool] ---
class RetrieveHandler : public IRequestHandler {
 public:
  explicit RetrieveHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::BLOCKING; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>& payloads) override {
    auto p = dec_.decode_retrieve_payload(payloads);
    auto [ev, ok] = engine_.retrieve(p.key, p.instance_id, p.gpu_block_ids,
                                     p.event_ipc_handle, p.skip_first_n_tokens);
    return enc_.encode_retrieve_response(ev, ok);
  }

 private:
  CacheEngine& engine_;
  Decoder dec_;
  Encoder enc_;
};

// --- LOOKUP: BLOCKING, returns int ---
class LookupHandler : public IRequestHandler {
 public:
  explicit LookupHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::BLOCKING; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>& payloads) override {
    auto p = dec_.decode_lookup_payload(payloads);
    int result = engine_.lookup(p.key, p.tp_size);
    return enc_.encode_int_response(result);
  }

 private:
  CacheEngine& engine_;
  Decoder dec_;
  Encoder enc_;
};

// --- QUERY_PREFETCH_STATUS: SYNC, returns int|None ---
class QueryPrefetchStatusHandler : public IRequestHandler {
 public:
  explicit QueryPrefetchStatusHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::SYNC; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>& payloads) override {
    auto p = dec_.decode_query_prefetch_status_payload(payloads);
    int result = engine_.query_prefetch_status(p.prefetch_job_id);
    if (result < 0) {
      return enc_.encode_optional_int_response(0, true);  // None
    }
    return enc_.encode_optional_int_response(result, false);
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }

 private:
  CacheEngine& engine_;
  Decoder dec_;
  Encoder enc_;
};

// --- QUERY_PREFETCH_LOOKUP_HITS: BLOCKING, returns int|None ---
class QueryPrefetchLookupHitsHandler : public IRequestHandler {
 public:
  explicit QueryPrefetchLookupHitsHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::BLOCKING; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>& payloads) override {
    auto p = dec_.decode_query_prefetch_lookup_hits_payload(payloads);
    int result = engine_.query_prefetch_lookup_hits(p.prefetch_job_id);
    if (result < 0) {
      return enc_.encode_optional_int_response(0, true);  // None
    }
    return enc_.encode_optional_int_response(result, false);
  }

 private:
  CacheEngine& engine_;
  Decoder dec_;
  Encoder enc_;
};

// --- FREE_LOOKUP_LOCKS: BLOCKING, no response ---
class FreeLookupLocksHandler : public IRequestHandler {
 public:
  explicit FreeLookupLocksHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::BLOCKING; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>& payloads) override {
    auto p = dec_.decode_free_lookup_locks_payload(payloads);
    engine_.free_lookup_locks(p.key, p.tp_size);
    return enc_.encode_none_response();
  }

 private:
  CacheEngine& engine_;
  Decoder dec_;
  Encoder enc_;
};

// --- END_SESSION: BLOCKING, no response ---
class EndSessionHandler : public IRequestHandler {
 public:
  explicit EndSessionHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::BLOCKING; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>& payloads) override {
    auto p = dec_.decode_end_session_payload(payloads);
    engine_.end_session(p.request_id);
    return enc_.encode_none_response();
  }

 private:
  CacheEngine& engine_;
  Decoder dec_;
  Encoder enc_;
};

// --- CLEAR: BLOCKING, no response ---
class ClearHandler : public IRequestHandler {
 public:
  explicit ClearHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::BLOCKING; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>&) override {
    engine_.clear();
    return enc_.encode_none_response();
  }

 private:
  CacheEngine& engine_;
  Encoder enc_;
};

// --- GET_CHUNK_SIZE: SYNC, returns int ---
class GetChunkSizeHandler : public IRequestHandler {
 public:
  explicit GetChunkSizeHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::SYNC; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>&) override {
    return enc_.encode_int_response(engine_.get_chunk_size());
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }

 private:
  CacheEngine& engine_;
  Encoder enc_;
};

// --- PING: BLOCKING, returns bool ---
class PingHandler : public IRequestHandler {
 public:
  explicit PingHandler(CacheEngine& e) : engine_(e) {}
  HandlerType handler_type() const override { return HandlerType::BLOCKING; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>&) override {
    return enc_.encode_bool_response(engine_.ping());
  }

 private:
  CacheEngine& engine_;
  Encoder enc_;
};

// --- NOOP: SYNC, returns str ---
class NoopHandler : public IRequestHandler {
 public:
  HandlerType handler_type() const override { return HandlerType::SYNC; }
  std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>&) override {
    return enc_.encode_string_response("OK");
  }
  std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>&) override {
    return {};
  }

 private:
  Encoder enc_;
};

// ============================================================================
// CLI argument parsing
// ============================================================================

struct ServerConfig {
  std::string host = "0.0.0.0";
  int port = 8001;
  int chunk_size = 256;
  size_t l1_capacity_gib = 8;
  int max_gpu_workers = 8;
  int max_cpu_workers = 8;
  bool hugepages = false;
  bool cuda_host_register = true;
};

static ServerConfig parse_args(int argc, char* argv[]) {
  ServerConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto get_next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "ERROR: %s requires a value\n", arg.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--host")
      cfg.host = get_next();
    else if (arg == "--port")
      cfg.port = std::stoi(get_next());
    else if (arg == "--chunk-size")
      cfg.chunk_size = std::stoi(get_next());
    else if (arg == "--l1-capacity-gib")
      cfg.l1_capacity_gib = std::stoull(get_next());
    else if (arg == "--max-workers")
      // Legacy: set both GPU and CPU workers
      cfg.max_gpu_workers = cfg.max_cpu_workers = std::stoi(get_next());
    else if (arg == "--max-gpu-workers")
      cfg.max_gpu_workers = std::stoi(get_next());
    else if (arg == "--max-cpu-workers")
      cfg.max_cpu_workers = std::stoi(get_next());
    else if (arg == "--hugepages")
      cfg.hugepages = true;
    else if (arg == "--no-cuda-host-register")
      cfg.cuda_host_register = false;
    else if (arg == "--help" || arg == "-h") {
      std::printf(
          "Usage: lmcache-server [options]\n"
          "  --host HOST             Bind address (default: 0.0.0.0)\n"
          "  --port PORT             Bind port (default: 8001)\n"
          "  --chunk-size N          Tokens per chunk (default: 256)\n"
          "  --l1-capacity-gib N     L1 slab capacity in GiB (default: 8)\n"
          "  --max-workers N         Thread pool workers (sets both GPU and "
          "CPU, default: 8)\n"
          "  --max-gpu-workers N     Affinity pool workers for STORE/RETRIEVE "
          "(default: 8)\n"
          "  --max-cpu-workers N     Normal pool workers for LOOKUP etc. "
          "(default: 8)\n"
          "  --hugepages             Use huge pages for L1 slab\n"
          "  --no-cuda-host-register Disable cudaHostRegister on L1 slab\n");
      std::exit(0);
    } else {
      std::fprintf(stderr, "WARNING: unknown arg: %s\n", arg.c_str());
    }
  }
  return cfg;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  std::signal(SIGSEGV, crash_handler);
  std::signal(SIGABRT, crash_handler);
  std::signal(SIGBUS, crash_handler);
  std::signal(SIGFPE, crash_handler);

  auto cfg = parse_args(argc, argv);

  std::printf("lmcache-server (pure C++)\n");
  std::printf("  bind: tcp://%s:%d\n", cfg.host.c_str(), cfg.port);
  std::printf("  chunk_size: %d\n", cfg.chunk_size);
  std::printf("  L1 capacity: %zu GiB\n", cfg.l1_capacity_gib);
  std::printf("  max_gpu_workers: %d\n", cfg.max_gpu_workers);
  std::printf("  max_cpu_workers: %d\n", cfg.max_cpu_workers);

  // Create L1 store config
  L1StoreConfig l1_config{};
  l1_config.capacity_bytes = cfg.l1_capacity_gib * 1024ULL * 1024 * 1024;
  l1_config.use_hugepages = cfg.hugepages;
  l1_config.cuda_host_register = cfg.cuda_host_register;
  l1_config.ttl_seconds = 300;

  // Initialize CUDA runtime + caching allocator BEFORE creating CacheEngine.
  // Critical ordering:
  //   1. CUDA runtime init (cudaFree per device)
  //   2. libtorch CUDACachingAllocator::init
  //   3. CacheEngine (creates L1 slab with cudaHostRegister)
  // If CUDA isn't initialized first, cudaHostRegister silently fails
  // and later cudaMemcpyAsync D2H hits illegal memory access.
  {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
      for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaFree(nullptr);
      }
      cudaSetDevice(0);
      c10::cuda::CUDACachingAllocator::init(device_count);
    }
    std::printf("CUDA initialized: %d device(s)\n", device_count);
  }

  // Create cache engine (L1 only for now; L2 adapter = nullptr)
  CacheEngine engine(cfg.chunk_size, l1_config, nullptr);

  // Create ZMQ server (max_workers param unused now, pools assigned below)
  std::string bind_url = "tcp://" + cfg.host + ":" + std::to_string(cfg.port);
  MessageQueueServer server(bind_url, 0);

  // Register all handlers (matching Python run_cache_server on origin/dev)
  server.add_handler(RequestType::REGISTER_KV_CACHE,
                     std::make_unique<RegisterHandler>(engine));
  server.add_handler(RequestType::UNREGISTER_KV_CACHE,
                     std::make_unique<UnregisterHandler>(engine));
  server.add_handler(RequestType::STORE,
                     std::make_unique<StoreHandler>(engine));
  server.add_handler(RequestType::RETRIEVE,
                     std::make_unique<RetrieveHandler>(engine));
  server.add_handler(RequestType::LOOKUP,
                     std::make_unique<LookupHandler>(engine));
  server.add_handler(RequestType::QUERY_PREFETCH_STATUS,
                     std::make_unique<QueryPrefetchStatusHandler>(engine));
  server.add_handler(RequestType::QUERY_PREFETCH_LOOKUP_HITS,
                     std::make_unique<QueryPrefetchLookupHitsHandler>(engine));
  server.add_handler(RequestType::FREE_LOOKUP_LOCKS,
                     std::make_unique<FreeLookupLocksHandler>(engine));
  server.add_handler(RequestType::END_SESSION,
                     std::make_unique<EndSessionHandler>(engine));
  server.add_handler(RequestType::CLEAR,
                     std::make_unique<ClearHandler>(engine));
  server.add_handler(RequestType::GET_CHUNK_SIZE,
                     std::make_unique<GetChunkSizeHandler>(engine));
  server.add_handler(RequestType::PING, std::make_unique<PingHandler>(engine));
  server.add_handler(RequestType::NOOP, std::make_unique<NoopHandler>());

  // Affinity pool for GPU-bound handlers (STORE, RETRIEVE)
  // Same identity → same worker thread, eliminates per-instance GPU locks
  server.add_affinity_thread_pool({RequestType::STORE, RequestType::RETRIEVE},
                                  cfg.max_gpu_workers);

  // Normal pool for CPU-bound handlers
  server.add_normal_thread_pool(
      {RequestType::LOOKUP, RequestType::QUERY_PREFETCH_LOOKUP_HITS,
       RequestType::FREE_LOOKUP_LOCKS, RequestType::END_SESSION,
       RequestType::CLEAR, RequestType::PING},
      cfg.max_cpu_workers);

  std::printf("Starting server...\n");
  server.start();
  std::printf("LMCache C++ server is running on %s\n", bind_url.c_str());

  // Sleep loop until signal
  while (!g_shutdown.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  std::printf("\nShutting down...\n");
  server.close();
  engine.close();
  std::printf("Done.\n");
  return 0;
}
