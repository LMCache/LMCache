// SPDX-License-Identifier: Apache-2.0
//
// KV transfer between the serving engine's GPU memory and host memory is
// one of the most performance-critical paths in LMCache. This recorder
// times each batch step of a transfer plan in two phases:
//   1. gather/scatter kernel: paged KV blocks <-> GPU staging buffers
//   2. DMA copies: GPU staging buffers <-> pinned host memory
// The metrics layer derives per-phase throughput from the drained samples.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <deque>
#include <mutex>
#include <tuple>
#include <vector>

// Timed sections of execute_object_group_transfer.
enum class TransferPhase : int {
  KERNEL = 0,   // gather/scatter kernel launches (paged blocks <-> staging)
  STAGING = 1,  // host<->device DMA staging copies
};

// One timed section.
struct PhaseTimingRecord {
  cudaEvent_t start;  // recorded on the transfer stream at section start
  cudaEvent_t end;    // recorded at section end; complete => sample ready
  int phase;          // TransferPhase value
  int direction;      // TransferDirection value
  int device_index;   // CUDA device the section ran on
  int64_t nbytes;     // staged payload bytes of the step
};

// Destroy an event pair (either handle may be null) and clear the CUDA
// error state, so a timing failure never surfaces on an unrelated call.
void destroy_phase_timing_events(cudaEvent_t start, cudaEvent_t end);

// Destroys the accumulated records on scope exit unless disarmed, so a
// transfer that failed mid-plan publishes no partial samples. The
// destructor only destroys events and cannot throw.
struct PhaseTimingDiscardGuard {
  std::vector<PhaseTimingRecord>& records;
  bool armed = true;
  ~PhaseTimingDiscardGuard() {
    if (armed) {
      for (auto& record : records) {
        destroy_phase_timing_events(record.start, record.end);
      }
    }
  }
};

// Process-wide buffer of in-flight timing records. Constructed on first
// use; records still pending at process exit are deliberately leaked (the
// CUDA runtime may already be shut down).
class PhaseTimingRecorder {
 public:
  static PhaseTimingRecorder& instance();

  // Enqueue one call's records under a single lock acquisition.
  void push_batch(const std::vector<PhaseTimingRecord>& records);

  // Hand over the whole queue so the caller can run CUDA calls unlocked.
  std::deque<PhaseTimingRecord> take_all();

  // Put not-yet-completed records back at the front (they are the oldest).
  void requeue_front(const std::deque<PhaseTimingRecord>& records);

 private:
  PhaseTimingRecorder() = default;

  // Bound on queued records; the oldest are evicted past this.
  static constexpr size_t kMaxPending = 8192;

  // Caller must hold mutex_. Evicts the oldest records until `headroom`
  // more fit under the cap.
  void evict_until_below_cap(size_t headroom);

  std::mutex mutex_;
  std::deque<PhaseTimingRecord> pending_;  // guarded by mutex_
};

/**
 * Pop completed gather/DMA phase timing samples.
 *
 * Returns the finished CUDA event pairs recorded by
 * execute_object_group_transfer; unfinished pairs stay queued.
 *
 * @return One tuple per finished section:
 *         (phase, direction, device_index, elapsed_ms, nbytes), with phase a
 *         TransferPhase value, direction a TransferDirection value, and
 *         nbytes the step's staged payload (shared by both phases).
 */
std::vector<std::tuple<int, int, int, double, int64_t>>
pop_completed_phase_timings();
