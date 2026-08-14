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

// Collects the timed sections of one executor call and hands them to the
// PhaseTimingRecorder on commit(). Without commit() -- an exception, or an
// early return -- the destructor discards everything, so a transfer that
// failed mid-plan publishes no partial samples.
//
// `stream` must be the stream the transfer work is enqueued on; sections
// record their events on it.
class PhaseTimer {
 public:
  // Times one section; the scope of this object IS the timed interval
  // (same grammar as CUDAGuard / RECORD_FUNCTION). Inert when timing is
  // disabled, nbytes <= 0, or any event operation fails; discards instead
  // of publishing when destroyed during exception unwinding.
  class Section {
   public:
    Section(PhaseTimer& timer, TransferPhase phase, int64_t nbytes);
    ~Section();
    Section(const Section&) = delete;
    Section& operator=(const Section&) = delete;

   private:
    PhaseTimer* timer_;  // owning timer; nullptr => inert section
    cudaEvent_t start_;
    cudaEvent_t end_;
    TransferPhase phase_;
    int64_t nbytes_;
    int base_exceptions_;  // std::uncaught_exceptions() at construction
  };

  PhaseTimer(bool enabled, cudaStream_t stream, int direction, int device_index,
             size_t max_sections);
  ~PhaseTimer();
  PhaseTimer(const PhaseTimer&) = delete;
  PhaseTimer& operator=(const PhaseTimer&) = delete;

  Section section(TransferPhase phase, int64_t nbytes) {
    return Section(*this, phase, nbytes);
  }

  // Hand the collected records to the recorder.
  void commit();

 private:
  // Takes ownership of a completed section's events. Never allocates
  // (records_ is reserved up front), so it is safe to call from a
  // destructor.
  void add(const PhaseTimingRecord& record);

  bool enabled_;
  cudaStream_t stream_;  // stream the sections record their events on
  int direction_;        // TransferDirection value
  int device_index_;
  std::vector<PhaseTimingRecord> records_;  // completed sections
  bool committed_ = false;
};

// Process-wide buffer of in-flight timing records. Constructed on first
// use; records still pending at process exit are deliberately leaked (the
// CUDA runtime may already be shut down).
class PhaseTimingRecorder {
 public:
  static PhaseTimingRecorder& instance();

  // Enqueue one call's records under a single lock acquisition.
  void push_batch(const std::vector<PhaseTimingRecord>& records);

  // Measure and remove the records whose end event has completed;
  // unfinished records stay queued. See pop_completed_phase_timings()
  // for the tuple layout.
  std::vector<std::tuple<int, int, int, double, int64_t>> pop_completed();

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
 * Pop completed gather/DMA phase timing samples
 * (PhaseTimingRecorder::pop_completed, exposed for the pybind layer).
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
