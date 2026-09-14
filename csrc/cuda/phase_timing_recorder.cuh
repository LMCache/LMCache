// SPDX-License-Identifier: Apache-2.0
//
// KV transfer between the serving engine's GPU memory and host memory is
// one of the most performance-critical paths in LMCache. This recorder
// times each batch step of a transfer plan in two phases:
//   1. gather/scatter kernel: paged KV blocks <-> GPU staging buffers
//   2. DMA copies: GPU staging buffers <-> pinned host memory
// The metrics layer derives per-phase throughput from the popped samples;
// the tracing layer uses their wall-clock bounds and session id.

#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Timed sections of execute_object_group_transfer.
enum class TransferPhase : int {
  KERNEL = 0,   // gather/scatter kernel launches (paged blocks <-> staging)
  STAGING = 1,  // host<->device DMA staging copies
};

// State shared by every section of one executor call, including the
// GPU-to-wall-clock reference: an event recorded on the stream before the
// first section, and the EventRecorder wall clock at which the GPU reached
// it (stamped by a host callback on the same stream).
struct PhaseTimingContext {
  std::string session_id;        // request the transfer served
  int direction;                 // TransferDirection value
  int device_index;              // CUDA device the transfer ran on
  cudaEvent_t anchor = nullptr;  // null if it could not be recorded
  std::atomic<double> anchor_wall_time_s{0.0};  // 0 => not stamped yet
  std::atomic<int> pending_records{0};  // records still referencing anchor
};

// One timed section in flight.
struct PhaseTimingRecord {
  TransferPhase phase;
  int64_t nbytes;     // bytes this section moved
  cudaEvent_t start;  // recorded on the transfer stream at section start
  cudaEvent_t end;    // recorded at section end; complete => sample ready
  std::shared_ptr<PhaseTimingContext> ctx;
};

// One popped sample. The wall-clock bounds are 0.0 if the call's anchor
// could not be recorded.
struct PhaseTimingSample {
  int phase;               // TransferPhase value
  int direction;           // TransferDirection value
  int device_index;        // CUDA device the section ran on
  double elapsed_ms;       // GPU time between the section's events
  int64_t nbytes;          // bytes this section moved
  std::string session_id;  // request the transfer served
  double start_time_s;     // EventRecorder wall clock at section start
  double end_time_s;       // ... and at section end
};

// Collects the timed sections of one executor call and hands them to the
// PhaseTimingRecorder on commit(). Without commit() -- an exception, or an
// early return -- the destructor discards everything, so a transfer that
// failed mid-plan publishes no partial samples.
class PhaseTimer {
 public:
  // A timed section of work: the constructor records the start event on
  // the section's stream, the destructor records the end and files the
  // sample with the timer. Inert when timing is off, nbytes <= 0, or an
  // event call fails; destruction during exception unwinding discards
  // the sample.
  class Section {
   public:
    Section(PhaseTimer& timer, TransferPhase phase, int64_t nbytes,
            cudaStream_t stream);
    ~Section();
    Section(const Section&) = delete;
    Section& operator=(const Section&) = delete;

   private:
    PhaseTimer* timer_;    // owning timer; nullptr => inert section
    cudaStream_t stream_;  // stream this section's events are recorded on
    cudaEvent_t start_;
    cudaEvent_t end_;
    TransferPhase phase_;
    int64_t nbytes_;
    int base_exceptions_;  // std::uncaught_exceptions() at construction
  };

  PhaseTimer(bool enabled, cudaStream_t stream, int direction, int device_index,
             size_t max_sections, std::string session_id);
  ~PhaseTimer();
  PhaseTimer(const PhaseTimer&) = delete;
  PhaseTimer& operator=(const PhaseTimer&) = delete;

  // `stream` must be the stream this section's work is actually enqueued
  // on (not necessarily the timer's, e.g. under a CUDAStreamGuard).
  Section section(TransferPhase phase, int64_t nbytes, cudaStream_t stream) {
    return Section(*this, phase, nbytes, stream);
  }

  // Hand the collected records to the recorder.
  void commit();

 private:
  // Takes ownership of a completed section's events. Never allocates
  // (records_ is reserved up front), so it is safe to call from a
  // destructor.
  void add(const PhaseTimingRecord& record);

  bool enabled_;
  std::shared_ptr<PhaseTimingContext> ctx_;  // shared by this call's records
  std::vector<PhaseTimingRecord> records_;   // completed sections
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
  // unfinished records stay queued.
  std::vector<PhaseTimingSample> pop_completed();

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
 * @return One PhaseTimingSample per finished section. nbytes is the bytes
 *         that section moved (staged payload for staging sections;
 *         skip-aware launch bytes for kernel sections).
 */
std::vector<PhaseTimingSample> pop_completed_phase_timings();
