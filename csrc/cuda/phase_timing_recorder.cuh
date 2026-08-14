// SPDX-License-Identifier: Apache-2.0
//
// Gather/DMA phase timing for the object-group transfer plan executor.
//
// The executor (mp_mem_kernels.cu) brackets each staging/kernel section with
// a CUDA event pair when the caller requests it, and hands the finished
// pairs to the process-wide PhaseTimingRecorder. Python later drains the
// completed samples via pop_completed_phase_timings().

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <deque>
#include <mutex>
#include <tuple>
#include <vector>

// Timed sections of execute_object_group_transfer, reported by
// pop_completed_phase_timings().
enum class TransferPhase : int {
  KERNEL = 0,   // gather/scatter kernel launches (paged blocks <-> staging)
  STAGING = 1,  // host<->device DMA staging copies
};

// One timed section: a recorded CUDA event pair plus its labels.
struct PhaseTimingRecord {
  cudaEvent_t start;
  cudaEvent_t end;
  int phase;      // TransferPhase value (kernel / staging)
  int direction;  // TransferDirection value
  int device_index;
  int64_t nbytes;
};

// Destroy an event pair (either handle may be null) and swallow the CUDA
// error state afterwards. Timing is best-effort instrumentation: a failure
// here must never surface on an unrelated CUDA call later in this thread.
void destroy_phase_timing_events(cudaEvent_t start, cudaEvent_t end);

// Scope guard over one call's accumulated records: unless disarmed, destroys
// their events on scope exit so sections completed before a failing step are
// dropped rather than published — a transfer that threw did not run to
// completion, and its partial phase samples would misrepresent the work
// actually done. The destructor only destroys events, so it cannot throw
// while unwinding.
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

// Process-wide buffer of in-flight timing records.
//
// Constructed on first use, so no static-initialization ordering applies.
// Teardown only releases the
// queue's memory: the CUDA events of records still pending at exit are
// deliberately left alone rather than destroyed, since the CUDA runtime may
// already be shut down by then and the process is going away regardless.
class PhaseTimingRecorder {
 public:
  static PhaseTimingRecorder& instance();

  // Enqueue one call's records under a single lock acquisition; the executor
  // hot path therefore locks once per transfer call, not once per section.
  void push_batch(const std::vector<PhaseTimingRecord>& records);

  // Hand the whole queue to the caller so it can run CUDA calls unlocked.
  std::deque<PhaseTimingRecord> take_all();

  // Return records whose events have not completed. They predate anything
  // pushed while the recorder was unlocked, so they go back at the front to
  // keep eviction oldest-first.
  void requeue_front(const std::deque<PhaseTimingRecord>& records);

 private:
  PhaseTimingRecorder() = default;

  // Backstop, not a tuning knob: normal operation pops after every request
  // and stays far below the cap. Sized well above the peak in-flight sample
  // count (devices x concurrent transfers x steps x 2 phases) while staying
  // cheap when full (a few MB of records plus their events).
  static constexpr size_t kMaxPending = 8192;

  // Caller must hold mutex_. Drops the oldest records until at least
  // `headroom` more can be pushed without exceeding the cap.
  void evict_until_below_cap(size_t headroom);

  std::mutex mutex_;
  std::deque<PhaseTimingRecord> pending_;
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
