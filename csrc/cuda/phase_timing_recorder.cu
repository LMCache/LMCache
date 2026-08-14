// SPDX-License-Identifier: Apache-2.0

#include "phase_timing_recorder.cuh"

#include <exception>

namespace {

// Destroy an event pair (either handle may be null) and clear the CUDA
// error state, so a timing failure never surfaces on an unrelated call.
void destroy_phase_timing_events(cudaEvent_t start, cudaEvent_t end) {
  if (start != nullptr) {
    cudaEventDestroy(start);
  }
  if (end != nullptr) {
    cudaEventDestroy(end);
  }
  (void)cudaGetLastError();
}

}  // namespace

PhaseTimer::PhaseTimer(bool enabled, cudaStream_t stream,
                                 int direction, int device_index,
                                 size_t max_sections)
    : enabled_(enabled),
      stream_(stream),
      direction_(direction),
      device_index_(device_index) {
  if (enabled_) {
    records_.reserve(max_sections);
  }
}

PhaseTimer::~PhaseTimer() {
  // Destroyed without commit(): the transfer failed or returned early, so
  // the collected samples are abandoned.
  if (committed_) {
    return;
  }
  for (auto& record : records_) {
    destroy_phase_timing_events(record.start, record.end);
  }
}

void PhaseTimer::commit() {
  // Set the flag first: if push_batch throws, the destructor must not
  // destroy events whose copies may already sit in the recorder.
  committed_ = true;
  PhaseTimingRecorder::instance().push_batch(records_);
}

void PhaseTimer::add(const PhaseTimingRecord& record) {
  // records_ was reserved for every possible section up front; the guard
  // keeps this path allocation-free and therefore unable to throw.
  if (records_.size() == records_.capacity()) {
    destroy_phase_timing_events(record.start, record.end);
    return;
  }
  records_.push_back(record);
}

PhaseTimer::Section::Section(PhaseTimer& timer, TransferPhase phase,
                             int64_t nbytes)
    : timer_(nullptr),
      start_(nullptr),
      end_(nullptr),
      phase_(phase),
      nbytes_(nbytes),
      base_exceptions_(std::uncaught_exceptions()) {
  // Stay inert (timer_ == nullptr) when timing is off or the section moves
  // no data.
  if (!timer.enabled_ || nbytes <= 0) {
    return;
  }

  // Open the interval: create the event pair and stamp its start onto the
  // transfer stream.
  const bool opened = cudaEventCreate(&start_) == cudaSuccess &&
                      cudaEventCreate(&end_) == cudaSuccess &&
                      cudaEventRecord(start_, timer.stream_) == cudaSuccess;
  if (!opened) {
    // Degrade to an inert section; the transfer itself is unaffected.
    destroy_phase_timing_events(start_, end_);
    start_ = nullptr;
    end_ = nullptr;
    return;
  }

  timer_ = &timer;
}

PhaseTimer::Section::~Section() {
  // Inert section: nothing was opened, nothing to close.
  if (timer_ == nullptr) {
    return;
  }

  // An exception is unwinding through the scope: the section did not run
  // to completion, so its sample would be garbage.
  if (std::uncaught_exceptions() > base_exceptions_) {
    destroy_phase_timing_events(start_, end_);
    return;
  }

  // Close the interval: stamp the end onto the stream. A never-recorded
  // event queries as "complete", so on failure the pair must be dropped
  // rather than kept as a bogus sample.
  if (cudaEventRecord(end_, timer_->stream_) != cudaSuccess) {
    destroy_phase_timing_events(start_, end_);
    return;
  }

  timer_->add({start_, end_, static_cast<int>(phase_), timer_->direction_,
               timer_->device_index_, nbytes_});
}

PhaseTimingRecorder& PhaseTimingRecorder::instance() {
  static PhaseTimingRecorder recorder;
  return recorder;
}

void PhaseTimingRecorder::push_batch(
    const std::vector<PhaseTimingRecord>& records) {
  if (records.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& record : records) {
    evict_until_below_cap(1);
    pending_.push_back(record);
  }
}

std::vector<std::tuple<int, int, int, double, int64_t>>
PhaseTimingRecorder::pop_completed() {
  // Take the whole queue, then run every CUDA call unlocked so executor
  // pushes never wait.
  std::deque<PhaseTimingRecord> pending;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending.swap(pending_);
  }

  std::vector<std::tuple<int, int, int, double, int64_t>> samples;
  std::deque<PhaseTimingRecord> not_ready;
  for (auto& record : pending) {
    const cudaError_t status = cudaEventQuery(record.end);
    if (status == cudaErrorNotReady) {
      (void)cudaGetLastError();  // clear cudaErrorNotReady
      not_ready.push_back(record);
      continue;
    }
    if (status == cudaSuccess) {
      float elapsed_ms = 0.0f;
      if (cudaEventElapsedTime(&elapsed_ms, record.start, record.end) ==
          cudaSuccess) {
        samples.emplace_back(record.phase, record.direction,
                             record.device_index,
                             static_cast<double>(elapsed_ms), record.nbytes);
      }
    }
    destroy_phase_timing_events(record.start, record.end);
  }

  // Not-yet-completed records go back at the front: they predate anything
  // pushed while the queue was unlocked, keeping eviction oldest-first.
  if (!not_ready.empty()) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.insert(pending_.begin(), not_ready.begin(), not_ready.end());
    evict_until_below_cap(0);
  }
  return samples;
}

void PhaseTimingRecorder::evict_until_below_cap(size_t headroom) {
  while (pending_.size() + headroom > kMaxPending) {
    PhaseTimingRecord& oldest = pending_.front();
    destroy_phase_timing_events(oldest.start, oldest.end);
    pending_.pop_front();
  }
}

std::vector<std::tuple<int, int, int, double, int64_t>>
pop_completed_phase_timings() {
  return PhaseTimingRecorder::instance().pop_completed();
}
