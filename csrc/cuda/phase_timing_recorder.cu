// SPDX-License-Identifier: Apache-2.0

#include "phase_timing_recorder.cuh"

void destroy_phase_timing_events(cudaEvent_t start, cudaEvent_t end) {
  if (start != nullptr) {
    cudaEventDestroy(start);
  }
  if (end != nullptr) {
    cudaEventDestroy(end);
  }
  (void)cudaGetLastError();
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

std::deque<PhaseTimingRecord> PhaseTimingRecorder::take_all() {
  std::deque<PhaseTimingRecord> taken;
  std::lock_guard<std::mutex> lock(mutex_);
  taken.swap(pending_);
  return taken;
}

void PhaseTimingRecorder::requeue_front(
    const std::deque<PhaseTimingRecord>& records) {
  if (records.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.insert(pending_.begin(), records.begin(), records.end());
  evict_until_below_cap(0);
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
  // Take the whole queue and run every CUDA call unlocked, so executor
  // threads pushing new records never wait behind a batch of event queries.
  PhaseTimingRecorder& recorder = PhaseTimingRecorder::instance();
  std::deque<PhaseTimingRecord> pending = recorder.take_all();

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
    // Reached on success, on an elapsed-time failure, and on any query error
    // other than not-ready; the destroy helper clears the error state so a
    // timing failure never surfaces on an unrelated CUDA call.
    destroy_phase_timing_events(record.start, record.end);
  }

  recorder.requeue_front(not_ready);
  return samples;
}
