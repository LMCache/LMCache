// SPDX-License-Identifier: Apache-2.0

#include <pthread.h>

#include <cstdio>

#include "phase_timing_recorder.cuh"

namespace {
thread_local int locks_held = 0;
int destroyed = 0;
int destroyed_under_lock = 0;
}  // namespace

// Only calls from the linked recorder objects are wrapped, not libcudart's
// internal mutexes. The CUDA operations themselves run normally.
extern "C" int __real_pthread_mutex_lock(pthread_mutex_t*);
extern "C" int __wrap_pthread_mutex_lock(pthread_mutex_t* mutex) {
  int result = __real_pthread_mutex_lock(mutex);
  if (result == 0) {
    ++locks_held;
  }
  return result;
}

extern "C" int __real_pthread_mutex_unlock(pthread_mutex_t*);
extern "C" int __wrap_pthread_mutex_unlock(pthread_mutex_t* mutex) {
  int result = __real_pthread_mutex_unlock(mutex);
  if (result == 0) {
    --locks_held;
  }
  return result;
}

extern "C" cudaError_t __real_cudaEventDestroy(cudaEvent_t);
extern "C" cudaError_t __wrap_cudaEventDestroy(cudaEvent_t event) {
  ++destroyed;
  if (locks_held != 0) {
    ++destroyed_under_lock;
  }
  return __real_cudaEventDestroy(event);
}

int main() {
  constexpr int capacity = 8192;
  PhaseTimer timer(true, nullptr, 0, 0, capacity + 1, "eviction");
  for (int i = 0; i <= capacity; ++i) {
    auto section = timer.section(TransferPhase::STAGING, i + 1, nullptr);
  }
  timer.commit();
  if (destroyed != 2 || destroyed_under_lock != 0) {
    std::fprintf(stderr, "Eviction destroyed %d events, %d under lock\n",
                 destroyed, destroyed_under_lock);
    return 1;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return 1;
  }
  auto samples = pop_completed_phase_timings();
  if (samples.size() != capacity || samples.front().nbytes != 2 ||
      samples.back().nbytes != capacity + 1 ||
      destroyed != 2 * (capacity + 1) + 1 || destroyed_under_lock != 0) {
    std::fprintf(stderr, "Unexpected samples or event count after drain\n");
    return 1;
  }
  return 0;
}
