// SPDX-License-Identifier: Apache-2.0

#include "event_recorder.h"

#include <chrono>
#include <utility>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double wall_clock_time() {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// EventRecorder
// ---------------------------------------------------------------------------

EventRecorder& EventRecorder::instance() {
  static EventRecorder recorder;
  return recorder;
}

void EventRecorder::push(PendingEvent* event) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.push_back(std::move(*event));
  }
  delete event;
}

std::vector<PendingEvent> EventRecorder::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PendingEvent> result;
  result.swap(buffer_);
  return result;
}

// ---------------------------------------------------------------------------
// CUDA host callback — runs on a CUDA driver thread, no GIL.
// ---------------------------------------------------------------------------

static void
#ifndef USE_ROCM
    CUDART_CB
#endif
    event_host_callback(void* data) {
  auto* event = static_cast<PendingEvent*>(data);
  event->timestamp = wall_clock_time();
  EventRecorder::instance().push(event);
}

// ---------------------------------------------------------------------------
// Free functions for pybind11
// ---------------------------------------------------------------------------

void record_event_on_stream(
    int64_t cuda_stream_ptr, const std::string& event_type_name,
    const std::string& session_id,
    const std::unordered_map<std::string, std::string>& str_metadata,
    const std::unordered_map<std::string, int64_t>& int_metadata) {
  auto* event = new PendingEvent{
      event_type_name, session_id, 0.0, str_metadata, int_metadata,
  };

  auto stream = reinterpret_cast<lmcache_stream_t>(
      static_cast<uintptr_t>(cuda_stream_ptr));
  LMCACHE_LAUNCH_HOST_FUNC(stream, event_host_callback, event);
}

DrainResult drain_recorded_events() {
  auto events = EventRecorder::instance().drain();
  DrainResult result;
  result.reserve(events.size());
  for (auto& e : events) {
    result.emplace_back(std::move(e.event_type_name), std::move(e.session_id),
                        e.timestamp, std::move(e.str_metadata),
                        std::move(e.int_metadata));
  }
  return result;
}
