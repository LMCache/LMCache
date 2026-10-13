// SPDX-License-Identifier: Apache-2.0

#include "stream_recorders_sycl.h"

#include <sycl/sycl.hpp>

#include <chrono>
#include <memory>
#include <mutex>

#include <c10/util/Exception.h>

namespace {

using PendingCompletion = std::pair<std::string, std::string>;
using PendingEvent = EventDrainResult::value_type;

template <typename T>
class RecordBuffer {
 public:
  void push(T&& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    items_.push_back(std::move(item));
  }

  std::vector<T> drain() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<T> result;
    result.swap(items_);
    return result;
  }

 private:
  std::mutex mutex_;
  std::vector<T> items_;
};

RecordBuffer<PendingCompletion>& completion_buffer() {
  static RecordBuffer<PendingCompletion> buffer;
  return buffer;
}

RecordBuffer<PendingEvent>& event_buffer() {
  static RecordBuffer<PendingEvent> buffer;
  return buffer;
}

// Same monotonic Unix-epoch clock as csrc/cuda/event_recorder.cpp, so
// start/end spans cannot go negative across an NTP slew.
double wall_clock_time() {
  static const auto epoch_sys = std::chrono::system_clock::now();
  static const auto epoch_steady = std::chrono::steady_clock::now();
  auto since_epoch = epoch_sys.time_since_epoch() +
                     (std::chrono::steady_clock::now() - epoch_steady);
  return std::chrono::duration<double>(since_epoch).count();
}

sycl::queue& queue_from_ptr(int64_t stream_ptr) {
  TORCH_CHECK(stream_ptr != 0, "stream recorder needs a sycl::queue pointer");
  auto& queue =
      *reinterpret_cast<sycl::queue*>(static_cast<uintptr_t>(stream_ptr));
  // Only an in-order queue orders the host_task after earlier commands.
  TORCH_CHECK(queue.is_in_order(), "stream recorder needs an in-order queue");
  return queue;
}

// Run ``record`` after all work already submitted to the queue. If the
// runtime rejects the host_task, fall back to draining the queue and
// recording inline: slower, but the ordering still holds.
template <typename Record>
void record_after_queue(int64_t stream_ptr, Record record) {
  auto& queue = queue_from_ptr(stream_ptr);
  try {
    queue.submit([&](sycl::handler& handler) { handler.host_task(record); });
  } catch (const sycl::exception& exception) {
    TORCH_WARN_ONCE("host_task submission failed (", exception.what(),
                    "); stream recorders fall back to queue.wait()");
    queue.wait_and_throw();
    record();
  }
}

}  // namespace

void record_completion_on_stream(int64_t stream_ptr, const std::string& kind,
                                 std::string payload) {
  // shared_ptr keeps the host_task functor copyable as SYCL requires.
  auto completion =
      std::make_shared<PendingCompletion>(kind, std::move(payload));
  record_after_queue(stream_ptr, [completion]() {
    completion_buffer().push(std::move(*completion));
  });
}

CompletionDrainResult drain_recorded_completions() {
  return completion_buffer().drain();
}

void record_event_on_stream(
    int64_t stream_ptr, const std::string& event_type_name,
    const std::string& session_id,
    const std::unordered_map<std::string, std::string>& str_metadata,
    const std::unordered_map<std::string, int64_t>& int_metadata) {
  auto event = std::make_shared<PendingEvent>(event_type_name, session_id, 0.0,
                                              str_metadata, int_metadata);
  record_after_queue(stream_ptr, [event]() {
    std::get<2>(*event) = wall_clock_time();
    event_buffer().push(std::move(*event));
  });
}

EventDrainResult drain_recorded_events() { return event_buffer().drain(); }
