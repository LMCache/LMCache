// SPDX-License-Identifier: Apache-2.0

#include "stream_recorders_sycl.h"

#include <sycl/sycl.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>

#include <c10/util/Exception.h>

namespace {

using PendingCompletion = std::pair<std::string, std::string>;
using PendingEvent = EventDrainResult::value_type;

// How often the timestamp thread re-polls while event records are pending.
constexpr auto kEventPollInterval = std::chrono::microseconds(100);

// Same monotonic Unix-epoch clock as csrc/cuda/event_recorder.cpp, so
// start/end spans cannot go negative across an NTP slew.
double wall_clock_time() {
  static const auto epoch_sys = std::chrono::system_clock::now();
  static const auto epoch_steady = std::chrono::steady_clock::now();
  auto since_epoch = epoch_sys.time_since_epoch() +
                     (std::chrono::steady_clock::now() - epoch_steady);
  return std::chrono::duration<double>(since_epoch).count();
}

bool marker_complete(const sycl::event& marker) {
  try {
    return marker.get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete;
  } catch (const sycl::exception& exception) {
    // Holding the record forever would leak whatever it releases.
    TORCH_WARN_ONCE("stream recorder could not query a queue marker (",
                    exception.what(), "); releasing its record");
    return true;
  }
}

// Records wait for their queue marker, then become drainable in submission
// order. Markers on one in-order queue complete in order; records on
// different queues are released as each queue reaches them.
template <typename T>
class OrderedRecords {
 public:
  void push(sycl::event marker, T&& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.emplace_back(std::move(marker), std::move(item));
  }

  // Returns how many records are still waiting.
  template <typename OnReady>
  size_t release_ready(OnReady on_ready) {
    std::lock_guard<std::mutex> lock(mutex_);
    return release_ready_locked(on_ready);
  }

  template <typename OnReady>
  std::vector<T> drain(OnReady on_ready) {
    std::lock_guard<std::mutex> lock(mutex_);
    release_ready_locked(on_ready);
    std::vector<T> result;
    result.swap(ready_);
    return result;
  }

 private:
  template <typename OnReady>
  size_t release_ready_locked(OnReady& on_ready) {
    std::vector<std::pair<sycl::event, T>> waiting;
    for (auto& entry : pending_) {
      if (marker_complete(entry.first)) {
        on_ready(entry.second);
        ready_.push_back(std::move(entry.second));
      } else {
        waiting.push_back(std::move(entry));
      }
    }
    pending_.swap(waiting);
    return pending_.size();
  }

  std::mutex mutex_;
  std::vector<std::pair<sycl::event, T>> pending_;
  std::vector<T> ready_;
};

// Never destroyed: the timestamp thread may still reference them during
// static destruction if shutdown_stream_recorders was not called.
OrderedRecords<PendingCompletion>& completion_records() {
  static auto* records = new OrderedRecords<PendingCompletion>();
  return *records;
}

OrderedRecords<PendingEvent>& event_records() {
  static auto* records = new OrderedRecords<PendingEvent>();
  return *records;
}

void stamp_event(PendingEvent& event) {
  std::get<2>(event) = wall_clock_time();
}

void leave_completion(PendingCompletion&) {}

// Observability events carry the time their queue position was reached.
// Drains run every ~100 ms, so a thread polls the markers while event
// records are pending. Completions need no timestamp and are released by
// their drain, which runs every few milliseconds.
class EventStamper {
 public:
  void wake() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {
      return;  // drain still releases the records, with a later timestamp
    }
    ++generation_;
    if (!thread_.joinable()) {
      thread_ = std::thread([this]() { run(); });
    }
    cv_.notify_one();
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
      cv_.notify_one();
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  void run() {
    std::unique_lock<std::mutex> lock(mutex_);
    size_t seen = 0;
    bool poll = false;
    while (true) {
      if (poll) {
        cv_.wait_for(lock, kEventPollInterval, [this]() { return stopped_; });
      } else {
        cv_.wait(lock,
                 [this, &seen]() { return stopped_ || generation_ != seen; });
      }
      if (stopped_) {
        return;
      }
      // A record pushed after this snapshot bumps the generation, so it
      // is either released below or wakes the next wait.
      seen = generation_;
      lock.unlock();
      poll = event_records().release_ready(stamp_event) > 0;
      lock.lock();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  size_t generation_ = 0;
  bool stopped_ = false;
};

EventStamper& event_stamper() {
  static auto* stamper = new EventStamper();
  return *stamper;
}

sycl::queue& queue_from_ptr(int64_t stream_ptr) {
  TORCH_CHECK(stream_ptr != 0, "stream recorder needs a sycl::queue pointer");
  auto& queue =
      *reinterpret_cast<sycl::queue*>(static_cast<uintptr_t>(stream_ptr));
  // Only an in-order queue orders the marker after earlier commands.
  TORCH_CHECK(queue.is_in_order(), "stream recorder needs an in-order queue");
  return queue;
}

// Return an event that completes after all work already submitted to the
// queue. A barrier is a native backend command, unlike a host_task, so it
// does not stop later reusable (IPC) events from being signaled or waited
// on this queue. If the runtime rejects it, drain the queue instead and
// return a default event, which is already complete.
sycl::event marker_after_queue(int64_t stream_ptr) {
  auto& queue = queue_from_ptr(stream_ptr);
  try {
    return queue.ext_oneapi_submit_barrier();
  } catch (const sycl::exception& exception) {
    TORCH_WARN_ONCE("queue barrier submission failed (", exception.what(),
                    "); stream recorders fall back to queue.wait()");
    queue.wait_and_throw();
    return sycl::event();
  }
}

}  // namespace

void record_completion_on_stream(int64_t stream_ptr, const std::string& kind,
                                 std::string payload) {
  completion_records().push(marker_after_queue(stream_ptr),
                            PendingCompletion(kind, std::move(payload)));
}

CompletionDrainResult drain_recorded_completions() {
  return completion_records().drain(leave_completion);
}

void record_event_on_stream(
    int64_t stream_ptr, const std::string& event_type_name,
    const std::string& session_id,
    const std::unordered_map<std::string, std::string>& str_metadata,
    const std::unordered_map<std::string, int64_t>& int_metadata) {
  event_records().push(marker_after_queue(stream_ptr),
                       PendingEvent(event_type_name, session_id, 0.0,
                                    str_metadata, int_metadata));
  event_stamper().wake();
}

EventDrainResult drain_recorded_events() {
  return event_records().drain(stamp_event);
}

void shutdown_stream_recorders() { event_stamper().stop(); }
