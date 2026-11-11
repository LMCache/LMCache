// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Ascend CANN runtime (>= 9.0, provides aclrtLaunchHostFunc)
#include "acl/acl.h"

using lmcache_stream_t = aclrtStream;

// ---------------------------------------------------------------------------
// PendingEvent — lightweight struct held in a lock-free-ish buffer.
// All fields are pure C++ (no Python objects) so the ACL host callback
// can write the timestamp without touching the GIL.
// ---------------------------------------------------------------------------

struct PendingEvent {
  std::string event_type_name;  // e.g. "mp.store.start"
  std::string session_id;
  double timestamp;  // wall-clock, set by host callback
  std::unordered_map<std::string, std::string> str_metadata;
  std::unordered_map<std::string, int64_t> int_metadata;
};

// ---------------------------------------------------------------------------
// EventRecorder — global singleton that buffers events from ACL callbacks.
// ---------------------------------------------------------------------------

class EventRecorder {
 public:
  static EventRecorder& instance();

  // Called from the ACL host callback (no GIL held).
  // Takes ownership of *event, moves it into the buffer, then deletes it.
  void push(PendingEvent* event);

  // Called from Python (GIL held) to drain all buffered events.
  std::vector<PendingEvent> drain();

 private:
  EventRecorder() = default;
  std::mutex mutex_;
  std::vector<PendingEvent> buffer_;
};

// ---------------------------------------------------------------------------
// Free functions exposed via pybind11
// ---------------------------------------------------------------------------

// Schedule an event recording on an NPU stream.  The host callback stamps
// the wall-clock time and pushes to the global EventRecorder.
// Called WITHOUT the GIL (py::call_guard<py::gil_scoped_release>).
void record_event_on_stream(
    int64_t stream_ptr, const std::string& event_type_name,
    const std::string& session_id,
    const std::unordered_map<std::string, std::string>& str_metadata,
    const std::unordered_map<std::string, int64_t>& int_metadata);

// Drain all buffered events.  Returns a list of tuples:
//   (event_type_name, session_id, timestamp, str_metadata, int_metadata)
using DrainResult =
    std::vector<std::tuple<std::string, std::string, double,
                           std::unordered_map<std::string, std::string>,
                           std::unordered_map<std::string, int64_t>>>;

DrainResult drain_recorded_events();
