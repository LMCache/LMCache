// SPDX-License-Identifier: Apache-2.0
//
// SYCL counterpart of csrc/cuda/completion_recorder.h and event_recorder.h.
// Each record submits a barrier to the caller's in-order queue and is held
// until that barrier completes, which gives the same ordering as
// cudaLaunchHostFunc without blocking the caller. A host_task is not used:
// SYCL refuses to signal or wait on a reusable (IPC) event behind a
// host_task that has not finished, and the MP server records IPC events on
// the same queue.
//
// Drains release every record whose barrier completed, in submission order.
// Event records also get their timestamp from a polling thread, because
// their drain runs too rarely to time them.

#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// Called WITHOUT the GIL. ``stream_ptr`` is a ``sycl::queue *``, as exposed
// by ``torch.xpu.Stream.sycl_queue``.
void record_completion_on_stream(int64_t stream_ptr, const std::string& kind,
                                 std::string payload);

using CompletionDrainResult = std::vector<std::pair<std::string, std::string>>;

CompletionDrainResult drain_recorded_completions();

// Called WITHOUT the GIL. The timestamp is taken when the barrier is seen
// complete.
void record_event_on_stream(
    int64_t stream_ptr, const std::string& event_type_name,
    const std::string& session_id,
    const std::unordered_map<std::string, std::string>& str_metadata,
    const std::unordered_map<std::string, int64_t>& int_metadata);

// (event_type_name, session_id, timestamp, str_metadata, int_metadata)
using EventDrainResult =
    std::vector<std::tuple<std::string, std::string, double,
                           std::unordered_map<std::string, std::string>,
                           std::unordered_map<std::string, int64_t>>>;

EventDrainResult drain_recorded_events();

// Stop the event timestamp thread. Registered with Python's atexit so the
// thread does not query SYCL events during runtime teardown. Later drains
// still release records.
void shutdown_stream_recorders();
