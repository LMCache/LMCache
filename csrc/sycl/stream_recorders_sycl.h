// SPDX-License-Identifier: Apache-2.0
//
// SYCL counterpart of csrc/cuda/completion_recorder.h and event_recorder.h.
// A host_task submitted to the caller's in-order queue runs only after all
// earlier commands on that queue finish, which gives the same ordering as
// cudaLaunchHostFunc without blocking the caller. The task runs on a SYCL
// runtime thread without the GIL and only appends to a mutex-guarded
// buffer; Python drains the buffer.

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

// Called WITHOUT the GIL. The timestamp is taken when the host_task runs.
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
