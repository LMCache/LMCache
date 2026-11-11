// SPDX-License-Identifier: Apache-2.0

#include "completion_recorder.h"

#include <utility>

CompletionRecorder& CompletionRecorder::instance() {
  static CompletionRecorder recorder;
  return recorder;
}

void CompletionRecorder::push(std::unique_ptr<PendingCompletion> completion) {
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_.push_back(std::move(completion));
}

std::vector<std::unique_ptr<PendingCompletion>> CompletionRecorder::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::unique_ptr<PendingCompletion>> result;
  result.swap(buffer_);
  return result;
}

// ACL host callback — runs on an ACL-internal thread, no GIL.
// Same restriction as CUDA host functions: never call any aclrt* API from
// this callback; std::chrono / mutex / vector only.
static void completion_host_callback(void* data) {
  // Adopt the raw pointer back into a unique_ptr.
  std::unique_ptr<PendingCompletion> completion(
      static_cast<PendingCompletion*>(data));
  CompletionRecorder::instance().push(std::move(completion));
}

void record_completion_on_stream(int64_t stream_ptr, const std::string& kind,
                                 std::string payload) {
  auto completion = std::make_unique<PendingCompletion>(
      PendingCompletion{kind, std::move(payload)});
  if (stream_ptr == 0) {
    // Null stream: no stream dependency — run the callback inline so the
    // record lands in the buffer immediately.
    completion_host_callback(completion.release());
    return;
  }
  auto stream = reinterpret_cast<lmcache_completion_stream_t>(
      static_cast<uintptr_t>(stream_ptr));
  // Pass ownership through the runtime as a raw pointer; the host callback
  // re-adopts it. Reclaim if the launch itself fails.
  PendingCompletion* raw = completion.release();
  auto err = aclrtLaunchHostFunc(stream, completion_host_callback, raw);
  if (err != ACL_SUCCESS) {
    delete raw;
  }
}

CompletionDrainResult drain_recorded_completions() {
  auto completions = CompletionRecorder::instance().drain();
  CompletionDrainResult result;
  result.reserve(completions.size());
  for (auto& c : completions) {
    result.emplace_back(std::move(c->kind), std::move(c->payload));
  }
  return result;
}
