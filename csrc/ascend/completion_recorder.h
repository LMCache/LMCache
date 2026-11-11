// SPDX-License-Identifier: Apache-2.0
//
// Ascend NPU host-callback buffer for stream-completion records.
// Ported from lmcache csrc/cuda/completion_recorder.h; the only device
// dependency is aclrtLaunchHostFunc (CANN >= 9.0), which mirrors
// cudaLaunchHostFunc. The callback runs on an ACL-internal thread without
// the GIL; Python drains the buffer and dispatches to a handler keyed by
// ``kind``.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// Ascend CANN runtime
#include "acl/acl.h"

using lmcache_completion_stream_t = aclrtStream;

struct PendingCompletion {
  std::string kind;     // dispatch key, e.g. "finish_write"
  std::string payload;  // opaque encoded bytes (e.g. msgpack)
};

class CompletionRecorder {
 public:
  static CompletionRecorder& instance();
  // Takes ownership of the heap-allocated PendingCompletion.
  void push(std::unique_ptr<PendingCompletion> completion);
  std::vector<std::unique_ptr<PendingCompletion>> drain();

 private:
  CompletionRecorder() = default;
  std::mutex mutex_;
  std::vector<std::unique_ptr<PendingCompletion>> buffer_;
};

// Schedule a completion record. Called WITHOUT the GIL.
void record_completion_on_stream(int64_t stream_ptr, const std::string& kind,
                                 std::string payload);

using CompletionDrainResult = std::vector<std::pair<std::string, std::string>>;

CompletionDrainResult drain_recorded_completions();
