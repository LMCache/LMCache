// SPDX-License-Identifier: Apache-2.0
// Observation points of the HF3FS SDK double in hf3fs_mock.cpp.
#pragma once

#include <atomic>
#include <string>

namespace hf3fs_mock {
// Path hf3fs_extract_mount_point reports for every input.
extern std::string mount_point;
// SDK rings and buffers created, destroyed, and destroyed when not live.
extern std::atomic<int> created;
extern std::atomic<int> destroyed;
extern std::atomic<int> duplicate_destroys;
}  // namespace hf3fs_mock
