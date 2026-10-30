// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace record_probe {

enum class Framing : std::uint8_t {
  kRaw = 1,
  kGzip = 2,
};

// Read a hex file, ignoring whitespace, and return its decoded record bytes.
// Only the frozen V1 "hello" record for the requested framing is accepted.
// Throws std::runtime_error on unreadable, malformed, or mismatching input.
std::vector<std::uint8_t> read_fixed_fixture(const std::string& path,
                                             Framing framing);

}  // namespace record_probe
