// SPDX-License-Identifier: Apache-2.0

#pragma once

// x86 cache-maintenance primitives for cross-host visibility.

#include <cstddef>
#include <cstdint>

namespace lmcache::dax_coordinated_l1 {

struct CpuVisibilityProfile {
  bool is_x86_64;
  bool has_clflush;
  bool has_clflushopt;
  std::uint32_t cache_line_bytes;
};

struct VisibilityRange {
  void* address;
  std::size_t length;
};

CpuVisibilityProfile query_cpu_visibility_profile();
void validate_cpu_visibility_profile();
void publish_range(void* address, std::size_t length);
void refresh_range(void* address, std::size_t length);
void publish_range_bulk(void* address, std::size_t length);
void publish_ranges_bulk(const VisibilityRange* ranges, std::size_t count);

}  // namespace lmcache::dax_coordinated_l1
