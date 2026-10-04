// SPDX-License-Identifier: Apache-2.0

#include "dax_coordinated_l1/visibility_x86.h"

#include <immintrin.h>

#include <stdexcept>

#include <cpuid.h>

#include "dax_coordinated_l1/layout.h"

namespace lmcache::dax_coordinated_l1 {
namespace {

inline void compiler_barrier() { asm volatile("" ::: "memory"); }

inline std::uintptr_t align_down_64(std::uintptr_t value) {
  return value & ~(static_cast<std::uintptr_t>(kCacheLineBytes) - 1);
}

inline std::uintptr_t align_up_64(std::uintptr_t value) {
  return (value + kCacheLineBytes - 1) &
         ~(static_cast<std::uintptr_t>(kCacheLineBytes) - 1);
}

void clflush_range(void* address, std::size_t length) {
  if (length == 0) {
    return;
  }
  const auto begin = align_down_64(reinterpret_cast<std::uintptr_t>(address));
  const auto end =
      align_up_64(reinterpret_cast<std::uintptr_t>(address) + length);
  for (auto line = begin; line < end; line += kCacheLineBytes) {
    _mm_clflush(reinterpret_cast<const void*>(line));
  }
}

__attribute__((target("clflushopt"))) void clflushopt_range(
    void* address, std::size_t length) {
  if (length == 0) {
    return;
  }
  const auto begin = align_down_64(reinterpret_cast<std::uintptr_t>(address));
  const auto end =
      align_up_64(reinterpret_cast<std::uintptr_t>(address) + length);
  for (auto line = begin; line < end; line += kCacheLineBytes) {
    _mm_clflushopt(reinterpret_cast<void*>(line));
  }
}

}  // namespace

CpuVisibilityProfile query_cpu_visibility_profile() {
  CpuVisibilityProfile profile{};
  profile.is_x86_64 = true;
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) != 0) {
    profile.has_clflush = (edx & (1U << 19U)) != 0;
    profile.cache_line_bytes = ((ebx >> 8U) & 0xffU) * 8U;
  }
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) != 0) {
    profile.has_clflushopt = (ebx & (1U << 23U)) != 0;
  }
  return profile;
}

void validate_cpu_visibility_profile() {
  const auto profile = query_cpu_visibility_profile();
  if (!profile.has_clflush) {
    throw std::runtime_error("DAX-Coordinated L1 requires CPUID CLFSH support");
  }
  if (profile.cache_line_bytes != kCacheLineBytes) {
    throw std::runtime_error(
        "DAX-Coordinated L1 requires a 64-byte CPUID cache line");
  }
}

void publish_range(void* address, std::size_t length) {
  compiler_barrier();
  // Ordinary CLFLUSH is ordered with respect to earlier writes.  Keep the
  // trailing MFENCE so neither a metadata publication nor another memory
  // access can pass completion of the flush range.
  clflush_range(address, length);
  _mm_mfence();
  compiler_barrier();
}

void refresh_range(void* address, std::size_t length) {
  // CLFLUSH + MFENCE also completes invalidation before the caller loads.
  publish_range(address, length);
}

void publish_range_bulk(void* address, std::size_t length) {
  const VisibilityRange range{address, length};
  publish_ranges_bulk(&range, 1);
}

void publish_ranges_bulk(const VisibilityRange* ranges, std::size_t count) {
  // Format/attach validates CLFLUSHOPT before this profile can be used.
  if (count == 0) {
    return;
  }
  compiler_barrier();
  for (std::size_t index = 0; index < count; ++index) {
    clflushopt_range(ranges[index].address, ranges[index].length);
  }
  _mm_sfence();
  compiler_barrier();
}

}  // namespace lmcache::dax_coordinated_l1
