// SPDX-License-Identifier: Apache-2.0

#include "cache_flush.h"

#include <stdexcept>

#if defined(__x86_64__)
  #include <cpuid.h>
  #include <immintrin.h>
#endif

namespace lmcache {
namespace lmcache_native {

#if defined(__x86_64__)

namespace {

struct FlushCaps {
  bool has_clflushopt;
  size_t line_size;
};

FlushCaps detect_flush_caps() {
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  // CPUID.01H:EBX[15:8] is the CLFLUSH line size in 8-byte units.
  size_t line_size = 64;
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    size_t reported = ((ebx >> 8) & 0xff) * 8;
    if (reported != 0) {
      line_size = reported;
    }
  }
  // CPUID.(EAX=07H,ECX=0):EBX[23] reports CLFLUSHOPT.
  bool has_clflushopt = false;
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    has_clflushopt = (ebx & (1u << 23)) != 0;
  }
  return FlushCaps{has_clflushopt, line_size};
}

const FlushCaps& flush_caps() {
  static const FlushCaps caps = detect_flush_caps();
  return caps;
}

__attribute__((target("clflushopt"))) void flush_lines_clflushopt(
    uintptr_t begin, uintptr_t end, size_t line_size) {
  for (uintptr_t line = begin; line < end; line += line_size) {
    _mm_clflushopt(reinterpret_cast<void*>(line));
  }
  _mm_sfence();
}

void flush_lines_clflush(uintptr_t begin, uintptr_t end, size_t line_size) {
  for (uintptr_t line = begin; line < end; line += line_size) {
    _mm_clflush(reinterpret_cast<const void*>(line));
  }
  _mm_mfence();
}

}  // namespace

bool cache_flush_supported() { return true; }

void cache_flush_range(uintptr_t ptr, size_t size) {
  if (size == 0) {
    return;
  }
  const FlushCaps& caps = flush_caps();
  uintptr_t begin = ptr & ~(static_cast<uintptr_t>(caps.line_size) - 1);
  uintptr_t end = ptr + size;
  if (caps.has_clflushopt) {
    flush_lines_clflushopt(begin, end, caps.line_size);
  } else {
    flush_lines_clflush(begin, end, caps.line_size);
  }
}

#else

bool cache_flush_supported() { return false; }

void cache_flush_range(uintptr_t /*ptr*/, size_t /*size*/) {
  throw std::runtime_error(
      "cache_flush_range is only implemented for x86-64 in this build");
}

#endif

}  // namespace lmcache_native
}  // namespace lmcache
