// SPDX-License-Identifier: Apache-2.0

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

#include <c10/util/Exception.h>
#include <c10/xpu/XPUStream.h>

#include "host_register.h"

namespace {

bool host_register(void* ptr, size_t n_bytes) {
  if (ptr == nullptr || n_bytes == 0) {
    return false;
  }

  const auto context = c10::xpu::getCurrentXPUStream().queue().get_context();
  try {
    sycl::ext::oneapi::experimental::prepare_for_device_copy(ptr, n_bytes,
                                                             context);
  } catch (const sycl::exception& exception) {
    TORCH_WARN("prepare_for_device_copy failed (", exception.what(),
               "); transfers remain correct but use pageable host memory");
    return false;
  }
  return true;
}

bool host_unregister(void* ptr) {
  if (ptr == nullptr) {
    return false;
  }

  const auto context = c10::xpu::getCurrentXPUStream().queue().get_context();
  try {
    sycl::ext::oneapi::experimental::release_from_device_copy(ptr, context);
  } catch (const sycl::exception& exception) {
    TORCH_WARN("release_from_device_copy failed (", exception.what(), ")");
    return false;
  }
  return true;
}

}  // namespace

bool xpu_host_register(int64_t ptr, int64_t n_bytes) {
  TORCH_CHECK(n_bytes >= 0, "n_bytes must be non-negative");
  if (ptr == 0 || n_bytes == 0) {
    return false;
  }
  return host_register(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)),
                       static_cast<size_t>(n_bytes));
}

bool xpu_host_unregister(int64_t ptr) {
  if (ptr == 0) {
    return false;
  }
  return host_unregister(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)));
}
