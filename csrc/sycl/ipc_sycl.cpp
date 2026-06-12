// SPDX-License-Identifier: Apache-2.0

#include "ipc_sycl.h"

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

void check_ze_result(ze_result_t result, const char* operation) {
  if (result != ZE_RESULT_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed with Level Zero error " +
                             std::to_string(static_cast<int>(result)));
  }
}

struct NativeLevelZeroHandles {
  ze_context_handle_t context;
  ze_device_handle_t device;
};

NativeLevelZeroHandles current_level_zero_handles(int device_index) {
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(device_index).queue();
  sycl::context sycl_context = queue.get_context();
  sycl::device sycl_device = queue.get_device();
  return {
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context),
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device),
  };
}

}  // namespace

std::string xpu_get_ipc_handle(uintptr_t data_ptr, int device_index) {
  const c10::OptionalDeviceGuard device_guard(
      at::Device(at::kXPU, device_index));
  NativeLevelZeroHandles handles = current_level_zero_handles(device_index);

  ze_ipc_mem_handle_t ipc_handle;
  std::memset(&ipc_handle, 0, sizeof(ipc_handle));
  check_ze_result(
      zeMemGetIpcHandle(
          handles.context, reinterpret_cast<const void*>(data_ptr), &ipc_handle),
      "zeMemGetIpcHandle");
  return std::string(
      reinterpret_cast<const char*>(&ipc_handle), sizeof(ipc_handle));
}

std::tuple<std::string, int64_t> xpu_get_ipc_handle_with_fd(
    uintptr_t data_ptr, int device_index) {
  const c10::OptionalDeviceGuard device_guard(
      at::Device(at::kXPU, device_index));
  NativeLevelZeroHandles handles = current_level_zero_handles(device_index);

  ze_ipc_mem_handle_t ipc_handle;
  std::memset(&ipc_handle, 0, sizeof(ipc_handle));
  check_ze_result(
      zeMemGetIpcHandle(
          handles.context, reinterpret_cast<const void*>(data_ptr), &ipc_handle),
      "zeMemGetIpcHandle");

  uint64_t fd = 0;
  check_ze_result(
      zeMemGetFileDescriptorFromIpcHandleExp(handles.context, ipc_handle, &fd),
      "zeMemGetFileDescriptorFromIpcHandleExp");

  return {
      std::string(reinterpret_cast<const char*>(&ipc_handle), sizeof(ipc_handle)),
      static_cast<int64_t>(fd),
  };
}

torch::Tensor xpu_open_ipc_handle(const std::string& handle_bytes,
                                  int64_t nbytes, int device_index) {
  if (handle_bytes.size() != sizeof(ze_ipc_mem_handle_t)) {
    throw std::runtime_error(
        "Invalid XPU IPC memory handle size: expected " +
        std::to_string(sizeof(ze_ipc_mem_handle_t)) + ", got " +
        std::to_string(handle_bytes.size()));
  }
  if (nbytes <= 0) {
    throw std::runtime_error("XPU IPC allocation size must be positive");
  }

  const c10::OptionalDeviceGuard device_guard(
      at::Device(at::kXPU, device_index));
  NativeLevelZeroHandles handles = current_level_zero_handles(device_index);

  ze_ipc_mem_handle_t ipc_handle;
  std::memcpy(&ipc_handle, handle_bytes.data(), sizeof(ipc_handle));
  void* ptr = nullptr;
  check_ze_result(
      zeMemOpenIpcHandle(
          handles.context, handles.device, ipc_handle,
          ZE_IPC_MEMORY_FLAG_BIAS_CACHED,
          &ptr),
      "zeMemOpenIpcHandle");

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .device(torch::Device(torch::kXPU, device_index));
  return torch::from_blob(
      ptr,
      {nbytes},
      [context = handles.context](void* mapped_ptr) {
        zeMemCloseIpcHandle(context, mapped_ptr);
      },
      options);
}

torch::Tensor xpu_open_ipc_handle_with_local_fd(
    const std::string& handle_bytes, int64_t local_fd, int64_t nbytes,
    int device_index) {
  if (handle_bytes.size() != sizeof(ze_ipc_mem_handle_t)) {
    throw std::runtime_error(
        "Invalid XPU IPC memory handle size: expected " +
        std::to_string(sizeof(ze_ipc_mem_handle_t)) + ", got " +
        std::to_string(handle_bytes.size()));
  }
  if (local_fd < 0) {
    throw std::runtime_error("XPU IPC local fd must be non-negative");
  }
  if (nbytes <= 0) {
    throw std::runtime_error("XPU IPC allocation size must be positive");
  }

  const c10::OptionalDeviceGuard device_guard(
      at::Device(at::kXPU, device_index));
  NativeLevelZeroHandles handles = current_level_zero_handles(device_index);

  ze_ipc_mem_handle_t ipc_handle;
  std::memcpy(&ipc_handle, handle_bytes.data(), sizeof(ipc_handle));
  int fd32 = static_cast<int>(local_fd);
  std::memcpy(&ipc_handle, &fd32, sizeof(fd32));

  void* ptr = nullptr;
  try {
    check_ze_result(
        zeMemOpenIpcHandle(
            handles.context, handles.device, ipc_handle,
            ZE_IPC_MEMORY_FLAG_BIAS_CACHED,
            &ptr),
        "zeMemOpenIpcHandle(local-fd)");
  } catch (...) {
    close(fd32);
    throw;
  }

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .device(torch::Device(torch::kXPU, device_index));
  return torch::from_blob(
      ptr,
      {nbytes},
      [context = handles.context, local_fd](void* mapped_ptr) {
        zeMemCloseIpcHandle(context, mapped_ptr);
        close(local_fd);
      },
      options);
}
