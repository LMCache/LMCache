// SPDX-License-Identifier: Apache-2.0

#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/sycl.hpp>
#include "ipc_sycl.h"

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ipc_memory = sycl::ext::oneapi::experimental::ipc_memory;

struct SyclIpcContext {
  sycl::context context;
  sycl::device device;
};

SyclIpcContext current_sycl_ipc_context(int device_index) {
  std::vector<sycl::device> ipc_devices;
  for (const auto& device :
       sycl::device::get_devices(sycl::info::device_type::gpu)) {
    if (device.has(sycl::aspect::ext_oneapi_ipc_memory)) {
      ipc_devices.push_back(device);
    }
  }
  if (device_index < 0 ||
      static_cast<size_t>(device_index) >= ipc_devices.size()) {
    throw std::runtime_error(
        "XPU device does not support SYCL aspect::ext_oneapi_ipc_memory");
  }
  sycl::device sycl_device = ipc_devices[device_index];
  sycl::context sycl_context =
      sycl_device.get_platform().khr_get_default_context();
  return {sycl_context, sycl_device};
}

}  // namespace

std::string xpu_get_ipc_handle(uintptr_t data_ptr, int device_index) {
  const c10::OptionalDeviceGuard device_guard(
      at::Device(at::kXPU, device_index));
  SyclIpcContext sycl_ipc = current_sycl_ipc_context(device_index);

  auto ipc_handle =
      ipc_memory::get(reinterpret_cast<void*>(data_ptr), sycl_ipc.context);
  auto handle_data = ipc_handle.data();
  std::string handle_bytes(
      reinterpret_cast<const char*>(handle_data.data()), handle_data.size());
  ipc_memory::put(ipc_handle, sycl_ipc.context);
  return handle_bytes;
}

torch::Tensor xpu_open_ipc_handle(const std::string& handle_bytes,
                                  int64_t nbytes, int device_index) {
  if (handle_bytes.empty()) {
    throw std::runtime_error("XPU IPC memory handle must not be empty");
  }
  if (nbytes <= 0) {
    throw std::runtime_error("XPU IPC allocation size must be positive");
  }

  const c10::OptionalDeviceGuard device_guard(
      at::Device(at::kXPU, device_index));
  SyclIpcContext sycl_ipc = current_sycl_ipc_context(device_index);

  std::vector<std::byte> handle_data(handle_bytes.size());
  for (size_t i = 0; i < handle_bytes.size(); ++i) {
    handle_data[i] = static_cast<std::byte>(handle_bytes[i]);
  }
  void* ptr = ipc_memory::open(handle_data, sycl_ipc.context, sycl_ipc.device);

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .device(torch::Device(torch::kXPU, device_index));
  return torch::from_blob(
      ptr,
      {nbytes},
      [context = sycl_ipc.context](void* mapped_ptr) {
        ipc_memory::close(mapped_ptr, context);
      },
      options);
}
