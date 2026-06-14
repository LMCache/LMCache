// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include <cstdint>
#include <string>

std::string xpu_get_ipc_handle(uintptr_t data_ptr, int device_index);

torch::Tensor xpu_open_ipc_handle(const std::string& handle_bytes,
                                  int64_t nbytes, int device_index);
