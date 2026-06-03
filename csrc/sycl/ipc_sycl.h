// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>

#include <cstdint>
#include <string>
#include <tuple>

std::string xpu_get_ipc_handle(uintptr_t data_ptr, int device_index);

std::tuple<std::string, int64_t> xpu_get_ipc_handle_with_fd(
    uintptr_t data_ptr, int device_index);

torch::Tensor xpu_open_ipc_handle(const std::string& handle_bytes,
                                  int64_t nbytes, int device_index);

torch::Tensor xpu_open_ipc_handle_with_local_fd(
    const std::string& handle_bytes, int64_t local_fd, int64_t nbytes,
    int device_index);
