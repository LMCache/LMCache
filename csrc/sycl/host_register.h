// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

bool xpu_host_register(int64_t ptr, int64_t n_bytes);
bool xpu_host_unregister(int64_t ptr);
