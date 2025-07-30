/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "kernels/types.h"
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

namespace vllm_ascend {
AscendType get_dtype_from_torch(at::ScalarType scalarType)
{
    if (scalarType == at::ScalarType::Float) {
        return AscendType::FP32;
    } else if (scalarType == at::ScalarType::BFloat16) {
        return AscendType::BF16;
    } else if (scalarType == at::ScalarType::Half) {
        return AscendType::FP16;
    } else if (scalarType == at::ScalarType::Long) {
        return AscendType::INT64;
    } else if (scalarType == at::ScalarType::Int) {
        return AscendType::INT32;
    } else {
        TORCH_CHECK(false, "ScalarType not supported.");
    }
};
} // namespace vllm_ascend
