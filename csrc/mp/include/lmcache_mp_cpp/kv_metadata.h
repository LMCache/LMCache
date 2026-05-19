// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "lmcache_mp_cpp/msgpack_lite.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lmcache::mp {

struct KvShapeInfo {
  std::uint64_t num_blocks = 0;
  std::uint64_t block_size = 0;
};

struct TrtLlmLayoutHints {
  bool present = false;
  std::uint64_t num_kv_heads = 0;
  std::uint64_t tokens_per_block = 0;
  std::uint64_t head_dim = 0;
};

using LayoutHints = std::unordered_map<std::string, msgpack::DecodedValue>;

bool ComputeTrtLlmFlatSize(const TrtLlmLayoutHints& hints,
                           std::uint64_t* flat_size);

std::optional<KvShapeInfo> InferKvShapeInfo(
    const std::vector<std::uint64_t>& shape, const std::string& kv_layout,
    const TrtLlmLayoutHints& trt_llm_hints = {});

const msgpack::DecodedValue* FindHint(const LayoutHints& layout_hints,
                                      const std::string& name);

bool DecodeDisabledHint(const msgpack::DecodedValue& value, bool* disabled);

std::optional<std::uint64_t> DecodePositiveRatioHint(
    const msgpack::DecodedValue& value);

bool DecodeOptionalUnsignedArrayHint(
    const LayoutHints& layout_hints, const std::string& name,
    std::optional<std::vector<std::uint64_t>>* out, std::string* error);

bool DecodeTrtLlmLayoutHints(const LayoutHints& layout_hints,
                             const std::string& engine_type,
                             const std::string& kv_layout,
                             TrtLlmLayoutHints* out, std::string* error);

bool ValidateSupportedLayoutHints(const LayoutHints& layout_hints,
                                  std::optional<std::uint64_t>* compress_ratio,
                                  std::string* error);

bool ValidateRegisteredKvWrappersSupported(
    const std::vector<msgpack::DecodedCudaIpcWrapper>& kv_wrappers,
    const std::string& kv_layout, std::uint32_t logical_block_size,
    const TrtLlmLayoutHints& trt_llm_hints,
    std::optional<std::uint64_t> compress_ratio_hint,
    const std::optional<std::vector<std::uint64_t>>& group_physical_block_sizes,
    const std::optional<std::vector<std::uint64_t>>& group_compress_ratios,
    std::string* error);

}  // namespace lmcache::mp
