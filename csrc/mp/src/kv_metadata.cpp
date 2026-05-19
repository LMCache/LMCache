// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/kv_metadata.h"

#include <limits>

namespace lmcache::mp {

bool ComputeTrtLlmFlatSize(const TrtLlmLayoutHints& hints,
                           std::uint64_t* flat_size) {
  if (hints.num_kv_heads == 0 || hints.tokens_per_block == 0 ||
      hints.head_dim == 0 ||
      hints.num_kv_heads >
          std::numeric_limits<std::uint64_t>::max() / hints.tokens_per_block) {
    return false;
  }
  const std::uint64_t heads_tokens =
      hints.num_kv_heads * hints.tokens_per_block;
  if (heads_tokens >
      std::numeric_limits<std::uint64_t>::max() / hints.head_dim) {
    return false;
  }
  *flat_size = heads_tokens * hints.head_dim;
  return true;
}

std::optional<KvShapeInfo> InferKvShapeInfo(
    const std::vector<std::uint64_t>& shape, const std::string& kv_layout,
    const TrtLlmLayoutHints& trt_llm_hints) {
  if (shape.size() < 3) {
    return std::nullopt;
  }
  const bool hnd = kv_layout == "HND";
  KvShapeInfo info;
  if (shape.size() == 6 && shape[2] == 2) {
    info.num_blocks = shape[0];
    info.block_size = hnd ? shape[4] : shape[3];
  } else if (shape.size() == 4 && shape[2] == 2 && trt_llm_hints.present) {
    std::uint64_t flat_size = 0;
    if (!ComputeTrtLlmFlatSize(trt_llm_hints, &flat_size) ||
        shape[3] != flat_size) {
      return std::nullopt;
    }
    info.num_blocks = shape[0];
    info.block_size = trt_llm_hints.tokens_per_block;
  } else if (shape.size() == 5 && shape[0] == 2) {
    info.num_blocks = shape[1];
    info.block_size = hnd ? shape[3] : shape[2];
  } else if (shape.size() == 5 && shape[1] == 2) {
    info.num_blocks = shape[0];
    info.block_size = hnd ? shape[3] : shape[2];
  } else if (shape.size() == 3) {
    info.num_blocks = shape[0];
    info.block_size = shape[1];
  } else if (shape.size() == 4 && shape[0] == 2) {
    info.num_blocks = shape[1];
    info.block_size = shape[2];
  } else {
    return std::nullopt;
  }
  if (info.num_blocks == 0 || info.block_size == 0) {
    return std::nullopt;
  }
  return info;
}

const msgpack::DecodedValue* FindHint(const LayoutHints& layout_hints,
                                      const std::string& name) {
  const auto it = layout_hints.find(name);
  if (it == layout_hints.end()) {
    return nullptr;
  }
  return &it->second;
}

bool DecodeDisabledHint(const msgpack::DecodedValue& value, bool* disabled) {
  switch (value.kind) {
    case msgpack::DecodedValue::Kind::kNil:
      *disabled = true;
      return true;
    case msgpack::DecodedValue::Kind::kBool:
      *disabled = !value.bool_value;
      return true;
    case msgpack::DecodedValue::Kind::kUnsigned:
      *disabled = value.unsigned_value == 0;
      return true;
    case msgpack::DecodedValue::Kind::kString:
      *disabled = value.string_value.empty() || value.string_value == "0" ||
                  value.string_value == "false" ||
                  value.string_value == "False";
      return true;
  }
  return false;
}

std::optional<std::uint64_t> DecodePositiveRatioHint(
    const msgpack::DecodedValue& value) {
  switch (value.kind) {
    case msgpack::DecodedValue::Kind::kUnsigned:
      if (value.unsigned_value == 0) {
        return std::nullopt;
      }
      return value.unsigned_value;
    case msgpack::DecodedValue::Kind::kString:
      if (value.string_value.empty()) {
        return std::nullopt;
      }
      {
        std::uint64_t parsed = 0;
        for (const char ch : value.string_value) {
          if (ch < '0' || ch > '9') {
            return std::nullopt;
          }
          const std::uint64_t digit = static_cast<std::uint64_t>(ch - '0');
          if (parsed >
              (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
            return std::nullopt;
          }
          parsed = parsed * 10 + digit;
        }
        if (parsed == 0) {
          return std::nullopt;
        }
        return parsed;
      }
    default:
      return std::nullopt;
  }
}

bool DecodeOptionalUnsignedArrayHint(
    const LayoutHints& layout_hints, const std::string& name,
    std::optional<std::vector<std::uint64_t>>* out, std::string* error) {
  *out = std::nullopt;
  const msgpack::DecodedValue* hint = FindHint(layout_hints, name);
  if (hint == nullptr) {
    return true;
  }
  if (hint->kind != msgpack::DecodedValue::Kind::kUnsignedArray) {
    *error = "invalid REGISTER_KV_CACHE " + name + " hint";
    return false;
  }
  *out = hint->unsigned_array_value;
  return true;
}

bool DecodeTrtLlmLayoutHints(const LayoutHints& layout_hints,
                             const std::string& engine_type,
                             const std::string& kv_layout,
                             TrtLlmLayoutHints* out, std::string* error) {
  *out = {};
  const msgpack::DecodedValue* num_kv_heads =
      FindHint(layout_hints, "num_kv_heads");
  const msgpack::DecodedValue* tokens_per_block =
      FindHint(layout_hints, "tokens_per_block");
  const msgpack::DecodedValue* head_dim = FindHint(layout_hints, "head_dim");
  if (num_kv_heads == nullptr && tokens_per_block == nullptr &&
      head_dim == nullptr) {
    return true;
  }
  if (num_kv_heads == nullptr || tokens_per_block == nullptr ||
      head_dim == nullptr) {
    *error = "native REGISTER_KV_CACHE requires complete TRT-LLM layout hints";
    return false;
  }
  if (engine_type != "trtllm") {
    *error =
        "native REGISTER_KV_CACHE accepts TRT-LLM layout hints only for "
        "trtllm engine";
    return false;
  }
  if (!kv_layout.empty() && kv_layout != "HND") {
    *error = "native REGISTER_KV_CACHE TRT-LLM layout hints require HND layout";
    return false;
  }
  if (num_kv_heads->kind != msgpack::DecodedValue::Kind::kUnsigned ||
      tokens_per_block->kind != msgpack::DecodedValue::Kind::kUnsigned ||
      head_dim->kind != msgpack::DecodedValue::Kind::kUnsigned ||
      num_kv_heads->unsigned_value == 0 ||
      tokens_per_block->unsigned_value == 0 || head_dim->unsigned_value == 0) {
    *error = "invalid REGISTER_KV_CACHE TRT-LLM layout hint";
    return false;
  }
  *out = {
      .present = true,
      .num_kv_heads = num_kv_heads->unsigned_value,
      .tokens_per_block = tokens_per_block->unsigned_value,
      .head_dim = head_dim->unsigned_value,
  };
  std::uint64_t flat_size = 0;
  if (!ComputeTrtLlmFlatSize(*out, &flat_size)) {
    *error = "REGISTER_KV_CACHE TRT-LLM layout hints overflow";
    return false;
  }
  return true;
}

bool ValidateSupportedLayoutHints(const LayoutHints& layout_hints,
                                  std::optional<std::uint64_t>* compress_ratio,
                                  std::string* error) {
  *compress_ratio = std::nullopt;
  if (const msgpack::DecodedValue* hint =
          FindHint(layout_hints, "compress_ratio")) {
    const std::optional<std::uint64_t> decoded = DecodePositiveRatioHint(*hint);
    if (!decoded) {
      *error = "invalid REGISTER_KV_CACHE compress_ratio hint";
      return false;
    }
    *compress_ratio = *decoded;
  }

  for (const std::string name : {"use_layerwise", "layerwise"}) {
    if (const msgpack::DecodedValue* hint = FindHint(layout_hints, name)) {
      bool disabled = false;
      if (!DecodeDisabledHint(*hint, &disabled)) {
        *error = "invalid REGISTER_KV_CACHE " + name + " hint";
        return false;
      }
    }
  }

  for (const std::string name : {"kv_layer_groups", "layout_groups"}) {
    if (FindHint(layout_hints, name) != nullptr) {
      *error =
          "native REGISTER_KV_CACHE does not support heterogeneous KV groups";
      return false;
    }
  }

  return true;
}

bool ValidateRegisteredKvWrappersSupported(
    const std::vector<msgpack::DecodedCudaIpcWrapper>& kv_wrappers,
    const std::string& kv_layout, std::uint32_t logical_block_size,
    const TrtLlmLayoutHints& trt_llm_hints,
    std::optional<std::uint64_t> compress_ratio_hint,
    const std::optional<std::vector<std::uint64_t>>& group_physical_block_sizes,
    const std::optional<std::vector<std::uint64_t>>& group_compress_ratios,
    std::string* error) {
  if (kv_wrappers.empty()) {
    if (trt_llm_hints.present) {
      *error =
          "native REGISTER_KV_CACHE requires KV wrapper metadata for TRT-LLM "
          "layouts";
      return false;
    }
    if ((compress_ratio_hint && *compress_ratio_hint != 1) ||
        (group_physical_block_sizes && !group_physical_block_sizes->empty()) ||
        (group_compress_ratios && !group_compress_ratios->empty())) {
      *error =
          "native REGISTER_KV_CACHE requires KV wrapper metadata for "
          "compressed layouts";
      return false;
    }
    return true;
  }

  if (compress_ratio_hint && *compress_ratio_hint != 1 &&
      logical_block_size == 0) {
    *error =
        "native REGISTER_KV_CACHE requires a logical block-size hint for "
        "compressed layouts";
    return false;
  }
  if (group_physical_block_sizes &&
      group_physical_block_sizes->size() != kv_wrappers.size()) {
    *error =
        "native REGISTER_KV_CACHE group_physical_block_sizes hint does not "
        "match registered KV wrapper count";
    return false;
  }
  if (group_compress_ratios &&
      group_compress_ratios->size() != kv_wrappers.size()) {
    *error =
        "native REGISTER_KV_CACHE group_compress_ratios hint does not match "
        "registered KV wrapper count";
    return false;
  }
  if (group_compress_ratios && logical_block_size == 0) {
    *error =
        "native REGISTER_KV_CACHE requires a logical block-size hint for "
        "group compression ratios";
    return false;
  }

  const auto first_shape_info =
      InferKvShapeInfo(kv_wrappers.front().shape, kv_layout, trt_llm_hints);
  if (!first_shape_info) {
    *error = "native REGISTER_KV_CACHE received unsupported KV tensor shape";
    return false;
  }

  const std::string& first_dtype = kv_wrappers.front().dtype;
  for (std::size_t i = 0; i < kv_wrappers.size(); ++i) {
    const msgpack::DecodedCudaIpcWrapper& wrapper = kv_wrappers[i];
    const auto shape_info =
        InferKvShapeInfo(wrapper.shape, kv_layout, trt_llm_hints);
    if (!shape_info) {
      *error = "native REGISTER_KV_CACHE received unsupported KV tensor shape";
      return false;
    }
    if (logical_block_size != 0 &&
        (shape_info->block_size > logical_block_size ||
         logical_block_size % shape_info->block_size != 0)) {
      *error =
          "native REGISTER_KV_CACHE requires physical KV block size to divide "
          "the logical block-size hint";
      return false;
    }
    if (group_physical_block_sizes &&
        shape_info->block_size != (*group_physical_block_sizes)[i]) {
      *error =
          "native REGISTER_KV_CACHE group_physical_block_sizes hint does not "
          "match registered KV block-size metadata";
      return false;
    }
    if (compress_ratio_hint && logical_block_size != 0) {
      const std::uint64_t actual_ratio =
          logical_block_size / shape_info->block_size;
      if (actual_ratio != *compress_ratio_hint) {
        *error =
            "native REGISTER_KV_CACHE compress_ratio hint does not match "
            "registered KV block-size metadata";
        return false;
      }
    }
    if (group_compress_ratios) {
      const std::uint64_t actual_ratio =
          logical_block_size / shape_info->block_size;
      if (actual_ratio != (*group_compress_ratios)[i]) {
        *error =
            "native REGISTER_KV_CACHE group_compress_ratios hint does not "
            "match registered KV block-size metadata";
        return false;
      }
    }
    if (wrapper.dtype != first_dtype) {
      *error =
          "native REGISTER_KV_CACHE does not support heterogeneous KV "
          "tensor dtypes";
      return false;
    }
  }

  return true;
}

}  // namespace lmcache::mp
