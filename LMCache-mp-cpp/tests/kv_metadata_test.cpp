// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/kv_metadata.h"

#include <cassert>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

lmcache::mp::msgpack::DecodedValue UnsignedValue(std::uint64_t value) {
  lmcache::mp::msgpack::DecodedValue out;
  out.kind = lmcache::mp::msgpack::DecodedValue::Kind::kUnsigned;
  out.unsigned_value = value;
  return out;
}

lmcache::mp::msgpack::DecodedValue StringValue(const std::string& value) {
  lmcache::mp::msgpack::DecodedValue out;
  out.kind = lmcache::mp::msgpack::DecodedValue::Kind::kString;
  out.string_value = value;
  return out;
}

lmcache::mp::msgpack::DecodedValue UnsignedArrayValue(
    std::vector<std::uint64_t> value) {
  lmcache::mp::msgpack::DecodedValue out;
  out.kind = lmcache::mp::msgpack::DecodedValue::Kind::kUnsignedArray;
  out.unsigned_array_value = std::move(value);
  return out;
}

lmcache::mp::msgpack::DecodedCudaIpcWrapper Wrapper(
    std::string dtype, std::vector<std::uint64_t> shape) {
  lmcache::mp::msgpack::DecodedCudaIpcWrapper out;
  out.dtype = std::move(dtype);
  out.shape = std::move(shape);
  return out;
}

}  // namespace

int main() {
  using lmcache::mp::ComputeTrtLlmFlatSize;
  using lmcache::mp::DecodeDisabledHint;
  using lmcache::mp::DecodeOptionalUnsignedArrayHint;
  using lmcache::mp::DecodePositiveRatioHint;
  using lmcache::mp::DecodeTrtLlmLayoutHints;
  using lmcache::mp::FindHint;
  using lmcache::mp::InferKvShapeInfo;
  using lmcache::mp::LayoutHints;
  using lmcache::mp::TrtLlmLayoutHints;
  using lmcache::mp::ValidateRegisteredKvWrappersSupported;
  using lmcache::mp::ValidateSupportedLayoutHints;

  const auto hnd_shape = InferKvShapeInfo({4, 2, 2, 32, 16, 8}, "HND");
  assert(hnd_shape);
  assert(hnd_shape->num_blocks == 4);
  assert(hnd_shape->block_size == 16);
  const auto nhd_shape = InferKvShapeInfo({4, 2, 2, 32, 16, 8}, "NHD");
  assert(nhd_shape);
  assert(nhd_shape->block_size == 32);
  assert(!InferKvShapeInfo({0, 2, 2, 32, 16, 8}, "HND"));

  TrtLlmLayoutHints trt_hints{
      .present = true, .num_kv_heads = 2, .tokens_per_block = 16, .head_dim = 8};
  std::uint64_t flat_size = 0;
  assert(ComputeTrtLlmFlatSize(trt_hints, &flat_size));
  assert(flat_size == 256);
  assert(InferKvShapeInfo({3, 1, 2, 256}, "HND", trt_hints)->block_size == 16);
  TrtLlmLayoutHints overflow_hints{
      .present = true,
      .num_kv_heads = std::numeric_limits<std::uint64_t>::max(),
      .tokens_per_block = 2,
      .head_dim = 8};
  assert(!ComputeTrtLlmFlatSize(overflow_hints, &flat_size));

  bool disabled = false;
  assert(DecodeDisabledHint(StringValue("false"), &disabled) && disabled);
  assert(DecodeDisabledHint(UnsignedValue(1), &disabled) && !disabled);
  assert(DecodePositiveRatioHint(StringValue("2")) == 2);
  assert(!DecodePositiveRatioHint(StringValue("0")));
  assert(!DecodePositiveRatioHint(StringValue("bad")));

  LayoutHints hints;
  hints["compress_ratio"] = StringValue("2");
  hints["use_layerwise"] = StringValue("false");
  assert(FindHint(hints, "compress_ratio") != nullptr);
  std::optional<std::uint64_t> compress_ratio;
  std::string error;
  assert(ValidateSupportedLayoutHints(hints, &compress_ratio, &error));
  assert(compress_ratio == 2);

  hints["group_physical_block_sizes"] = UnsignedArrayValue({16, 16});
  std::optional<std::vector<std::uint64_t>> group_sizes;
  assert(DecodeOptionalUnsignedArrayHint(hints, "group_physical_block_sizes",
                                         &group_sizes, &error));
  assert(group_sizes && *group_sizes == std::vector<std::uint64_t>({16, 16}));

  LayoutHints trt_layout_hints;
  trt_layout_hints["num_kv_heads"] = UnsignedValue(2);
  trt_layout_hints["tokens_per_block"] = UnsignedValue(16);
  trt_layout_hints["head_dim"] = UnsignedValue(8);
  TrtLlmLayoutHints decoded_trt;
  assert(DecodeTrtLlmLayoutHints(trt_layout_hints, "trtllm", "HND",
                                 &decoded_trt, &error));
  assert(decoded_trt.present);
  assert(decoded_trt.tokens_per_block == 16);
  assert(!DecodeTrtLlmLayoutHints(trt_layout_hints, "vllm", "HND",
                                  &decoded_trt, &error));

  const std::vector<lmcache::mp::msgpack::DecodedCudaIpcWrapper> wrappers = {
      Wrapper("torch.float16", {4, 2, 2, 32, 16, 8}),
      Wrapper("torch.float16", {4, 2, 2, 32, 16, 8}),
  };
  assert(ValidateRegisteredKvWrappersSupported(
      wrappers, "HND", 32, {}, 2, std::vector<std::uint64_t>{16, 16},
      std::vector<std::uint64_t>{2, 2}, &error));
  assert(!ValidateRegisteredKvWrappersSupported(
      {Wrapper("torch.float16", {4, 2, 2, 32, 16, 8}),
       Wrapper("torch.bfloat16", {4, 2, 2, 32, 16, 8})},
      "HND", 32, {}, 2, std::vector<std::uint64_t>{16, 16},
      std::vector<std::uint64_t>{2, 2}, &error));
  assert(error ==
         "native REGISTER_KV_CACHE does not support heterogeneous KV tensor "
         "dtypes");

  return 0;
}

