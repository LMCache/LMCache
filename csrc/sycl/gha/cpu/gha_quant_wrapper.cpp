// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <algorithm>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <cstring>

#ifdef USE_MULTITHREADING
  #include <omp.h>
#endif

#include "quant.h"

namespace {

#ifdef USE_MULTITHREADING
void ensure_nested_openmp_for_fused_halves(int requested_threads) {
  if (requested_threads < 2) {
    return;
  }

  static std::once_flag omp_nested_once;
  std::call_once(omp_nested_once, []() {
    omp_set_nested(1);
    omp_set_max_active_levels(2);
  });
}
#endif

void ensure_cpu_tensor(const torch::Tensor& tensor, const char* name) {
  if (!tensor.device().is_cpu()) {
    throw std::invalid_argument(std::string(name) +
                                " tensor must reside on CPU.");
  }
}

// Attaches the optional RH preconditioner. With rh on, the kernels index signs
// and perm once per element of a quantization chunk, so both must cover exactly
// one chunk and every perm entry must index into it.
void attach_precond(quantization_args& args, int64_t plane_numel,
                    int64_t num_layers,
                    const c10::optional<torch::Tensor>& signs_opt,
                    const c10::optional<torch::Tensor>& perm_opt,
                    torch::Tensor& signs_t, torch::Tensor& perm_t) {
  const bool has_signs = signs_opt.has_value() && signs_opt->defined();
  const bool has_perm = perm_opt.has_value() && perm_opt->defined();
  if (has_signs) {
    signs_t = signs_opt->to(torch::kFloat32).contiguous();
  }
  if (has_perm) {
    perm_t = perm_opt->to(torch::kInt32).contiguous();
  }
  if (args.rh && (has_signs || has_perm)) {
    int64_t chunks_per_layer = 1;
    if (args.scaling_method == "per_token") {
      chunks_per_layer =
          static_cast<int64_t>(args.blocks_num) * args.block_size;
    } else if (args.scaling_method == "per_channel") {
      chunks_per_layer = args.head_dim;
    }
    const int64_t chunks = std::max<int64_t>(1, num_layers) * chunks_per_layer;
    if (chunks <= 0 || plane_numel % chunks != 0) {
      throw std::invalid_argument(
          "element count is not divisible by the number of quantization "
          "chunks (check blocks_num/block_size/head_dim/num_layers)");
    }
    const int64_t transform_len = plane_numel / chunks;
    if (has_signs && signs_t.numel() != transform_len) {
      throw std::invalid_argument(
          "signs length must equal the quantization chunk size");
    }
    if (has_perm) {
      if (perm_t.numel() != transform_len) {
        throw std::invalid_argument(
            "perm length must equal the quantization chunk size");
      }
      const int32_t* perm = perm_t.data_ptr<int32_t>();
      for (int64_t i = 0; i < transform_len; ++i) {
        if (perm[i] < 0 || perm[i] >= transform_len) {
          throw std::invalid_argument("perm entry out of range");
        }
      }
    }
  }
  if (has_signs) args.signs = signs_t.data_ptr<float>();
  if (has_perm) args.perm = perm_t.data_ptr<int32_t>();
}

quantization_args build_quant_args(int qbit, int blocks_num, int block_size,
                                   int head_num, int head_dim, bool rh,
                                   bool asym,
                                   const std::string& scaling_method) {
  quantization_args args;
  args.qbit = qbit;
  args.blocks_num = blocks_num;
  args.block_size = block_size;
  args.head_num = head_num;
  args.head_dim = head_dim;
  args.rh = rh;
  args.asym = asym;
  args.scaling_method = scaling_method;
  return args;
}

int64_t quantized_numel_for_qbit(int64_t logical_numel, int qbit) {
  if (qbit == 4) {
    return q4_detail::packed_size(logical_numel);
  }
  return logical_numel;
}

template <typename KVType, typename QType>
void run_quantize_kernel(torch::Tensor& src, torch::Tensor& dst,
                         int64_t scale_id, quantization_args& args) {
  auto* src_ptr = reinterpret_cast<KVType*>(src.data_ptr());
  auto* dst_ptr = reinterpret_cast<QType*>(dst.data_ptr());
  int64_t numel = src.numel();
  quantize_kvcache<KVType, QType>(src_ptr, dst_ptr, numel, scale_id, args);
}

template <typename KVType, typename QType>
void run_quantize(torch::Tensor& src, torch::Tensor& dst, int64_t scale_id,
                  quantization_args& args) {
  // Ensure contiguous storage before raw-pointer casts
  // dst is written in place; contiguous() on a strided view would silently
  // write to a discarded copy.
  TORCH_CHECK(dst.is_contiguous(), "run_quantize: dst must be contiguous");
  src = src.contiguous();
  // Release the GIL via PyEval_SaveThread/RestoreThread (avoids pybind11's
  // decref_pyobject).
  Py_BEGIN_ALLOW_THREADS run_quantize_kernel<KVType, QType>(src, dst, scale_id,
                                                            args);
  Py_END_ALLOW_THREADS
}

template <typename KVType, typename QType>
void run_dequantize_kernel(torch::Tensor& src, torch::Tensor& dst,
                           int64_t scale_id, quantization_args& args) {
  auto* src_ptr = reinterpret_cast<QType*>(src.data_ptr());
  auto* dst_ptr = reinterpret_cast<KVType*>(dst.data_ptr());
  int64_t numel = dst.numel();
  dequantize_kvcache<KVType, QType>(src_ptr, dst_ptr, numel, scale_id, args);
}

template <typename KVType, typename QType>
void run_dequantize(torch::Tensor& src, torch::Tensor& dst, int64_t scale_id,
                    quantization_args& args) {
  // dst is written in place; contiguous() on a strided view would silently
  // write to a discarded copy.
  TORCH_CHECK(dst.is_contiguous(), "run_dequantize: dst must be contiguous");
  src = src.contiguous();
  // Release GIL for kernel
  Py_BEGIN_ALLOW_THREADS run_dequantize_kernel<KVType, QType>(src, dst,
                                                              scale_id, args);
  Py_END_ALLOW_THREADS
}

void dispatch_quantization_kernel(torch::Tensor& src, torch::Tensor& dst,
                                  int64_t scale_id, quantization_args& args) {
  const auto src_type = src.scalar_type();
  const auto dst_type = dst.scalar_type();

  if (dst_type == torch::kInt8) {
    if (src_type == torch::kFloat32) {
      run_quantize_kernel<float, int8_t>(src, dst, scale_id, args);
    } else if (src_type == torch::kFloat16) {
      run_quantize_kernel<_Float16, int8_t>(src, dst, scale_id, args);
    } else if (src_type == torch::kBFloat16) {
      run_quantize_kernel<__bf16, int8_t>(src, dst, scale_id, args);
    } else if (src_type == torch::kFloat8_e4m3fn) {
      run_quantize_kernel<gha_fp8_e4m3, int8_t>(src, dst, scale_id, args);
    } else if (src_type == torch::kFloat8_e5m2) {
      run_quantize_kernel<gha_fp8_e5m2, int8_t>(src, dst, scale_id, args);
    } else {
      throw std::invalid_argument(
          "Unsupported src dtype for quantization (int8 destination).");
    }
  } else if (dst_type == torch::kInt16) {
    if (src_type == torch::kFloat32) {
      run_quantize_kernel<float, int16_t>(src, dst, scale_id, args);
    } else if (src_type == torch::kFloat16) {
      run_quantize_kernel<_Float16, int16_t>(src, dst, scale_id, args);
    } else if (src_type == torch::kBFloat16) {
      run_quantize_kernel<__bf16, int16_t>(src, dst, scale_id, args);
    } else if (src_type == torch::kFloat8_e4m3fn) {
      run_quantize_kernel<gha_fp8_e4m3, int16_t>(src, dst, scale_id, args);
    } else if (src_type == torch::kFloat8_e5m2) {
      run_quantize_kernel<gha_fp8_e5m2, int16_t>(src, dst, scale_id, args);
    } else {
      throw std::invalid_argument(
          "Unsupported src dtype for quantization (int16 destination).");
    }
  } else {
    throw std::invalid_argument(
        "Destination dtype must be int8 or int16 for quantization.");
  }
}

void dispatch_dequantization(torch::Tensor& src, torch::Tensor& dst,
                             int64_t scale_id, quantization_args& args) {
  const auto src_type = src.scalar_type();
  const auto dst_type = dst.scalar_type();

  if (src_type == torch::kInt8) {
    if (dst_type == torch::kFloat32) {
      run_dequantize<float, int8_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat16) {
      run_dequantize<_Float16, int8_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kBFloat16) {
      run_dequantize<__bf16, int8_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat8_e4m3fn) {
      run_dequantize<gha_fp8_e4m3, int8_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat8_e5m2) {
      run_dequantize<gha_fp8_e5m2, int8_t>(src, dst, scale_id, args);
    } else {
      throw std::invalid_argument(
          "Unsupported destination dtype for int8 source.");
    }
  } else if (src_type == torch::kInt16) {
    if (dst_type == torch::kFloat32) {
      run_dequantize<float, int16_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat16) {
      run_dequantize<_Float16, int16_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kBFloat16) {
      run_dequantize<__bf16, int16_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat8_e4m3fn) {
      run_dequantize<gha_fp8_e4m3, int16_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat8_e5m2) {
      run_dequantize<gha_fp8_e5m2, int16_t>(src, dst, scale_id, args);
    } else {
      throw std::invalid_argument(
          "Unsupported destination dtype for int16 source.");
    }
  } else {
    throw std::invalid_argument(
        "Source dtype must be int8 or int16 for dequantization.");
  }
}

void dispatch_dequantization_kernel(torch::Tensor& src, torch::Tensor& dst,
                                    int64_t scale_id, quantization_args& args) {
  const auto src_type = src.scalar_type();
  const auto dst_type = dst.scalar_type();

  if (src_type == torch::kInt8) {
    if (dst_type == torch::kFloat32) {
      run_dequantize_kernel<float, int8_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat16) {
      run_dequantize_kernel<_Float16, int8_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kBFloat16) {
      run_dequantize_kernel<__bf16, int8_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat8_e4m3fn) {
      run_dequantize_kernel<gha_fp8_e4m3, int8_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat8_e5m2) {
      run_dequantize_kernel<gha_fp8_e5m2, int8_t>(src, dst, scale_id, args);
    } else {
      throw std::invalid_argument(
          "Unsupported destination dtype for int8 source.");
    }
  } else if (src_type == torch::kInt16) {
    if (dst_type == torch::kFloat32) {
      run_dequantize_kernel<float, int16_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat16) {
      run_dequantize_kernel<_Float16, int16_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kBFloat16) {
      run_dequantize_kernel<__bf16, int16_t>(src, dst, scale_id, args);
    } else if (dst_type == torch::kFloat8_e4m3fn) {
      run_dequantize_kernel<gha_fp8_e4m3, int16_t>(src, dst, scale_id,
                                                       args);
    } else if (dst_type == torch::kFloat8_e5m2) {
      run_dequantize_kernel<gha_fp8_e5m2, int16_t>(src, dst, scale_id,
                                                       args);
    } else {
      throw std::invalid_argument(
          "Unsupported destination dtype for int16 source.");
    }
  } else {
    throw std::invalid_argument(
        "Source dtype must be int8 or int16 for dequantization.");
  }
}

}  // namespace

// ---- Helpers to access thread-local scale maps from C++ ----

// Number of (scale, zero point) entries one quantize call produces.
static int64_t scale_count(const quantization_args& args) {
  if (args.scaling_method == "per_token") {
    return static_cast<int64_t>(args.blocks_num) * args.block_size;
  }
  if (args.scaling_method == "per_channel") {
    return args.head_dim;
  }
  return 1;
}

// Scale blob: [u32 count][{u32 chunk_idx, f32 scale, f32 zp} x count], entries
// in chunk order.
static std::string pack_scales(const std::vector<float>& scales,
                               const std::vector<float>& zps) {
  const uint32_t count = static_cast<uint32_t>(scales.size());
  constexpr size_t entry_size = sizeof(uint32_t) + 2 * sizeof(float);
  std::string buf(sizeof(uint32_t) + count * entry_size, '\0');
  char* ptr = buf.data();
  std::memcpy(ptr, &count, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  for (uint32_t i = 0; i < count; ++i) {
    std::memcpy(ptr, &i, sizeof(uint32_t));
    std::memcpy(ptr + sizeof(uint32_t), &scales[i], sizeof(float));
    std::memcpy(ptr + sizeof(uint32_t) + sizeof(float), &zps[i], sizeof(float));
    ptr += entry_size;
  }
  return buf;
}

// Inverse of pack_scales into ``count``-entry arrays; entries may come in any
// order, absent chunks read as 0.
static void unpack_scales(const std::string& blob, int64_t count,
                          std::vector<float>& scales, std::vector<float>& zps) {
  scales.assign(static_cast<size_t>(count), 0.0f);
  zps.assign(static_cast<size_t>(count), 0.0f);
  const char* ptr = blob.data();
  const char* end = ptr + blob.size();
  if (static_cast<size_t>(end - ptr) < sizeof(uint32_t)) {
    throw std::invalid_argument("Scale blob is too small.");
  }
  uint32_t n = 0;
  std::memcpy(&n, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  constexpr size_t entry_size = sizeof(uint32_t) + 2 * sizeof(float);
  if (static_cast<size_t>(end - ptr) < static_cast<size_t>(n) * entry_size) {
    throw std::invalid_argument("Scale blob is truncated.");
  }
  for (uint32_t i = 0; i < n; ++i, ptr += entry_size) {
    uint32_t idx = 0;
    std::memcpy(&idx, ptr, sizeof(uint32_t));
    if (static_cast<int64_t>(idx) >= count) {
      throw std::invalid_argument("scale blob chunk index out of range");
    }
    std::memcpy(&scales[idx], ptr + sizeof(uint32_t), sizeof(float));
    std::memcpy(&zps[idx], ptr + sizeof(uint32_t) + sizeof(float),
                sizeof(float));
  }
}

static std::vector<std::string> split_scale_blob(const std::string& blob,
                                                 int64_t num_layers) {
  std::vector<std::string> parts;
  parts.reserve(static_cast<size_t>(num_layers));
  size_t offset = 0;
  for (int64_t i = 0; i < num_layers; ++i) {
    if (offset + sizeof(uint32_t) > blob.size()) {
      throw std::invalid_argument("scale blob too short for split");
    }
    uint32_t count;
    std::memcpy(&count, blob.data() + offset, sizeof(uint32_t));
    constexpr size_t entry_size = sizeof(uint32_t) + 2 * sizeof(float);
    size_t blob_size = sizeof(uint32_t) + count * entry_size;
    parts.emplace_back(blob.data() + offset, blob_size);
    offset += blob_size;
  }
  return parts;
}

// per_token quantizes each row on its own, so all layers can go through one
// kernel call (one parallel region instead of one per layer) and the scales be
// split per layer afterwards; with 4 bits a layer must hold an even number of
// values so no packed byte straddles two layers.
static bool layers_merge_for_per_token(const quantization_args& args,
                                       int64_t layer_numel) {
  return args.scaling_method == "per_token" &&
         (args.qbit != 4 || layer_numel % 2 == 0);
}

static std::string quantize_one_half_per_layer(torch::Tensor& src,
                                               torch::Tensor& dst,
                                               int64_t base_scale_id,
                                               int64_t num_layers,
                                               quantization_args args) {
  int64_t layer_numel = src.numel() / num_layers;
  int64_t layer_q_numel = dst.numel() / num_layers;
  if (layers_merge_for_per_token(args, layer_numel)) {
    const int64_t rows = scale_count(args);  // per layer
    quantization_args all = args;
    all.blocks_num = 1;
    all.block_size = static_cast<int>(rows * num_layers);
    std::vector<float> scales(static_cast<size_t>(rows * num_layers));
    std::vector<float> zps(scales.size());
    all.scales_out = scales.data();
    all.zps_out = zps.data();
    dispatch_quantization_kernel(src, dst, base_scale_id, all);
    std::string all_scales;
    for (int64_t l = 0; l < num_layers; ++l) {
      all_scales +=
          pack_scales(std::vector<float>(scales.begin() + l * rows,
                                         scales.begin() + (l + 1) * rows),
                      std::vector<float>(zps.begin() + l * rows,
                                         zps.begin() + (l + 1) * rows));
    }
    return all_scales;
  }
  std::vector<float> scales(static_cast<size_t>(scale_count(args)));
  std::vector<float> zps(scales.size());
  args.scales_out = scales.data();
  args.zps_out = zps.data();
  std::string all_scales;
  for (int64_t l = 0; l < num_layers; ++l) {
    auto layer_src = torch::from_blob(static_cast<char*>(src.data_ptr()) +
                                          l * layer_numel * src.element_size(),
                                      {layer_numel}, src.options());
    auto layer_dst =
        torch::from_blob(static_cast<char*>(dst.data_ptr()) +
                             l * layer_q_numel * dst.element_size(),
                         {layer_q_numel}, dst.options());
    dispatch_quantization_kernel(layer_src, layer_dst, base_scale_id + l, args);
    all_scales += pack_scales(scales, zps);
  }
  return all_scales;
}

static void dequantize_one_half_per_layer(
    torch::Tensor& src, torch::Tensor& dst, int64_t scale_id,
    const std::vector<std::string>& layer_scales, int64_t num_layers,
    quantization_args args, int layer_num_threads = 1) {
  int64_t layer_q_numel = src.numel() / num_layers;
  int64_t layer_numel = dst.numel() / num_layers;
  if (layers_merge_for_per_token(args, layer_numel)) {
    const int64_t rows = scale_count(args);  // per layer
    std::vector<float> scales(static_cast<size_t>(rows * num_layers));
    std::vector<float> zps(scales.size());
    std::vector<float> ls, lz;
    for (int64_t l = 0; l < num_layers; ++l) {
      unpack_scales(layer_scales[static_cast<size_t>(l)], rows, ls, lz);
      std::copy(ls.begin(), ls.end(), scales.begin() + l * rows);
      std::copy(lz.begin(), lz.end(), zps.begin() + l * rows);
    }
    quantization_args all = args;
    all.blocks_num = 1;
    all.block_size = static_cast<int>(rows * num_layers);
    all.scales_in = scales.data();
    all.zps_in = zps.data();
    dispatch_dequantization_kernel(src, dst, scale_id, all);
    return;
  }
#ifdef USE_MULTITHREADING
  const int outer_threads =
      std::max(1, std::min(layer_num_threads, static_cast<int>(num_layers)));
  #pragma omp parallel for schedule(static) \
      num_threads(outer_threads) if (outer_threads > 1 && num_layers > 1)
#endif
  for (int64_t l = 0; l < num_layers; ++l) {
    auto layer_src =
        torch::from_blob(static_cast<char*>(src.data_ptr()) +
                             l * layer_q_numel * src.element_size(),
                         {layer_q_numel}, src.options());
    auto layer_dst = torch::from_blob(static_cast<char*>(dst.data_ptr()) +
                                          l * layer_numel * dst.element_size(),
                                      {layer_numel}, dst.options());
    quantization_args layer_args = args;
    std::vector<float> scales, zps;
    unpack_scales(layer_scales[static_cast<size_t>(l)], scale_count(args),
                  scales, zps);
    layer_args.scales_in = scales.data();
    layer_args.zps_in = zps.data();
    dispatch_dequantization_kernel(layer_src, layer_dst, scale_id, layer_args);
  }
}

static void dequantize_one_half(torch::Tensor& src, torch::Tensor& dst,
                                int64_t scale_id,
                                const std::string& scale_bytes,
                                quantization_args args) {
  std::vector<float> scales, zps;
  unpack_scales(scale_bytes, scale_count(args), scales, zps);
  args.scales_in = scales.data();
  args.zps_in = zps.data();
  dispatch_dequantization_kernel(src, dst, scale_id, args);
}

// dequantize_one_half that releases the GIL itself (dispatch_dequantization).
static void dequantize_one_half_releasing_gil(torch::Tensor& src,
                                              torch::Tensor& dst,
                                              int64_t scale_id,
                                              const std::string& scale_bytes,
                                              quantization_args args) {
  std::vector<float> scales, zps;
  unpack_scales(scale_bytes, scale_count(args), scales, zps);
  args.scales_in = scales.data();
  args.zps_in = zps.data();
  dispatch_dequantization(src, dst, scale_id, args);
}

static std::string quantize_one_half(torch::Tensor& src, torch::Tensor& dst,
                                     int64_t scale_id, quantization_args args) {
  std::vector<float> scales(static_cast<size_t>(scale_count(args)));
  std::vector<float> zps(scales.size());
  args.scales_out = scales.data();
  args.zps_out = zps.data();
  dispatch_quantization_kernel(src, dst, scale_id, args);
  return pack_scales(scales, zps);
}

static bool use_per_layer_quantization(int64_t num_layers,
                                       const std::string& scaling_method) {
  if (num_layers <= 1) {
    return false;
  }
  return (scaling_method == "per_token" || scaling_method == "per_channel" ||
          scaling_method == "per_tensor");
}

// -----------------------------------------------------------------------

// Fused dequantization of a layerwise K/V chunk.
torch::Tensor gha_dequantize_chunk_py(
    torch::Tensor q_data, pybind11::bytes k_scale_bytes,
    pybind11::bytes v_scale_bytes, int64_t num_layers, int64_t chunk_tokens,
    int64_t h_merged, int qbit, int blocks_num, int block_size, int head_num,
    int head_dim, bool rh, bool asym, const std::string& scaling_method,
    torch::ScalarType output_dtype, c10::optional<torch::Tensor> signs_opt,
    c10::optional<torch::Tensor> perm_opt, int num_threads = 0) {
  ensure_cpu_tensor(q_data, "q_data");
  auto q_contig = q_data.contiguous();

  auto q_dtype = (qbit <= 8) ? torch::kInt8 : torch::kInt16;
  if (q_contig.scalar_type() != q_dtype) {
    throw std::invalid_argument("q_data dtype does not match qbit.");
  }

  int64_t half_numel = num_layers * chunk_tokens * h_merged;
  int64_t half_q_numel = quantized_numel_for_qbit(half_numel, qbit);
  int64_t expected_numel = 2 * half_q_numel;
  if (q_contig.numel() < expected_numel) {
    throw std::invalid_argument(
        "q_data is smaller than expected for K/V payload.");
  }

  auto result = torch::empty({2, num_layers, chunk_tokens, h_merged},
                             q_contig.options().dtype(output_dtype));

  auto q_k =
      torch::from_blob(q_contig.data_ptr(), {half_q_numel}, q_contig.options());
  auto q_v = torch::from_blob(static_cast<char*>(q_contig.data_ptr()) +
                                  half_q_numel * q_contig.element_size(),
                              {half_q_numel}, q_contig.options());
  auto out_k =
      torch::from_blob(result.data_ptr(), {half_numel}, result.options());
  auto out_v = torch::from_blob(static_cast<char*>(result.data_ptr()) +
                                    half_numel * result.element_size(),
                                {half_numel}, result.options());

  const bool per_layer_quant =
      use_per_layer_quantization(num_layers, scaling_method);
  const int effective_blocks_num = blocks_num;

  auto args = build_quant_args(qbit, effective_blocks_num, block_size, head_num,
                               head_dim, rh, asym, scaling_method);
  args.num_threads = num_threads;

  torch::Tensor signs_t, perm_t;
  attach_precond(args, half_numel, num_layers, signs_opt, perm_opt, signs_t,
                 perm_t);

  constexpr int64_t scale_id_k = 0xD001;
  constexpr int64_t scale_id_v = 0xD002;

  const std::string k_scales = std::string(k_scale_bytes);
  const std::string v_scales = std::string(v_scale_bytes);

  if (per_layer_quant) {
    auto k_layer_scales = split_scale_blob(k_scales, num_layers);
    auto v_layer_scales = split_scale_blob(v_scales, num_layers);

#ifdef USE_MULTITHREADING
    if (args.num_threads >= 2) {
      ensure_nested_openmp_for_fused_halves(args.num_threads);
      std::exception_ptr k_error = nullptr;
      std::exception_ptr v_error = nullptr;
      quantization_args k_args = args;
      quantization_args v_args = args;
      const int inner_threads = std::max(1, args.num_threads / 2);
      k_args.num_threads = inner_threads;
      v_args.num_threads = inner_threads;

      Py_BEGIN_ALLOW_THREADS
  #pragma omp parallel sections num_threads(2)
      {
  #pragma omp section
        {
          try {
            dequantize_one_half_per_layer(q_k, out_k, scale_id_k,
                                          k_layer_scales, num_layers, k_args);
          } catch (...) {
            k_error = std::current_exception();
          }
        }
  #pragma omp section
        {
          try {
            dequantize_one_half_per_layer(q_v, out_v, scale_id_v,
                                          v_layer_scales, num_layers, v_args);
          } catch (...) {
            v_error = std::current_exception();
          }
        }
      }
      Py_END_ALLOW_THREADS

          if (k_error) std::rethrow_exception(k_error);
      if (v_error) std::rethrow_exception(v_error);
    } else
#endif
    {
      Py_BEGIN_ALLOW_THREADS dequantize_one_half_per_layer(
          q_k, out_k, scale_id_k, k_layer_scales, num_layers, args);
      dequantize_one_half_per_layer(q_v, out_v, scale_id_v, v_layer_scales,
                                    num_layers, args);
      Py_END_ALLOW_THREADS
    }
  } else {
#ifdef USE_MULTITHREADING
    if (args.num_threads >= 2) {
      ensure_nested_openmp_for_fused_halves(args.num_threads);
      std::exception_ptr k_error = nullptr;
      std::exception_ptr v_error = nullptr;
      quantization_args k_args = args;
      quantization_args v_args = args;
      const int inner_threads = std::max(1, args.num_threads / 2);
      k_args.num_threads = inner_threads;
      v_args.num_threads = inner_threads;

      Py_BEGIN_ALLOW_THREADS
  #pragma omp parallel sections num_threads(2)
      {
  #pragma omp section
        {
          try {
            dequantize_one_half(q_k, out_k, scale_id_k, k_scales, k_args);
          } catch (...) {
            k_error = std::current_exception();
          }
        }
  #pragma omp section
        {
          try {
            dequantize_one_half(q_v, out_v, scale_id_v, v_scales, v_args);
          } catch (...) {
            v_error = std::current_exception();
          }
        }
      }
      Py_END_ALLOW_THREADS

          if (k_error) std::rethrow_exception(k_error);
      if (v_error) std::rethrow_exception(v_error);
    } else
#endif
    {
      dequantize_one_half_releasing_gil(q_k, out_k, scale_id_k, k_scales, args);
      dequantize_one_half_releasing_gil(q_v, out_v, scale_id_v, v_scales, args);
    }
  }

  return result;
}

void gha_dequantize_chunk_into_4d_py(
    torch::Tensor q_data, pybind11::bytes k_scale_bytes,
    pybind11::bytes v_scale_bytes, torch::Tensor dst, int64_t num_layers,
    int64_t chunk_tokens, int64_t h_merged, int qbit, int blocks_num,
    int block_size, int head_num, int head_dim, bool rh, bool asym,
    const std::string& scaling_method, torch::ScalarType output_dtype,
    c10::optional<torch::Tensor> signs_opt,
    c10::optional<torch::Tensor> perm_opt, int num_threads = 0,
    int layer_num_threads = 1) {
  ensure_cpu_tensor(q_data, "q_data");
  ensure_cpu_tensor(dst, "dst");
  if (!dst.is_contiguous()) {
    throw std::invalid_argument("dst tensor must be contiguous.");
  }
  if (dst.dim() != 4 || dst.size(0) != 2 || dst.size(1) != num_layers ||
      dst.size(2) != chunk_tokens || dst.size(3) != h_merged) {
    throw std::invalid_argument(
        "dst must be contiguous [2, num_layers, chunk_tokens, h_merged].");
  }
  if (dst.scalar_type() != output_dtype) {
    throw std::invalid_argument("dst dtype must match output_dtype.");
  }

  auto q_contig = q_data.contiguous();
  auto q_dtype = (qbit <= 8) ? torch::kInt8 : torch::kInt16;
  if (q_contig.scalar_type() != q_dtype) {
    throw std::invalid_argument("q_data dtype does not match qbit.");
  }

  int64_t half_numel = num_layers * chunk_tokens * h_merged;
  int64_t half_q_numel = quantized_numel_for_qbit(half_numel, qbit);
  int64_t expected_numel = 2 * half_q_numel;
  if (q_contig.numel() < expected_numel) {
    throw std::invalid_argument(
        "q_data is smaller than expected for K/V payload.");
  }

  auto q_k =
      torch::from_blob(q_contig.data_ptr(), {half_q_numel}, q_contig.options());
  auto q_v = torch::from_blob(static_cast<char*>(q_contig.data_ptr()) +
                                  half_q_numel * q_contig.element_size(),
                              {half_q_numel}, q_contig.options());

  auto out_k = torch::from_blob(dst.data_ptr(), {half_numel}, dst.options());
  auto out_v = torch::from_blob(
      static_cast<char*>(dst.data_ptr()) + half_numel * dst.element_size(),
      {half_numel}, dst.options());

  const bool per_layer_quant =
      use_per_layer_quantization(num_layers, scaling_method);
  const int effective_blocks_num = blocks_num;

  auto args = build_quant_args(qbit, effective_blocks_num, block_size, head_num,
                               head_dim, rh, asym, scaling_method);
  args.num_threads = num_threads;

  torch::Tensor signs_t, perm_t;
  attach_precond(args, half_numel, num_layers, signs_opt, perm_opt, signs_t,
                 perm_t);

  constexpr int64_t scale_id_k = 0xD201;
  constexpr int64_t scale_id_v = 0xD202;
  const std::string k_scales = std::string(k_scale_bytes);
  const std::string v_scales = std::string(v_scale_bytes);

  if (per_layer_quant) {
    auto k_layer_scales = split_scale_blob(k_scales, num_layers);
    auto v_layer_scales = split_scale_blob(v_scales, num_layers);

    if (layer_num_threads > 1) {
      Py_BEGIN_ALLOW_THREADS dequantize_one_half_per_layer(
          q_k, out_k, scale_id_k, k_layer_scales, num_layers, args,
          layer_num_threads);
      dequantize_one_half_per_layer(q_v, out_v, scale_id_v, v_layer_scales,
                                    num_layers, args, layer_num_threads);
      Py_END_ALLOW_THREADS return;
    }

#ifdef USE_MULTITHREADING
    if (args.num_threads >= 2) {
      ensure_nested_openmp_for_fused_halves(args.num_threads);
      std::exception_ptr k_error = nullptr;
      std::exception_ptr v_error = nullptr;
      quantization_args k_args = args;
      quantization_args v_args = args;
      const int inner_threads = std::max(1, args.num_threads / 2);
      k_args.num_threads = inner_threads;
      v_args.num_threads = inner_threads;

      Py_BEGIN_ALLOW_THREADS
  #pragma omp parallel sections num_threads(2)
      {
  #pragma omp section
        {
          try {
            dequantize_one_half_per_layer(q_k, out_k, scale_id_k,
                                          k_layer_scales, num_layers, k_args);
          } catch (...) {
            k_error = std::current_exception();
          }
        }
  #pragma omp section
        {
          try {
            dequantize_one_half_per_layer(q_v, out_v, scale_id_v,
                                          v_layer_scales, num_layers, v_args);
          } catch (...) {
            v_error = std::current_exception();
          }
        }
      }
      Py_END_ALLOW_THREADS

          if (k_error) std::rethrow_exception(k_error);
      if (v_error) std::rethrow_exception(v_error);
      return;
    }
#endif
    Py_BEGIN_ALLOW_THREADS dequantize_one_half_per_layer(
        q_k, out_k, scale_id_k, k_layer_scales, num_layers, args);
    dequantize_one_half_per_layer(q_v, out_v, scale_id_v, v_layer_scales,
                                  num_layers, args);
    Py_END_ALLOW_THREADS return;
  }

#ifdef USE_MULTITHREADING
  if (args.num_threads >= 2) {
    ensure_nested_openmp_for_fused_halves(args.num_threads);
    std::exception_ptr k_error = nullptr;
    std::exception_ptr v_error = nullptr;
    quantization_args k_args = args;
    quantization_args v_args = args;
    const int inner_threads = std::max(1, args.num_threads / 2);
    k_args.num_threads = inner_threads;
    v_args.num_threads = inner_threads;

    Py_BEGIN_ALLOW_THREADS
  #pragma omp parallel sections num_threads(2)
    {
  #pragma omp section
      {
        try {
          dequantize_one_half(q_k, out_k, scale_id_k, k_scales, k_args);
        } catch (...) {
          k_error = std::current_exception();
        }
      }
  #pragma omp section
      {
        try {
          dequantize_one_half(q_v, out_v, scale_id_v, v_scales, v_args);
        } catch (...) {
          v_error = std::current_exception();
        }
      }
    }
    Py_END_ALLOW_THREADS

        if (k_error) std::rethrow_exception(k_error);
    if (v_error) std::rethrow_exception(v_error);
    return;
  }
#endif

  dequantize_one_half_releasing_gil(q_k, out_k, scale_id_k, k_scales, args);
  dequantize_one_half_releasing_gil(q_v, out_v, scale_id_v, v_scales, args);
}

// Quantize K+V and pack the GHA2 payload in C++. The GIL is released for each
// kernel.
//
// Arguments:
//   src          : [2, L, T, H*D] contiguous CPU tensor (fp16/bf16/fp32)
//   header_bytes : GHA2 header bytes (magic + metadata + precond + shape),
//                  built once in Python and reused across calls.
//   scale_id_k   : scale ID for K quantization (unique per call)
//   scale_id_v   : scale ID for V quantization (unique per call)
//   qbit, blocks_num, block_size, head_num, head_dim, rh, asym, scaling_method,
//   signs, perm, num_threads : quantization parameters.
//
// Returns: complete GHA2 payload bytes (header + scale_section + q_data),
//          ready for direct disk write.
pybind11::object gha_serialize_chunk_py(
    torch::Tensor src, pybind11::bytes header_bytes, int64_t scale_id_k,
    int64_t scale_id_v, int qbit, int blocks_num, int block_size, int head_num,
    int head_dim, int64_t num_layers, bool rh, bool asym,
    const std::string& scaling_method, c10::optional<torch::Tensor> signs_opt,
    c10::optional<torch::Tensor> perm_opt, int num_threads,
    c10::optional<torch::Tensor> out_opt) {
  ensure_cpu_tensor(src, "src");

  // Contiguous copy; use from_blob to avoid Python-backed views.
  auto src_contig = src.contiguous();
  int64_t half_numel = src_contig.numel() / 2;

  // Slice K and V halves via from_blob (pure C++ tensors, no Python backing).
  auto src_k = torch::from_blob(src_contig.data_ptr(), {half_numel},
                                src_contig.options());
  auto src_v = torch::from_blob(static_cast<char*>(src_contig.data_ptr()) +
                                    half_numel * src_contig.element_size(),
                                {half_numel}, src_contig.options());

  const bool per_layer_quant =
      use_per_layer_quantization(num_layers, scaling_method);
  const int effective_blocks_num = blocks_num;

  auto args = build_quant_args(qbit, effective_blocks_num, block_size, head_num,
                               head_dim, rh, asym, scaling_method);
  args.num_threads = num_threads;

  // Keep sign/perm tensors alive for the duration of quantization.
  torch::Tensor signs_t, perm_t;
  attach_precond(args, half_numel, num_layers, signs_opt, perm_opt, signs_t,
                 perm_t);

  // Payload: header | BE32 k_len | k_scales | BE32 v_len | v_scales | q_k |
  // q_v. Every size is known up front (pack_scales emits scale_count(args)
  // entries per segment), so the quantized halves are written straight into
  // the final buffer -- the caller's ``out`` or a new bytes object.
  std::string hdr_str(header_bytes);
  const int64_t blob_segments = per_layer_quant ? num_layers : 1;
  const size_t blob_len =
      static_cast<size_t>(blob_segments) *
      (sizeof(uint32_t) + static_cast<size_t>(scale_count(args)) *
                              (sizeof(uint32_t) + 2 * sizeof(float)));
  if (blob_len > UINT32_MAX) {
    throw std::runtime_error(
        "gha scale blob exceeds uint32 length framing limit");
  }
  const size_t item_bytes = (qbit <= 8) ? 1 : 2;
  const int64_t q_half_numel = quantized_numel_for_qbit(half_numel, qbit);
  const size_t q_half_bytes = static_cast<size_t>(q_half_numel) * item_bytes;
  const size_t total =
      hdr_str.size() + 2 * (sizeof(uint32_t) + blob_len) + 2 * q_half_bytes;

  pybind11::object result;
  uint8_t* base = nullptr;
  if (out_opt.has_value() && out_opt->defined()) {
    auto& out = *out_opt;
    ensure_cpu_tensor(out, "out");
    if (out.scalar_type() != torch::kUInt8 || !out.is_contiguous() ||
        static_cast<size_t>(out.numel()) < total) {
      throw std::invalid_argument(
          "out must be a contiguous uint8 tensor large enough for the payload");
    }
    base = out.data_ptr<uint8_t>();
    result = pybind11::int_(total);
  } else {
    PyObject* raw =
        PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(total));
    if (raw == nullptr) {
      throw pybind11::error_already_set();
    }
    result = pybind11::reinterpret_steal<pybind11::object>(raw);
    base = reinterpret_cast<uint8_t*>(PyBytes_AS_STRING(raw));
  }
  uint8_t* k_len_at = base + hdr_str.size();
  uint8_t* v_len_at = k_len_at + sizeof(uint32_t) + blob_len;
  uint8_t* q_at = v_len_at + sizeof(uint32_t) + blob_len;

  auto dst_dtype = (qbit <= 8) ? torch::kInt8 : torch::kInt16;
  auto q_opts = src_k.options().dtype(dst_dtype);
  // int16 output may sit at an odd offset: quantize into scratch, copy after.
  torch::Tensor q_k = item_bytes == 1
                          ? torch::from_blob(q_at, {q_half_numel}, q_opts)
                          : torch::empty({q_half_numel}, q_opts);
  torch::Tensor q_v = item_bytes == 1 ? torch::from_blob(q_at + q_half_bytes,
                                                         {q_half_numel}, q_opts)
                                      : torch::empty({q_half_numel}, q_opts);

  std::string k_scale_data;
  std::string v_scale_data;

  if (per_layer_quant) {
#ifdef USE_MULTITHREADING
    if (args.num_threads >= 2) {
      ensure_nested_openmp_for_fused_halves(args.num_threads);
      std::exception_ptr k_error = nullptr;
      std::exception_ptr v_error = nullptr;
      quantization_args k_args = args;
      quantization_args v_args = args;
      const int inner_threads = std::max(1, args.num_threads / 2);
      k_args.num_threads = inner_threads;
      v_args.num_threads = inner_threads;

      Py_BEGIN_ALLOW_THREADS
  #pragma omp parallel sections num_threads(2)
      {
  #pragma omp section
        {
          try {
            k_scale_data = quantize_one_half_per_layer(src_k, q_k, scale_id_k,
                                                       num_layers, k_args);
          } catch (...) {
            k_error = std::current_exception();
          }
        }
  #pragma omp section
        {
          try {
            v_scale_data = quantize_one_half_per_layer(src_v, q_v, scale_id_v,
                                                       num_layers, v_args);
          } catch (...) {
            v_error = std::current_exception();
          }
        }
      }
      Py_END_ALLOW_THREADS

          if (k_error) std::rethrow_exception(k_error);
      if (v_error) std::rethrow_exception(v_error);
    } else
#endif
    {
      Py_BEGIN_ALLOW_THREADS k_scale_data =
          quantize_one_half_per_layer(src_k, q_k, scale_id_k, num_layers, args);
      v_scale_data =
          quantize_one_half_per_layer(src_v, q_v, scale_id_v, num_layers, args);
      Py_END_ALLOW_THREADS
    }
  } else {
#ifdef USE_MULTITHREADING
    if (args.num_threads >= 2) {
      ensure_nested_openmp_for_fused_halves(args.num_threads);
      std::exception_ptr k_error = nullptr;
      std::exception_ptr v_error = nullptr;
      quantization_args k_args = args;
      quantization_args v_args = args;
      const int inner_threads = std::max(1, args.num_threads / 2);
      k_args.num_threads = inner_threads;
      v_args.num_threads = inner_threads;

      Py_BEGIN_ALLOW_THREADS
  #pragma omp parallel sections num_threads(2)
      {
  #pragma omp section
        {
          try {
            k_scale_data = quantize_one_half(src_k, q_k, scale_id_k, k_args);
          } catch (...) {
            k_error = std::current_exception();
          }
        }
  #pragma omp section
        {
          try {
            v_scale_data = quantize_one_half(src_v, q_v, scale_id_v, v_args);
          } catch (...) {
            v_error = std::current_exception();
          }
        }
      }
      Py_END_ALLOW_THREADS

          if (k_error) std::rethrow_exception(k_error);
      if (v_error) std::rethrow_exception(v_error);
    } else
#endif
    {
      Py_BEGIN_ALLOW_THREADS k_scale_data =
          quantize_one_half(src_k, q_k, scale_id_k, args);
      v_scale_data = quantize_one_half(src_v, q_v, scale_id_v, args);
      Py_END_ALLOW_THREADS
    }
  }

  if (k_scale_data.size() != blob_len || v_scale_data.size() != blob_len) {
    throw std::runtime_error(
        "gha scale blob size does not match its layout");
  }
  auto put_be32 = [](uint8_t* at, uint32_t v) {
    at[0] = static_cast<uint8_t>((v >> 24) & 0xFF);
    at[1] = static_cast<uint8_t>((v >> 16) & 0xFF);
    at[2] = static_cast<uint8_t>((v >> 8) & 0xFF);
    at[3] = static_cast<uint8_t>(v & 0xFF);
  };
  std::memcpy(base, hdr_str.data(), hdr_str.size());
  put_be32(k_len_at, static_cast<uint32_t>(blob_len));
  std::memcpy(k_len_at + sizeof(uint32_t), k_scale_data.data(), blob_len);
  put_be32(v_len_at, static_cast<uint32_t>(blob_len));
  std::memcpy(v_len_at + sizeof(uint32_t), v_scale_data.data(), blob_len);
  if (item_bytes != 1) {
    std::memcpy(q_at, q_k.data_ptr(), q_half_bytes);
    std::memcpy(q_at + q_half_bytes, q_v.data_ptr(), q_half_bytes);
  }
  return result;
}

// Single-tensor variant of gha_serialize_chunk_py (no K/V split).
pybind11::bytes gha_serialize_chunk_state_py(
    torch::Tensor src, pybind11::bytes header_bytes, int64_t scale_id, int qbit,
    int blocks_num, int block_size, int head_num, int head_dim,
    int64_t num_layers, bool rh, bool asym, const std::string& scaling_method,
    c10::optional<torch::Tensor> signs_opt,
    c10::optional<torch::Tensor> perm_opt, int num_threads = 0) {
  ensure_cpu_tensor(src, "src");

  auto src_contig = src.contiguous();
  int64_t numel = src_contig.numel();
  auto src_view =
      torch::from_blob(src_contig.data_ptr(), {numel}, src_contig.options());

  auto dst_dtype = (qbit <= 8) ? torch::kInt8 : torch::kInt16;
  auto q_data = torch::empty({quantized_numel_for_qbit(numel, qbit)},
                             src_contig.options().dtype(dst_dtype));

  const bool per_layer_quant =
      use_per_layer_quantization(num_layers, scaling_method);

  auto args = build_quant_args(qbit, blocks_num, block_size, head_num, head_dim,
                               rh, asym, scaling_method);
  args.num_threads = num_threads;

  torch::Tensor signs_t, perm_t;
  attach_precond(args, numel, num_layers, signs_opt, perm_opt, signs_t, perm_t);

  std::string scale_data;
  if (per_layer_quant) {
    Py_BEGIN_ALLOW_THREADS scale_data = quantize_one_half_per_layer(
        src_view, q_data, scale_id, num_layers, args);
    Py_END_ALLOW_THREADS
  } else {
    Py_BEGIN_ALLOW_THREADS scale_data =
        quantize_one_half(src_view, q_data, scale_id, args);
    Py_END_ALLOW_THREADS
  }

  // Pack: header | scale_len(4BE) | scales | q_data
  std::string hdr_str(header_bytes);
  if (scale_data.size() > UINT32_MAX) {
    throw std::runtime_error(
        "gha scale blob exceeds uint32 length framing limit");
  }
  uint32_t scale_len = static_cast<uint32_t>(scale_data.size());
  size_t item_bytes = (qbit <= 8) ? 1 : 2;

  size_t total = hdr_str.size() + 4u + scale_len +
                 static_cast<size_t>(q_data.numel()) * item_bytes;

  std::string out;
  out.reserve(total);
  out += hdr_str;

  auto append_be32 = [&](uint32_t v) {
    out += static_cast<char>((v >> 24) & 0xFF);
    out += static_cast<char>((v >> 16) & 0xFF);
    out += static_cast<char>((v >> 8) & 0xFF);
    out += static_cast<char>((v >> 0) & 0xFF);
  };

  append_be32(scale_len);
  out += scale_data;

  auto q_data_contig = q_data.contiguous();
  out.append(reinterpret_cast<const char*>(q_data_contig.data_ptr()),
             static_cast<size_t>(q_data_contig.numel()) * item_bytes);

  return pybind11::bytes(out);
}

// Single-tensor variant of gha_dequantize_chunk_py.
torch::Tensor gha_dequantize_chunk_state_py(
    torch::Tensor q_data, pybind11::bytes scale_bytes, int64_t num_layers,
    int64_t chunk_tokens, int64_t h_merged, int qbit, int blocks_num,
    int block_size, int head_num, int head_dim, bool rh, bool asym,
    const std::string& scaling_method, torch::ScalarType output_dtype,
    c10::optional<torch::Tensor> signs_opt,
    c10::optional<torch::Tensor> perm_opt, int num_threads = 0) {
  ensure_cpu_tensor(q_data, "q_data");
  auto q_contig = q_data.contiguous();

  auto q_dtype = (qbit <= 8) ? torch::kInt8 : torch::kInt16;
  if (q_contig.scalar_type() != q_dtype) {
    throw std::invalid_argument("q_data dtype does not match qbit.");
  }

  int64_t numel = num_layers * chunk_tokens * h_merged;
  int64_t q_numel = quantized_numel_for_qbit(numel, qbit);
  if (q_contig.numel() < q_numel) {
    throw std::invalid_argument(
        "q_data is smaller than expected for the state payload.");
  }

  auto result = torch::empty({num_layers, chunk_tokens, h_merged},
                             q_contig.options().dtype(output_dtype));

  auto q_view =
      torch::from_blob(q_contig.data_ptr(), {q_numel}, q_contig.options());
  auto out_view =
      torch::from_blob(result.data_ptr(), {numel}, result.options());

  const bool per_layer_quant =
      use_per_layer_quantization(num_layers, scaling_method);

  auto args = build_quant_args(qbit, blocks_num, block_size, head_num, head_dim,
                               rh, asym, scaling_method);
  args.num_threads = num_threads;

  torch::Tensor signs_t, perm_t;
  attach_precond(args, numel, num_layers, signs_opt, perm_opt, signs_t, perm_t);

  constexpr int64_t scale_id = 0xD003;
  const std::string scales = std::string(scale_bytes);

  if (per_layer_quant) {
    auto layer_scales = split_scale_blob(scales, num_layers);
    Py_BEGIN_ALLOW_THREADS dequantize_one_half_per_layer(
        q_view, out_view, scale_id, layer_scales, num_layers, args);
    Py_END_ALLOW_THREADS
  } else {
    dequantize_one_half_releasing_gil(q_view, out_view, scale_id, scales, args);
  }

  return result;
}

// Reorder [layers, tokens, H*D] into [layers*H, tokens, D] (OpenMP).
torch::Tensor gha_split_fused_heads_py(
    torch::Tensor src, int64_t head_num, int64_t head_dim, int num_threads = 0,
    c10::optional<torch::Tensor> out_opt = c10::nullopt) {
  ensure_cpu_tensor(src, "src");
  auto src_contig = src.contiguous();
  if (src_contig.dim() != 3) {
    throw std::invalid_argument(
        "src must be a 3-D [layers, tokens, head_num*head_dim] tensor");
  }
  int64_t layers = src_contig.size(0);
  int64_t tokens = src_contig.size(1);
  int64_t hidden = src_contig.size(2);
  if (hidden != head_num * head_dim) {
    throw std::invalid_argument(
        "src's trailing dim must equal head_num * head_dim");
  }

  torch::Tensor dst;
  if (out_opt.has_value() && out_opt->defined()) {
    dst = *out_opt;
    std::vector<int64_t> expected = {layers * head_num, tokens, head_dim};
    if (dst.sizes().vec() != expected ||
        dst.scalar_type() != src_contig.scalar_type() || !dst.is_contiguous()) {
      throw std::invalid_argument(
          "out does not match the expected split shape/dtype/contiguity");
    }
  } else {
    dst = torch::empty({layers * head_num, tokens, head_dim},
                       src_contig.options());
  }

  if (src_contig.scalar_type() == torch::kFloat8_e4m3fn ||
      src_contig.scalar_type() == torch::kFloat8_e5m2) {
    // fp8 is only moved, never converted: copy it as raw bytes.
    Py_BEGIN_ALLOW_THREADS split_fused_attention_heads_native<uint8_t>(
        static_cast<const uint8_t*>(src_contig.data_ptr()),
        static_cast<uint8_t*>(dst.data_ptr()), layers, tokens, head_num,
        head_dim, num_threads);
    Py_END_ALLOW_THREADS return dst;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, src_contig.scalar_type(),
      "gha_split_fused_heads", [&] {
        Py_BEGIN_ALLOW_THREADS split_fused_attention_heads_native<scalar_t>(
            src_contig.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(), layers,
            tokens, head_num, head_dim, num_threads);
        Py_END_ALLOW_THREADS
      });
  return dst;
}

// Inverse of gha_split_fused_heads_py: merges
// [layers*head_num, tokens, head_dim] back into [layers, tokens,
// head_num*head_dim].
torch::Tensor gha_merge_fused_heads_py(
    torch::Tensor src, int64_t layers, int64_t head_num, int64_t head_dim,
    int num_threads = 0, c10::optional<torch::Tensor> out_opt = c10::nullopt) {
  ensure_cpu_tensor(src, "src");
  auto src_contig = src.contiguous();
  if (src_contig.dim() != 3) {
    throw std::invalid_argument(
        "src must be a 3-D [layers*head_num, tokens, head_dim] tensor");
  }
  int64_t tokens = src_contig.size(1);
  if (src_contig.size(0) != layers * head_num ||
      src_contig.size(2) != head_dim) {
    throw std::invalid_argument(
        "src shape does not match layers*head_num, head_dim");
  }

  torch::Tensor dst;
  if (out_opt.has_value() && out_opt->defined()) {
    dst = *out_opt;
    std::vector<int64_t> expected = {layers, tokens, head_num * head_dim};
    if (dst.sizes().vec() != expected ||
        dst.scalar_type() != src_contig.scalar_type() || !dst.is_contiguous()) {
      throw std::invalid_argument(
          "out does not match the expected merged shape/dtype/contiguity");
    }
  } else {
    dst = torch::empty({layers, tokens, head_num * head_dim},
                       src_contig.options());
  }

  if (src_contig.scalar_type() == torch::kFloat8_e4m3fn ||
      src_contig.scalar_type() == torch::kFloat8_e5m2) {
    // fp8 is only moved, never converted: copy it as raw bytes.
    Py_BEGIN_ALLOW_THREADS merge_fused_attention_heads_native<uint8_t>(
        static_cast<const uint8_t*>(src_contig.data_ptr()),
        static_cast<uint8_t*>(dst.data_ptr()), layers, tokens, head_num,
        head_dim, num_threads);
    Py_END_ALLOW_THREADS return dst;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, src_contig.scalar_type(),
      "gha_merge_fused_heads", [&] {
        Py_BEGIN_ALLOW_THREADS merge_fused_attention_heads_native<scalar_t>(
            src_contig.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(), layers,
            tokens, head_num, head_dim, num_threads);
        Py_END_ALLOW_THREADS
      });
  return dst;
}

// Quantizes fused [layers, tokens, head_num*head_dim] attention without a head
// split. Payload bytes match gha_serialize_chunk_state run on the split
// [layers*head_num, tokens, head_dim] tensor (scale entries are emitted in
// index order).
pybind11::object gha_serialize_fused_chunk_direct_py(
    torch::Tensor src, pybind11::bytes header_bytes, int qbit, int64_t head_num,
    int64_t head_dim, bool rh, bool asym, const std::string& scaling_method,
    c10::optional<torch::Tensor> signs_opt,
    c10::optional<torch::Tensor> perm_opt, int num_threads,
    c10::optional<torch::Tensor> out_opt) {
  ensure_cpu_tensor(src, "src");
  auto src_contig = src.contiguous();
  if (src_contig.dim() != 3) {
    throw std::invalid_argument(
        "src must be a 3-D [layers, tokens, head_num*head_dim] tensor");
  }
  const int64_t layers = src_contig.size(0);
  const int64_t tokens = src_contig.size(1);
  const int64_t hidden = src_contig.size(2);
  if (layers <= 0 || tokens <= 0 || head_num <= 0 || head_dim <= 0 ||
      hidden != head_num * head_dim) {
    throw std::invalid_argument(
        "src's trailing dim must equal head_num * head_dim");
  }
  if (qbit < 2 || qbit > 8) {
    throw std::invalid_argument(
        "direct fused quantization supports qbit in [2, 8]");
  }
  if (scaling_method != "per_channel") {
    throw std::invalid_argument(
        "direct fused quantization supports per_channel only");
  }
  if (qbit == 4 && (head_dim % 2) != 0) {
    throw std::invalid_argument(
        "4-bit direct fused quantization requires an even head_dim");
  }
  const auto dtype = src_contig.scalar_type();
  if (dtype != torch::kHalf && dtype != torch::kBFloat16 &&
      dtype != torch::kFloat32 && dtype != torch::kFloat8_e4m3fn &&
      dtype != torch::kFloat8_e5m2) {
    throw std::invalid_argument(
        "src dtype must be float16, bfloat16, float32, float8_e4m3fn or "
        "float8_e5m2");
  }

  fused_direct_params params;
  params.layers = layers;
  params.tokens = tokens;
  params.head_num = head_num;
  params.head_dim = head_dim;
  params.qbit = qbit;
  params.rh = rh;
  params.asym = asym;
  params.num_threads = num_threads;

  const int64_t transform_len = tokens;
  if (rh && (transform_len & (transform_len - 1)) != 0) {
    throw std::invalid_argument(
        "rh (Hadamard) requires the transform length to be a power of 2");
  }
  torch::Tensor signs_t, perm_t;
  if (rh && signs_opt.has_value() && signs_opt->defined()) {
    signs_t = signs_opt->to(torch::kFloat32).contiguous();
    if (signs_t.numel() != transform_len) {
      throw std::invalid_argument(
          "signs length must equal the RH transform length");
    }
    params.signs = signs_t.data_ptr<float>();
  }
  if (rh && perm_opt.has_value() && perm_opt->defined()) {
    perm_t = perm_opt->to(torch::kInt32).contiguous();
    if (perm_t.numel() != transform_len) {
      throw std::invalid_argument(
          "perm length must equal the RH transform length");
    }
    const int32_t* perm = perm_t.data_ptr<int32_t>();
    for (int64_t i = 0; i < transform_len; ++i) {
      if (perm[i] < 0 || perm[i] >= transform_len) {
        throw std::invalid_argument("perm entry out of range");
      }
    }
    params.perm = perm;
  }

  const int64_t units = layers * head_num;
  const int64_t nchunks = head_dim;
  const int64_t unit_elems = tokens * head_dim;
  const int64_t unit_q_bytes = (qbit == 4) ? unit_elems / 2 : unit_elems;
  constexpr size_t entry_size = sizeof(uint32_t) + 2 * sizeof(float);
  const size_t scale_len =
      static_cast<size_t>(units) *
      (sizeof(uint32_t) + static_cast<size_t>(nchunks) * entry_size);
  if (scale_len > UINT32_MAX) {
    throw std::runtime_error(
        "gha scale blob exceeds uint32 length framing limit");
  }
  std::string hdr_str(header_bytes);
  const size_t q_bytes = static_cast<size_t>(units * unit_q_bytes);
  const size_t total = hdr_str.size() + 4u + scale_len + q_bytes;

  pybind11::object result;
  uint8_t* base = nullptr;
  if (out_opt.has_value() && out_opt->defined()) {
    auto& out = *out_opt;
    ensure_cpu_tensor(out, "out");
    if (out.scalar_type() != torch::kUInt8 || !out.is_contiguous() ||
        static_cast<size_t>(out.numel()) < total) {
      throw std::invalid_argument(
          "out must be a contiguous uint8 tensor large enough for the payload");
    }
    base = out.data_ptr<uint8_t>();
    result = pybind11::int_(total);
  } else {
    PyObject* raw =
        PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(total));
    if (raw == nullptr) {
      throw pybind11::error_already_set();
    }
    result = pybind11::reinterpret_steal<pybind11::object>(raw);
    base = reinterpret_cast<uint8_t*>(PyBytes_AS_STRING(raw));
  }

  std::memcpy(base, hdr_str.data(), hdr_str.size());
  uint8_t* cursor = base + hdr_str.size();
  const uint32_t scale_len32 = static_cast<uint32_t>(scale_len);
  cursor[0] = static_cast<uint8_t>((scale_len32 >> 24) & 0xFF);
  cursor[1] = static_cast<uint8_t>((scale_len32 >> 16) & 0xFF);
  cursor[2] = static_cast<uint8_t>((scale_len32 >> 8) & 0xFF);
  cursor[3] = static_cast<uint8_t>(scale_len32 & 0xFF);
  uint8_t* scale_blob = cursor + 4;
  int8_t* q_dst = reinterpret_cast<int8_t*>(scale_blob + scale_len);

  std::vector<float> scales(static_cast<size_t>(units * nchunks));
  std::vector<float> zps(static_cast<size_t>(units * nchunks));
  const void* src_ptr = src_contig.data_ptr();
  Py_BEGIN_ALLOW_THREADS if (dtype == torch::kHalf) {
    quantize_fused_direct<_Float16>(static_cast<const _Float16*>(src_ptr),
                                    q_dst, scales.data(), zps.data(), params);
  }
  else if (dtype == torch::kBFloat16) {
    quantize_fused_direct<__bf16>(static_cast<const __bf16*>(src_ptr), q_dst,
                                  scales.data(), zps.data(), params);
  }
  else if (dtype == torch::kFloat8_e4m3fn) {
    quantize_fused_direct<gha_fp8_e4m3>(
        static_cast<const gha_fp8_e4m3*>(src_ptr), q_dst, scales.data(),
        zps.data(), params);
  }
  else if (dtype == torch::kFloat8_e5m2) {
    quantize_fused_direct<gha_fp8_e5m2>(
        static_cast<const gha_fp8_e5m2*>(src_ptr), q_dst, scales.data(),
        zps.data(), params);
  }
  else {
    quantize_fused_direct<float>(static_cast<const float*>(src_ptr), q_dst,
                                 scales.data(), zps.data(), params);
  }
  Py_END_ALLOW_THREADS

      uint8_t* sp = scale_blob;
  const uint32_t count = static_cast<uint32_t>(nchunks);
  for (int64_t unit = 0; unit < units; ++unit) {
    std::memcpy(sp, &count, sizeof(uint32_t));
    sp += sizeof(uint32_t);
    for (int64_t k = 0; k < nchunks; ++k) {
      const uint32_t idx = static_cast<uint32_t>(k);
      const size_t pos = static_cast<size_t>(unit * nchunks + k);
      std::memcpy(sp, &idx, sizeof(uint32_t));
      sp += sizeof(uint32_t);
      std::memcpy(sp, &scales[pos], sizeof(float));
      sp += sizeof(float);
      std::memcpy(sp, &zps[pos], sizeof(float));
      sp += sizeof(float);
    }
  }
  return result;
}

// Inverse of gha_serialize_fused_chunk_direct: dequantizes per-(layer,
// head) unit planes and writes them straight into fused dst [layers, tokens,
// head_num*head_dim], skipping the split-plane temporary and head merge.
void gha_dequantize_fused_chunk_direct_py(
    torch::Tensor q_data, pybind11::bytes scale_bytes, torch::Tensor dst,
    int64_t layers, int64_t tokens, int64_t head_num, int64_t head_dim,
    int qbit, bool rh, bool asym, const std::string& scaling_method,
    c10::optional<torch::Tensor> signs_opt,
    c10::optional<torch::Tensor> perm_opt, int num_threads) {
  ensure_cpu_tensor(q_data, "q_data");
  ensure_cpu_tensor(dst, "dst");
  if (layers <= 0 || tokens <= 0 || head_num <= 0 || head_dim <= 0) {
    throw std::invalid_argument(
        "layers, tokens, head_num and head_dim must be positive");
  }
  if (qbit < 2 || qbit > 8) {
    throw std::invalid_argument(
        "direct fused dequantization supports qbit in [2, 8]");
  }
  if (scaling_method != "per_channel") {
    throw std::invalid_argument(
        "direct fused dequantization supports per_channel only");
  }
  if (qbit == 4 && (head_dim % 2) != 0) {
    throw std::invalid_argument(
        "4-bit direct fused dequantization requires an even head_dim");
  }
  const auto dtype = dst.scalar_type();
  if (dtype != torch::kHalf && dtype != torch::kBFloat16 &&
      dtype != torch::kFloat32 && dtype != torch::kFloat8_e4m3fn &&
      dtype != torch::kFloat8_e5m2) {
    throw std::invalid_argument(
        "dst dtype must be float16, bfloat16, float32, float8_e4m3fn or "
        "float8_e5m2");
  }
  if (!dst.is_contiguous() ||
      dst.numel() != layers * tokens * head_num * head_dim) {
    throw std::invalid_argument(
        "dst must be contiguous with layers*tokens*head_num*head_dim elements");
  }
  if (q_data.scalar_type() != torch::kInt8 &&
      q_data.scalar_type() != torch::kUInt8) {
    throw std::invalid_argument("q_data must be int8 or uint8");
  }
  auto q_contig = q_data.contiguous();
  const int64_t units = layers * head_num;
  const int64_t unit_elems = tokens * head_dim;
  const int64_t unit_q_bytes = (qbit == 4) ? unit_elems / 2 : unit_elems;
  // Trailing bytes (e.g. SHM slot padding) are allowed and ignored.
  if (q_contig.numel() < units * unit_q_bytes) {
    throw std::invalid_argument(
        "q_data is smaller than expected for the fused payload");
  }

  fused_direct_params params;
  params.layers = layers;
  params.tokens = tokens;
  params.head_num = head_num;
  params.head_dim = head_dim;
  params.qbit = qbit;
  params.rh = rh;
  params.asym = asym;
  params.num_threads = num_threads;

  const int64_t transform_len = tokens;
  if (rh && (transform_len & (transform_len - 1)) != 0) {
    throw std::invalid_argument(
        "rh (Hadamard) requires the transform length to be a power of 2");
  }
  torch::Tensor signs_t, perm_t;
  if (rh && signs_opt.has_value() && signs_opt->defined()) {
    signs_t = signs_opt->to(torch::kFloat32).contiguous();
    if (signs_t.numel() != transform_len) {
      throw std::invalid_argument(
          "signs length must equal the RH transform length");
    }
    params.signs = signs_t.data_ptr<float>();
  }
  if (rh && perm_opt.has_value() && perm_opt->defined()) {
    perm_t = perm_opt->to(torch::kInt32).contiguous();
    if (perm_t.numel() != transform_len) {
      throw std::invalid_argument(
          "perm length must equal the RH transform length");
    }
    const int32_t* perm = perm_t.data_ptr<int32_t>();
    for (int64_t i = 0; i < transform_len; ++i) {
      if (perm[i] < 0 || perm[i] >= transform_len) {
        throw std::invalid_argument("perm entry out of range");
      }
    }
    params.perm = perm;
  }

  const int64_t nchunks = head_dim;
  constexpr size_t entry_size = sizeof(uint32_t) + 2 * sizeof(float);
  const std::string blob(scale_bytes);
  std::vector<float> scales(static_cast<size_t>(units * nchunks), 0.0f);
  std::vector<float> zps(static_cast<size_t>(units * nchunks), 0.0f);
  size_t off = 0;
  for (int64_t unit = 0; unit < units; ++unit) {
    uint32_t count = 0;
    if (off + sizeof(uint32_t) > blob.size()) {
      throw std::invalid_argument("scale blob too short for the fused payload");
    }
    std::memcpy(&count, blob.data() + off, sizeof(uint32_t));
    off += sizeof(uint32_t);
    if (static_cast<int64_t>(count) != nchunks ||
        off + static_cast<size_t>(count) * entry_size > blob.size()) {
      throw std::invalid_argument(
          "scale blob does not match the fused payload layout");
    }
    for (uint32_t k = 0; k < count; ++k) {
      uint32_t idx = 0;
      std::memcpy(&idx, blob.data() + off, sizeof(uint32_t));
      if (idx >= count) {
        throw std::invalid_argument("scale blob chunk index out of range");
      }
      const size_t pos = static_cast<size_t>(unit * nchunks + idx);
      std::memcpy(&scales[pos], blob.data() + off + sizeof(uint32_t),
                  sizeof(float));
      std::memcpy(&zps[pos],
                  blob.data() + off + sizeof(uint32_t) + sizeof(float),
                  sizeof(float));
      off += entry_size;
    }
  }

  const int8_t* q_ptr = reinterpret_cast<const int8_t*>(q_contig.data_ptr());
  void* dst_ptr = dst.data_ptr();
  Py_BEGIN_ALLOW_THREADS if (dtype == torch::kHalf) {
    dequantize_fused_direct<_Float16>(q_ptr, static_cast<_Float16*>(dst_ptr),
                                      scales.data(), zps.data(), params);
  }
  else if (dtype == torch::kBFloat16) {
    dequantize_fused_direct<__bf16>(q_ptr, static_cast<__bf16*>(dst_ptr),
                                    scales.data(), zps.data(), params);
  }
  else if (dtype == torch::kFloat8_e4m3fn) {
    dequantize_fused_direct<gha_fp8_e4m3>(
        q_ptr, static_cast<gha_fp8_e4m3*>(dst_ptr), scales.data(),
        zps.data(), params);
  }
  else if (dtype == torch::kFloat8_e5m2) {
    dequantize_fused_direct<gha_fp8_e5m2>(
        q_ptr, static_cast<gha_fp8_e5m2*>(dst_ptr), scales.data(),
        zps.data(), params);
  }
  else {
    dequantize_fused_direct<float>(q_ptr, static_cast<float*>(dst_ptr),
                                   scales.data(), zps.data(), params);
  }
  Py_END_ALLOW_THREADS
}

PYBIND11_MODULE(gha_quant, m) {
  m.doc() = "PyBind11 bindings for gha quantization utilities";

  m.def("gha_dequantize_chunk", &gha_dequantize_chunk_py,
        pybind11::arg("q_data"), pybind11::arg("k_scale_bytes"),
        pybind11::arg("v_scale_bytes"), pybind11::arg("num_layers"),
        pybind11::arg("chunk_tokens"), pybind11::arg("h_merged"),
        pybind11::arg("qbit"), pybind11::arg("blocks_num"),
        pybind11::arg("block_size"), pybind11::arg("head_num"),
        pybind11::arg("head_dim"), pybind11::arg("rh") = false,
        pybind11::arg("asym") = false,
        pybind11::arg("scaling_method") = "per_token",
        pybind11::arg("output_dtype"),
        pybind11::arg("signs") = c10::optional<torch::Tensor>(),
        pybind11::arg("perm") = c10::optional<torch::Tensor>(),
        pybind11::arg("num_threads") = 0,
        R"doc(Dequantize K+V quant payloads and return [2, L, T, H] in one call.

This is a fused Python/C++ API boundary over the existing per-half kernels. It
reduces Python overhead and intermediate copies while preserving the same math.
)doc");

  m.def(
      "gha_dequantize_chunk_into_4d", &gha_dequantize_chunk_into_4d_py,
      pybind11::arg("q_data"), pybind11::arg("k_scale_bytes"),
      pybind11::arg("v_scale_bytes"), pybind11::arg("dst"),
      pybind11::arg("num_layers"), pybind11::arg("chunk_tokens"),
      pybind11::arg("h_merged"), pybind11::arg("qbit"),
      pybind11::arg("blocks_num"), pybind11::arg("block_size"),
      pybind11::arg("head_num"), pybind11::arg("head_dim"),
      pybind11::arg("rh") = false, pybind11::arg("asym") = false,
      pybind11::arg("scaling_method") = "per_token",
      pybind11::arg("output_dtype"),
      pybind11::arg("signs") = c10::optional<torch::Tensor>(),
      pybind11::arg("perm") = c10::optional<torch::Tensor>(),
      pybind11::arg("num_threads") = 0, pybind11::arg("layer_num_threads") = 1,
      R"doc(Dequantize K+V quant payloads directly into contiguous [2, L, T, H] dst.

This path writes each half into its final destination layout and avoids the
    extra allocate-and-copy step used by gha_dequantize_chunk().

    For multi-layer per_token/per_channel/per_tensor payloads, `layer_num_threads`
    can be used to parallelize the per-layer loop experimentally instead of relying
    only on the inner block-level dequant threading.
)doc");

  // All-in-one serialize (quantize + pack header+scales+qdata in C++).
  m.def("gha_serialize_chunk", &gha_serialize_chunk_py,
        pybind11::arg("src"), pybind11::arg("header_bytes"),
        pybind11::arg("scale_id_k"), pybind11::arg("scale_id_v"),
        pybind11::arg("qbit"), pybind11::arg("blocks_num"),
        pybind11::arg("block_size"), pybind11::arg("head_num"),
        pybind11::arg("head_dim"),
        pybind11::arg("num_layers") = static_cast<int64_t>(1),
        pybind11::arg("rh") = false, pybind11::arg("asym") = false,
        pybind11::arg("scaling_method") = "per_token",
        pybind11::arg("signs") = c10::optional<torch::Tensor>(),
        pybind11::arg("perm") = c10::optional<torch::Tensor>(),
        pybind11::arg("num_threads") = 0,
        pybind11::arg("out") = c10::optional<torch::Tensor>(),
        R"doc(Quantize K+V and pack complete GHA2 payload in C++.

Eliminates Python GIL-held buffer copies (bytearray() + tobytes()).
The GIL is released for each quantization kernel, allowing 8 workers
to run their quant kernels in parallel.

Arguments:
    src (torch.Tensor): [2, L, T, H*D] contiguous CPU tensor (fp16/bf16/fp32).
    header_bytes (bytes): Pre-built GHA2 header (magic+metadata+precond+shape).
    scale_id_k, scale_id_v (int): Unique scale IDs for K and V halves.
    qbit, blocks_num, block_size: Layout metadata.
    num_layers (int): Number of layers packed in the tensor. When > 1 with
        per_token/per_channel/per_tensor, quantization is always done
        per-layer, so each layer gets its own independent scale (also
        satisfies the power-of-2 Hadamard constraint when rh is enabled).
    head_num, head_dim: Head layout metadata.
    rh (bool): Use RH transform.
    asym (bool): Use asymmetric quantization.
    scaling_method (str): "per_token" or "per_channel".
    signs, perm (torch.Tensor, optional): Preconditioning tensors.
    num_threads (int): OpenMP threads. 0 = omp_get_max_threads().

Returns: bytes — complete GHA2 payload (header + scales + q_data); with
         ``out`` (contiguous uint8, large enough) the payload is written
         there instead and its length is returned.
)doc");

  m.def(
      "gha_serialize_chunk_state", &gha_serialize_chunk_state_py,
      pybind11::arg("src"), pybind11::arg("header_bytes"),
      pybind11::arg("scale_id"), pybind11::arg("qbit"),
      pybind11::arg("blocks_num"), pybind11::arg("block_size"),
      pybind11::arg("head_num"), pybind11::arg("head_dim"),
      pybind11::arg("num_layers") = static_cast<int64_t>(1),
      pybind11::arg("rh") = false, pybind11::arg("asym") = false,
      pybind11::arg("scaling_method") = "per_token",
      pybind11::arg("signs") = c10::optional<torch::Tensor>(),
      pybind11::arg("perm") = c10::optional<torch::Tensor>(),
      pybind11::arg("num_threads") = 0,
      R"doc(Quantize a single tensor (no K/V split) and pack a complete payload in C++.

Single-tensor analog of gha_serialize_chunk, for payloads with no real
K/V split (e.g. fused-K/V attention chunks).

Arguments:
    src (torch.Tensor): contiguous CPU tensor (fp16/bf16/fp32), no leading K/V axis.
    header_bytes (bytes): Pre-built header (magic+metadata+precond+shape).
    scale_id (int): Unique scale ID for this call.
    qbit, blocks_num, block_size, head_num, head_dim: Layout metadata.
    num_layers (int): Number of layers packed in the tensor. When > 1 with
        per_token/per_channel/per_tensor, quantization is always done
        per-layer, so each layer gets its own independent scale (also
        satisfies the power-of-2 Hadamard constraint when rh is enabled).
    rh (bool): Use RH transform.
    asym (bool): Use asymmetric quantization.
    scaling_method (str): "per_token", "per_channel", or "per_tensor".
    signs, perm (torch.Tensor, optional): Preconditioning tensors.
    num_threads (int): OpenMP threads. 0 = omp_get_max_threads().

    rh (bool): Apply randomized Hadamard preconditioning. RH uses a
        contiguous native scratch sub-state; the RH-disabled path remains
        zero-copy. ``signs`` and ``perm`` are required when RH is enabled.

Returns: bytes — complete payload (header + scale_section + q_data).
)doc");

  m.def(
      "gha_dequantize_chunk_state", &gha_dequantize_chunk_state_py,
      pybind11::arg("q_data"), pybind11::arg("scale_bytes"),
      pybind11::arg("num_layers"), pybind11::arg("chunk_tokens"),
      pybind11::arg("h_merged"), pybind11::arg("qbit"),
      pybind11::arg("blocks_num"), pybind11::arg("block_size"),
      pybind11::arg("head_num"), pybind11::arg("head_dim"),
      pybind11::arg("rh") = false, pybind11::arg("asym") = false,
      pybind11::arg("scaling_method") = "per_token",
      pybind11::arg("output_dtype"),
      pybind11::arg("signs") = c10::optional<torch::Tensor>(),
      pybind11::arg("perm") = c10::optional<torch::Tensor>(),
      pybind11::arg("num_threads") = 0,
      R"doc(Dequantize a single-tensor (no K/V split) quant payload in one call.

Single-tensor analog of gha_dequantize_chunk, for payloads with no real
K/V split (e.g. fused-K/V attention chunks). Returns shape
[num_layers, chunk_tokens, h_merged].
)doc");

  m.def("gha_split_fused_heads", &gha_split_fused_heads_py,
        pybind11::arg("src"), pybind11::arg("head_num"),
        pybind11::arg("head_dim"), pybind11::arg("num_threads") = 0,
        pybind11::arg("out") = c10::optional<torch::Tensor>(),
        R"doc(Reorder [layers, tokens, head_num*head_dim] into
[layers*head_num, tokens, head_dim] via a native OpenMP-parallel copy --
an alternative to _GHACodec.split_fused_attention_heads's ATen
permute()/reshape()/copy_() for the same reorder. Pass `out` to write
directly into a pre-allocated buffer instead of allocating a fresh tensor.
)doc");

  m.def("gha_merge_fused_heads", &gha_merge_fused_heads_py,
        pybind11::arg("src"), pybind11::arg("layers"),
        pybind11::arg("head_num"), pybind11::arg("head_dim"),
        pybind11::arg("num_threads") = 0,
        pybind11::arg("out") = c10::optional<torch::Tensor>(),
        R"doc(Inverse of gha_split_fused_heads: merges
[layers*head_num, tokens, head_dim] back into [layers, tokens, head_num*head_dim].
Pass `out` to write directly into a pre-allocated buffer.
)doc");

  m.def("gha_serialize_fused_chunk_direct",
        &gha_serialize_fused_chunk_direct_py, pybind11::arg("src"),
        pybind11::arg("header_bytes"), pybind11::arg("qbit"),
        pybind11::arg("head_num"), pybind11::arg("head_dim"),
        pybind11::arg("rh") = false, pybind11::arg("asym") = false,
        pybind11::arg("scaling_method") = "per_channel",
        pybind11::arg("signs") = c10::optional<torch::Tensor>(),
        pybind11::arg("perm") = c10::optional<torch::Tensor>(),
        pybind11::arg("num_threads") = 0,
        pybind11::arg("out") = c10::optional<torch::Tensor>(),
        R"doc(Quantize fused [layers, tokens, head_num*head_dim] attention per
(layer, head) without a separate head-split copy. Produces the same payload as
gha_serialize_chunk_state on the split tensor with num_layers=layers*head_num.
Supports per_channel only. Returns bytes, or the written length when `out`
(contiguous uint8 tensor) is given.
)doc");

  m.def(
      "gha_dequantize_fused_chunk_direct",
      &gha_dequantize_fused_chunk_direct_py, pybind11::arg("q_data"),
      pybind11::arg("scale_bytes"), pybind11::arg("dst"),
      pybind11::arg("layers"), pybind11::arg("tokens"),
      pybind11::arg("head_num"), pybind11::arg("head_dim"),
      pybind11::arg("qbit"), pybind11::arg("rh") = false,
      pybind11::arg("asym") = false,
      pybind11::arg("scaling_method") = "per_channel",
      pybind11::arg("signs") = c10::optional<torch::Tensor>(),
      pybind11::arg("perm") = c10::optional<torch::Tensor>(),
      pybind11::arg("num_threads") = 0,
      R"doc(Inverse of gha_serialize_fused_chunk_direct: dequantize the per-(layer,
head) q planes + scale blob and write straight into the contiguous fused dst
[layers, tokens, head_num*head_dim] (no split temporary, no head merge).
Supports per_channel only.
)doc");
}
