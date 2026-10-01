// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(LMCACHE_COMPAT_NVIDIA) == defined(LMCACHE_COMPAT_AMD)
  #error "Define exactly one vendor compatibility target"
#endif

#if defined(LMCACHE_COMPAT_NVIDIA)
  #include <cuda_runtime_api.h>
  #include <nvcomp.h>
  #include <nvcomp/deflate.h>
  #include <nvcomp/gzip.h>

using GpuError = cudaError_t;
using GpuStreamHandle = cudaStream_t;
using VendorStatus = nvcompStatus_t;

constexpr GpuError kGpuSuccess = cudaSuccess;
constexpr VendorStatus kVendorSuccess = nvcompSuccess;
constexpr const char* kVendorName = "nvCOMP";
#else
  #include <hip/hip_runtime.h>
  #include <hipcomp.h>
  #include <hipcomp/deflate.h>
  #include <hipcomp/gzip.h>

using GpuError = hipError_t;
using GpuStreamHandle = hipStream_t;
using VendorStatus = hipcompStatus_t;

constexpr GpuError kGpuSuccess = hipSuccess;
constexpr VendorStatus kVendorSuccess = hipcompSuccess;
constexpr const char* kVendorName = "hipCOMP";
#endif

namespace {

constexpr std::size_t kFixedHeaderSize = 44;
constexpr std::size_t kChunkDescriptorSize = 24;
constexpr std::size_t kPayloadAlignment = 16;

enum class Framing : std::uint8_t {
  kRaw = 1,
  kGzip = 2,
};

struct ChunkDescriptor {
  std::size_t payload_offset;
  std::size_t compressed_size;
  std::size_t uncompressed_size;
};

struct ParsedRecord {
  std::vector<std::uint8_t> bytes;
  Framing framing;
  std::vector<ChunkDescriptor> chunks;
  std::size_t uncompressed_size;
};

struct AlignmentRequirements {
  std::size_t input;
  std::size_t output;
  std::size_t temporary;
};

std::string gpu_error_string(GpuError status) {
#if defined(LMCACHE_COMPAT_NVIDIA)
  return cudaGetErrorString(status);
#else
  return hipGetErrorString(status);
#endif
}

std::string vendor_error_string(VendorStatus status) {
#if defined(LMCACHE_COMPAT_NVIDIA)
  return nvcompGetStatusString(status);
#else
  return "hipCOMP status " + std::to_string(static_cast<int>(status));
#endif
}

void check_gpu(GpuError status, const char* operation) {
  if (status != kGpuSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + gpu_error_string(status));
  }
}

void check_vendor(VendorStatus status, const char* operation) {
  if (status != kVendorSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + vendor_error_string(status));
  }
}

void gpu_allocate(void** pointer, std::size_t size) {
#if defined(LMCACHE_COMPAT_NVIDIA)
  check_gpu(cudaMalloc(pointer, size), "cudaMalloc");
#else
  check_gpu(hipMalloc(pointer, size), "hipMalloc");
#endif
}

void gpu_free(void* pointer) {
#if defined(LMCACHE_COMPAT_NVIDIA)
  static_cast<void>(cudaFree(pointer));
#else
  static_cast<void>(hipFree(pointer));
#endif
}

void copy_to_device(void* destination, const void* source, std::size_t size) {
#if defined(LMCACHE_COMPAT_NVIDIA)
  check_gpu(cudaMemcpy(destination, source, size, cudaMemcpyHostToDevice),
            "cudaMemcpy host to device");
#else
  check_gpu(hipMemcpy(destination, source, size, hipMemcpyHostToDevice),
            "hipMemcpy host to device");
#endif
}

void copy_from_device(void* destination, const void* source, std::size_t size) {
#if defined(LMCACHE_COMPAT_NVIDIA)
  check_gpu(cudaMemcpy(destination, source, size, cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host");
#else
  check_gpu(hipMemcpy(destination, source, size, hipMemcpyDeviceToHost),
            "hipMemcpy device to host");
#endif
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t size) {
    if (size != 0) {
      gpu_allocate(&pointer_, size);
    }
  }

  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      gpu_free(pointer_);
    }
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  void* get() const { return pointer_; }

  template <typename T>
  T* as() const {
    return static_cast<T*>(pointer_);
  }

 private:
  void* pointer_ = nullptr;
};

class GpuStream {
 public:
  GpuStream() {
#if defined(LMCACHE_COMPAT_NVIDIA)
    check_gpu(cudaStreamCreate(&stream_), "cudaStreamCreate");
#else
    check_gpu(hipStreamCreate(&stream_), "hipStreamCreate");
#endif
  }

  ~GpuStream() {
#if defined(LMCACHE_COMPAT_NVIDIA)
    static_cast<void>(cudaStreamDestroy(stream_));
#else
    static_cast<void>(hipStreamDestroy(stream_));
#endif
  }

  GpuStream(const GpuStream&) = delete;
  GpuStream& operator=(const GpuStream&) = delete;

  GpuStreamHandle get() const { return stream_; }

  void synchronize() const {
#if defined(LMCACHE_COMPAT_NVIDIA)
    check_gpu(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
#else
    check_gpu(hipStreamSynchronize(stream_), "hipStreamSynchronize");
#endif
  }

 private:
  GpuStreamHandle stream_{};
};

int hex_digit(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  return -1;
}

std::vector<std::uint8_t> read_hex_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open fixture " + path);
  }

  std::vector<std::uint8_t> bytes;
  int high_nibble = -1;
  char value = 0;
  while (input.get(value)) {
    if (std::isspace(static_cast<unsigned char>(value)) != 0) {
      continue;
    }
    const int digit = hex_digit(value);
    if (digit < 0) {
      throw std::runtime_error("fixture contains a non-hex character: " + path);
    }
    if (high_nibble < 0) {
      high_nibble = digit;
    } else {
      bytes.push_back(static_cast<std::uint8_t>((high_nibble << 4) | digit));
      high_nibble = -1;
    }
  }
  if (high_nibble >= 0) {
    throw std::runtime_error("fixture contains an incomplete hex byte: " +
                             path);
  }
  return bytes;
}

void require_range(const std::vector<std::uint8_t>& bytes, std::size_t offset,
                   std::size_t size) {
  if (offset > bytes.size() || size > bytes.size() - offset) {
    throw std::runtime_error("fixture field is outside the record");
  }
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& bytes,
                       std::size_t offset) {
  require_range(bytes, offset, sizeof(std::uint32_t));
  std::uint32_t value = 0;
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    value |= static_cast<std::uint32_t>(bytes[offset + index]) << (index * 8);
  }
  return value;
}

std::uint64_t read_u64(const std::vector<std::uint8_t>& bytes,
                       std::size_t offset) {
  require_range(bytes, offset, sizeof(std::uint64_t));
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    value |= static_cast<std::uint64_t>(bytes[offset + index]) << (index * 8);
  }
  return value;
}

std::size_t as_size(std::uint64_t value, const char* field) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error(std::string(field) + " does not fit size_t");
  }
  return static_cast<std::size_t>(value);
}

std::size_t align_payload_offset(std::size_t offset) {
  if (offset >
      std::numeric_limits<std::size_t>::max() - (kPayloadAlignment - 1)) {
    throw std::runtime_error("payload offset alignment overflows");
  }
  return ((offset + kPayloadAlignment - 1) / kPayloadAlignment) *
         kPayloadAlignment;
}

ParsedRecord parse_record(const std::string& path, Framing expected_framing) {
  std::vector<std::uint8_t> bytes = read_hex_file(path);
  require_range(bytes, 0, kFixedHeaderSize);
  if (bytes[0] != 'L' || bytes[1] != 'M' || bytes[2] != 'C' ||
      bytes[3] != 'R') {
    throw std::runtime_error("fixture has the wrong magic");
  }
  if (bytes[4] != 1 || bytes[5] != 1 || bytes[7] != 0) {
    throw std::runtime_error("fixture is not version-1 untransformed Deflate");
  }
  if (bytes[6] != static_cast<std::uint8_t>(expected_framing)) {
    throw std::runtime_error("fixture has the wrong framing");
  }
  if (read_u32(bytes, 40) != 0) {
    throw std::runtime_error("fixture has unsupported header flags");
  }

  const std::size_t record_size = as_size(read_u64(bytes, 12), "record_size");
  if (record_size != bytes.size()) {
    throw std::runtime_error("fixture record_size does not match its length");
  }

  const std::size_t chunk_count = read_u32(bytes, 28);
  if (chunk_count >
      (std::numeric_limits<std::size_t>::max() - kFixedHeaderSize) /
          kChunkDescriptorSize) {
    throw std::runtime_error("fixture chunk table overflows size_t");
  }
  const std::size_t expected_header_size =
      kFixedHeaderSize + chunk_count * kChunkDescriptorSize;
  if (read_u32(bytes, 8) != expected_header_size) {
    throw std::runtime_error("fixture header_size is inconsistent");
  }

  std::vector<ChunkDescriptor> chunks;
  chunks.reserve(chunk_count);
  std::size_t previous_end = expected_header_size;
  std::size_t uncompressed_size = 0;
  for (std::size_t index = 0; index < chunk_count; ++index) {
    const std::size_t descriptor_offset =
        kFixedHeaderSize + index * kChunkDescriptorSize;
    const std::size_t payload_offset =
        as_size(read_u64(bytes, descriptor_offset), "payload_offset");
    const std::size_t compressed_size = read_u32(bytes, descriptor_offset + 8);
    const std::size_t chunk_uncompressed_size =
        read_u32(bytes, descriptor_offset + 12);
    if (read_u32(bytes, descriptor_offset + 20) != 0) {
      throw std::runtime_error("fixture has unsupported chunk flags");
    }
    if (payload_offset != align_payload_offset(previous_end)) {
      throw std::runtime_error("fixture payload offset is not canonical");
    }
    require_range(bytes, payload_offset, compressed_size);
    for (std::size_t padding_offset = previous_end;
         padding_offset < payload_offset; ++padding_offset) {
      if (bytes[padding_offset] != 0) {
        throw std::runtime_error("fixture has non-zero alignment padding");
      }
    }
    if (index + 1 < chunk_count &&
        chunk_uncompressed_size % kPayloadAlignment != 0) {
      throw std::runtime_error("fixture has an unaligned non-final output");
    }
    if (chunk_uncompressed_size >
        std::numeric_limits<std::size_t>::max() - uncompressed_size) {
      throw std::runtime_error("fixture uncompressed size overflows");
    }
    uncompressed_size += chunk_uncompressed_size;
    previous_end = payload_offset + compressed_size;
    chunks.push_back(
        {payload_offset, compressed_size, chunk_uncompressed_size});
  }
  if (chunks.empty() || previous_end != record_size) {
    throw std::runtime_error("fixture does not end at record_size");
  }
  if (as_size(read_u64(bytes, 20), "uncompressed_size") != uncompressed_size) {
    throw std::runtime_error("fixture uncompressed total is inconsistent");
  }

  return {std::move(bytes), expected_framing, std::move(chunks),
          uncompressed_size};
}

AlignmentRequirements get_alignment_requirements(Framing framing) {
#if defined(LMCACHE_COMPAT_NVIDIA)
  nvcompAlignmentRequirements_t requirements{};
  if (framing == Framing::kRaw) {
    const auto options = nvcompBatchedDeflateDecompressDefaultOpts;
    check_vendor(nvcompBatchedDeflateDecompressGetRequiredAlignments(
                     options, &requirements),
                 "nvcompBatchedDeflateDecompressGetRequiredAlignments");
  } else {
    const auto options = nvcompBatchedGzipDecompressDefaultOpts;
    check_vendor(nvcompBatchedGzipDecompressGetRequiredAlignments(
                     options, &requirements),
                 "nvcompBatchedGzipDecompressGetRequiredAlignments");
  }
  return {requirements.input, requirements.output, requirements.temp};
#else
  const std::size_t alignment = framing == Framing::kRaw
                                    ? hipcompDeflateRequiredAlignment
                                    : hipcompGzipRequiredAlignment;
  return {alignment, alignment, alignment};
#endif
}

std::size_t get_temporary_size(Framing framing, std::size_t chunk_count,
                               std::size_t max_uncompressed_chunk_size,
                               std::size_t total_uncompressed_size) {
  std::size_t temporary_size = 0;
#if defined(LMCACHE_COMPAT_NVIDIA)
  if (framing == Framing::kRaw) {
    const auto options = nvcompBatchedDeflateDecompressDefaultOpts;
    check_vendor(nvcompBatchedDeflateDecompressGetTempSizeAsync(
                     chunk_count, max_uncompressed_chunk_size, options,
                     &temporary_size, total_uncompressed_size),
                 "nvcompBatchedDeflateDecompressGetTempSizeAsync");
  } else {
    const auto options = nvcompBatchedGzipDecompressDefaultOpts;
    check_vendor(nvcompBatchedGzipDecompressGetTempSizeAsync(
                     chunk_count, max_uncompressed_chunk_size, options,
                     &temporary_size, total_uncompressed_size),
                 "nvcompBatchedGzipDecompressGetTempSizeAsync");
  }
#else
  static_cast<void>(total_uncompressed_size);
  if (framing == Framing::kRaw) {
    check_vendor(hipcompBatchedDeflateDecompressGetTempSize(
                     chunk_count, max_uncompressed_chunk_size, &temporary_size),
                 "hipcompBatchedDeflateDecompressGetTempSize");
  } else {
    check_vendor(hipcompBatchedGzipDecompressGetTempSize(
                     chunk_count, max_uncompressed_chunk_size, &temporary_size),
                 "hipcompBatchedGzipDecompressGetTempSize");
  }
#endif
  return temporary_size;
}

void submit_decompression(Framing framing,
                          const void* const* device_compressed_pointers,
                          const std::size_t* device_compressed_sizes,
                          const std::size_t* device_output_capacities,
                          std::size_t* device_actual_output_sizes,
                          std::size_t chunk_count, void* device_temporary,
                          std::size_t temporary_size,
                          void* const* device_output_pointers,
                          VendorStatus* device_statuses,
                          GpuStreamHandle stream) {
#if defined(LMCACHE_COMPAT_NVIDIA)
  if (framing == Framing::kRaw) {
    const auto options = nvcompBatchedDeflateDecompressDefaultOpts;
    check_vendor(nvcompBatchedDeflateDecompressAsync(
                     device_compressed_pointers, device_compressed_sizes,
                     device_output_capacities, device_actual_output_sizes,
                     chunk_count, device_temporary, temporary_size,
                     device_output_pointers, options, device_statuses, stream),
                 "nvcompBatchedDeflateDecompressAsync");
  } else {
    const auto options = nvcompBatchedGzipDecompressDefaultOpts;
    check_vendor(nvcompBatchedGzipDecompressAsync(
                     device_compressed_pointers, device_compressed_sizes,
                     device_output_capacities, device_actual_output_sizes,
                     chunk_count, device_temporary, temporary_size,
                     device_output_pointers, options, device_statuses, stream),
                 "nvcompBatchedGzipDecompressAsync");
  }
#else
  if (framing == Framing::kRaw) {
    check_vendor(hipcompBatchedDeflateDecompressAsync(
                     device_compressed_pointers, device_compressed_sizes,
                     device_output_capacities, device_actual_output_sizes,
                     chunk_count, device_temporary, temporary_size,
                     device_output_pointers, device_statuses, stream),
                 "hipcompBatchedDeflateDecompressAsync");
  } else {
    check_vendor(hipcompBatchedGzipDecompressAsync(
                     device_compressed_pointers, device_compressed_sizes,
                     device_output_capacities, device_actual_output_sizes,
                     chunk_count, device_temporary, temporary_size,
                     device_output_pointers, device_statuses, stream),
                 "hipcompBatchedGzipDecompressAsync");
  }
#endif
}

void require_alignment(const void* pointer, std::size_t alignment,
                       const char* buffer_name) {
  if (alignment == 0 ||
      reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0) {
    throw std::runtime_error(std::string(buffer_name) +
                             " does not satisfy vendor alignment");
  }
}

void run_probe(const ParsedRecord& record) {
  const std::size_t chunk_count = record.chunks.size();
  const AlignmentRequirements alignments =
      get_alignment_requirements(record.framing);

  DeviceAllocation device_record(record.bytes.size());
  DeviceAllocation device_output(record.uncompressed_size);
  copy_to_device(device_record.get(), record.bytes.data(), record.bytes.size());

  auto* device_record_bytes = device_record.as<std::uint8_t>();
  auto* device_output_bytes = device_output.as<std::uint8_t>();
  std::vector<const void*> compressed_pointers;
  std::vector<void*> output_pointers;
  std::vector<std::size_t> compressed_sizes;
  std::vector<std::size_t> output_capacities;
  compressed_pointers.reserve(chunk_count);
  output_pointers.reserve(chunk_count);
  compressed_sizes.reserve(chunk_count);
  output_capacities.reserve(chunk_count);

  std::size_t output_offset = 0;
  std::size_t max_uncompressed_chunk_size = 0;
  for (const ChunkDescriptor& chunk : record.chunks) {
    const void* compressed_pointer = device_record_bytes + chunk.payload_offset;
    void* output_pointer = device_output_bytes + output_offset;
    require_alignment(compressed_pointer, alignments.input,
                      "compressed payload");
    require_alignment(output_pointer, alignments.output, "output payload");
    compressed_pointers.push_back(compressed_pointer);
    output_pointers.push_back(output_pointer);
    compressed_sizes.push_back(chunk.compressed_size);
    output_capacities.push_back(chunk.uncompressed_size);
    output_offset += chunk.uncompressed_size;
    max_uncompressed_chunk_size =
        std::max(max_uncompressed_chunk_size, chunk.uncompressed_size);
  }

  DeviceAllocation device_compressed_pointers(chunk_count * sizeof(void*));
  DeviceAllocation device_output_pointers(chunk_count * sizeof(void*));
  DeviceAllocation device_compressed_sizes(chunk_count * sizeof(std::size_t));
  DeviceAllocation device_output_capacities(chunk_count * sizeof(std::size_t));
  DeviceAllocation device_actual_output_sizes(chunk_count *
                                              sizeof(std::size_t));
  DeviceAllocation device_statuses(chunk_count * sizeof(VendorStatus));
  copy_to_device(device_compressed_pointers.get(), compressed_pointers.data(),
                 chunk_count * sizeof(void*));
  copy_to_device(device_output_pointers.get(), output_pointers.data(),
                 chunk_count * sizeof(void*));
  copy_to_device(device_compressed_sizes.get(), compressed_sizes.data(),
                 chunk_count * sizeof(std::size_t));
  copy_to_device(device_output_capacities.get(), output_capacities.data(),
                 chunk_count * sizeof(std::size_t));

  const std::size_t temporary_size =
      get_temporary_size(record.framing, chunk_count,
                         max_uncompressed_chunk_size, record.uncompressed_size);
  DeviceAllocation device_temporary(temporary_size);
  if (temporary_size != 0) {
    require_alignment(device_temporary.get(), alignments.temporary,
                      "temporary buffer");
  }

  GpuStream stream;
  submit_decompression(record.framing,
                       device_compressed_pointers.as<const void*>(),
                       device_compressed_sizes.as<std::size_t>(),
                       device_output_capacities.as<std::size_t>(),
                       device_actual_output_sizes.as<std::size_t>(),
                       chunk_count, device_temporary.get(), temporary_size,
                       device_output_pointers.as<void*>(),
                       device_statuses.as<VendorStatus>(), stream.get());
  stream.synchronize();

  std::vector<VendorStatus> statuses(chunk_count);
  std::vector<std::size_t> actual_output_sizes(chunk_count);
  std::vector<std::uint8_t> output(record.uncompressed_size);
  copy_from_device(statuses.data(), device_statuses.get(),
                   chunk_count * sizeof(VendorStatus));
  copy_from_device(actual_output_sizes.data(), device_actual_output_sizes.get(),
                   chunk_count * sizeof(std::size_t));
  copy_from_device(output.data(), device_output.get(), output.size());

  for (std::size_t index = 0; index < chunk_count; ++index) {
    check_vendor(statuses[index], "per-chunk decompression status");
    if (actual_output_sizes[index] != record.chunks[index].uncompressed_size) {
      throw std::runtime_error("vendor returned the wrong output size");
    }
  }
  const std::vector<std::uint8_t> expected = {'h', 'e', 'l', 'l', 'o'};
  if (output != expected) {
    throw std::runtime_error("vendor output does not match frozen record");
  }

  const char* framing_name =
      record.framing == Framing::kRaw ? "raw Deflate" : "Gzip";
  std::cout << kVendorName << " accepted LMCache V1 " << framing_name
            << " record" << std::endl;
}

}  // namespace

int main(int argument_count, char** arguments) {
  if (argument_count != 3) {
    std::cerr << "usage: " << arguments[0]
              << " RAW_DEFLATE_FIXTURE GZIP_FIXTURE" << std::endl;
    return 2;
  }

  try {
    run_probe(parse_record(arguments[1], Framing::kRaw));
    run_probe(parse_record(arguments[2], Framing::kGzip));
  } catch (const std::exception& error) {
    std::cerr << "compatibility probe failed: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
