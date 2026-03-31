// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — wire_protocol.cpp
//
// msgpack encode/decode matching Python msgspec.msgpack wire format.
// The Python client uses msgspec Structs (encoded as msgpack arrays of
// fields in declaration order) and msgspec Ext types for CudaIPCWrapper.

#include "wire_protocol.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <msgpack.hpp>

namespace lmcache {
namespace server {

// ============================================================================
// Encoder
// ============================================================================

struct Encoder::Impl {
  // We allocate a fresh sbuffer per call and return the bytes.
  // This is cheap and avoids lifetime issues.
};

Encoder::Encoder() : impl_(new Impl()) {}
Encoder::~Encoder() { delete impl_; }

std::vector<uint8_t> Encoder::encode_request_uid(RequestUID uid) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, uid);
  return {reinterpret_cast<const uint8_t*>(buf.data()),
          reinterpret_cast<const uint8_t*>(buf.data()) + buf.size()};
}

std::vector<uint8_t> Encoder::encode_request_type(RequestType type) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, static_cast<int>(type));
  return {reinterpret_cast<const uint8_t*>(buf.data()),
          reinterpret_cast<const uint8_t*>(buf.data()) + buf.size()};
}

std::vector<uint8_t> Encoder::encode_store_response(
    const std::vector<uint8_t>& event_ipc_handle, bool success) {
  // Python: tuple[bytes, bool] -> msgpack array of 2 elements
  msgpack::sbuffer buf;
  msgpack::packer<msgpack::sbuffer> pk(buf);
  pk.pack_array(2);
  // bytes -> msgpack bin
  pk.pack_bin(event_ipc_handle.size());
  pk.pack_bin_body(reinterpret_cast<const char*>(event_ipc_handle.data()),
                   event_ipc_handle.size());
  pk.pack(success);
  return {reinterpret_cast<const uint8_t*>(buf.data()),
          reinterpret_cast<const uint8_t*>(buf.data()) + buf.size()};
}

std::vector<uint8_t> Encoder::encode_retrieve_response(
    const std::vector<uint8_t>& event_ipc_handle, bool success) {
  // Same format as store response: tuple[bytes, bool]
  return encode_store_response(event_ipc_handle, success);
}

std::vector<uint8_t> Encoder::encode_int_response(int64_t value) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, value);
  return {reinterpret_cast<const uint8_t*>(buf.data()),
          reinterpret_cast<const uint8_t*>(buf.data()) + buf.size()};
}

std::vector<uint8_t> Encoder::encode_bool_response(bool value) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, value);
  return {reinterpret_cast<const uint8_t*>(buf.data()),
          reinterpret_cast<const uint8_t*>(buf.data()) + buf.size()};
}

std::vector<uint8_t> Encoder::encode_none_response() {
  msgpack::sbuffer buf;
  msgpack::packer<msgpack::sbuffer> pk(buf);
  pk.pack_nil();
  return {reinterpret_cast<const uint8_t*>(buf.data()),
          reinterpret_cast<const uint8_t*>(buf.data()) + buf.size()};
}

std::vector<uint8_t> Encoder::encode_optional_int_response(int64_t value,
                                                           bool is_none) {
  if (is_none) {
    return encode_none_response();
  }
  return encode_int_response(value);
}

std::vector<uint8_t> Encoder::encode_string_response(const std::string& value) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, value);
  return {reinterpret_cast<const uint8_t*>(buf.data()),
          reinterpret_cast<const uint8_t*>(buf.data()) + buf.size()};
}

// ============================================================================
// Decoder helpers
// ============================================================================

namespace {

/// Unpack a single msgpack object from raw bytes.
msgpack::object_handle unpack_one(const uint8_t* data, size_t len) {
  return msgpack::unpack(reinterpret_cast<const char*>(data), len);
}

/// Decode an IPCCacheEngineKey from a msgpack-encoded frame.
///
/// Python msgspec encodes a dataclass as a msgpack MAP (dict with string keys):
///   {"model_name": str, "world_size": int, "worker_id": int|None,
///    "token_ids": [int...], "start": int, "end": int, "request_id": str}
///
/// worker_id may be None (msgpack nil) for lookup requests.

namespace {
/// Helper: look up a key in a msgpack map, return pointer or nullptr.
const msgpack::object* map_find(const msgpack::object_map& m, const char* key) {
  for (uint32_t i = 0; i < m.size; ++i) {
    if (m.ptr[i].key.type == msgpack::type::STR) {
      const auto& s = m.ptr[i].key.via.str;
      if (s.size == std::strlen(key) && std::memcmp(s.ptr, key, s.size) == 0) {
        return &m.ptr[i].val;
      }
    }
  }
  return nullptr;
}
}  // namespace

IPCCacheEngineKey decode_ipc_key(const uint8_t* data, size_t len) {
  auto oh = unpack_one(data, len);
  const auto& obj = oh.get();

  IPCCacheEngineKey key;

  if (obj.type == msgpack::type::MAP) {
    // Map (dict) encoding — the actual format used by msgspec for dataclasses
    const auto& m = obj.via.map;

    if (auto* v = map_find(m, "model_name"))
      key.model_name = v->as<std::string>();
    if (auto* v = map_find(m, "world_size")) key.world_size = v->as<int32_t>();

    if (auto* v = map_find(m, "worker_id")) {
      if (v->type == msgpack::type::NIL) {
        key.worker_id = -1;
      } else {
        key.worker_id = v->as<int32_t>();
      }
    } else {
      key.worker_id = -1;
    }

    if (auto* v = map_find(m, "token_ids")) {
      const auto& tok_arr = v->via.array;
      key.token_ids.resize(tok_arr.size);
      for (uint32_t i = 0; i < tok_arr.size; ++i) {
        key.token_ids[i] = tok_arr.ptr[i].as<int32_t>();
      }
    }

    if (auto* v = map_find(m, "start")) key.start = v->as<int32_t>();
    if (auto* v = map_find(m, "end")) key.end = v->as<int32_t>();
    if (auto* v = map_find(m, "request_id"))
      key.request_id = v->as<std::string>();

  } else if (obj.type == msgpack::type::ARRAY) {
    // Array encoding (legacy/alternate Struct encoding)
    const auto& arr = obj.via.array;
    if (arr.size < 7) {
      throw std::runtime_error("IPCCacheEngineKey: expected 7 fields, got " +
                               std::to_string(arr.size));
    }

    key.model_name = arr.ptr[0].as<std::string>();
    key.world_size = arr.ptr[1].as<int32_t>();

    if (arr.ptr[2].type == msgpack::type::NIL) {
      key.worker_id = -1;
    } else {
      key.worker_id = arr.ptr[2].as<int32_t>();
    }

    {
      const auto& tok_arr = arr.ptr[3].via.array;
      key.token_ids.resize(tok_arr.size);
      for (uint32_t i = 0; i < tok_arr.size; ++i) {
        key.token_ids[i] = tok_arr.ptr[i].as<int32_t>();
      }
    }

    key.start = arr.ptr[4].as<int32_t>();
    key.end = arr.ptr[5].as<int32_t>();
    key.request_id = arr.ptr[6].as<std::string>();

  } else {
    throw std::runtime_error(
        "IPCCacheEngineKey: expected msgpack map or array, got type " +
        std::to_string(static_cast<int>(obj.type)));
  }

  return key;
}

/// Decode a list[int] from msgpack bytes (e.g., gpu_block_ids).
std::vector<int32_t> decode_int_list(const uint8_t* data, size_t len) {
  auto oh = unpack_one(data, len);
  const auto& obj = oh.get();
  if (obj.type != msgpack::type::ARRAY) {
    throw std::runtime_error("decode_int_list: expected array");
  }
  const auto& arr = obj.via.array;
  std::vector<int32_t> result(arr.size);
  for (uint32_t i = 0; i < arr.size; ++i) {
    result[i] = arr.ptr[i].as<int32_t>();
  }
  return result;
}

/// Decode raw bytes from msgpack bin format.
std::vector<uint8_t> decode_bytes(const uint8_t* data, size_t len) {
  auto oh = unpack_one(data, len);
  const auto& obj = oh.get();
  if (obj.type != msgpack::type::BIN) {
    throw std::runtime_error("decode_bytes: expected bin, got type " +
                             std::to_string(static_cast<int>(obj.type)));
  }
  const char* bin_data = obj.via.bin.ptr;
  uint32_t bin_size = obj.via.bin.size;
  return {reinterpret_cast<const uint8_t*>(bin_data),
          reinterpret_cast<const uint8_t*>(bin_data) + bin_size};
}

/// Map a Python torch.dtype name to our DType enum.
DType dtype_from_string(const std::string& s) {
  if (s == "float16") return DType::Float16;
  if (s == "bfloat16") return DType::BFloat16;
  if (s == "float32") return DType::Float32;
  if (s == "float8_e4m3fn") return DType::Float8E4M3FN;
  if (s == "float8_e5m2") return DType::Float8E5M2;
  if (s == "int8") return DType::Int8;
  if (s == "uint8") return DType::Int8;  // fp8 KV cache uses torch.uint8
  if (s == "int32") return DType::Int32;
  if (s == "int64") return DType::Int64;
  std::fprintf(stderr,
               "WARNING: unknown dtype string: '%s', defaulting to float16\n",
               s.c_str());
  return DType::Float16;
}

/// Structured pickle parser for CudaIPCWrapper objects.
///
/// Pickle protocol 4 layout (from pickletools.dis):
///
///   PROTO 4
///   FRAME
///   SHORT_BINUNICODE 'lmcache.v1.multiprocess.custom_types'
///   SHORT_BINUNICODE 'CudaIPCWrapper'
///   STACK_GLOBAL + NEWOBJ (creates empty instance)
///   EMPTY_DICT
///   MARK
///     SHORT_BINUNICODE 'handle'
///     MARK
///       BININT1 <device_idx>
///       SHORT_BINBYTES <66 bytes>         -- handle[1]: storage handle
///       BININT1 <storage_size>
///       BININT1 <storage_offset_in_handle>
///       SHORT_BINBYTES <~27 bytes>        -- handle[4]: storage filename
///       BININT1 0
///       SHORT_BINBYTES <64 bytes>         -- handle[6]: cudaIpcMemHandle_t <<<
///       NEWTRUE
///     TUPLE
///     SHORT_BINUNICODE 'dtype'
///     SHORT_BINUNICODE 'torch'
///     SHORT_BINUNICODE '<dtype_name>'     -- e.g. 'float16', 'bfloat16'
///     STACK_GLOBAL
///     SHORT_BINUNICODE 'shape'
///     <tuple of ints>
///     SHORT_BINUNICODE 'stride'
///     <tuple of ints>
///     SHORT_BINUNICODE 'storage_offset'
///     <int>
///     SHORT_BINUNICODE 'device_uuid'
///     SHORT_BINUNICODE '<uuid string>'    -- bare UUID, no "GPU-" prefix
///   SETITEMS
///   BUILD
///   STOP
///
/// We parse opcodes sequentially, tracking field names from SHORT_BINUNICODE
/// that precede values, and extracting the fields we need.

// Pickle protocol 4 opcodes we care about
constexpr uint8_t PK_PROTO = 0x80;
constexpr uint8_t PK_FRAME = 0x95;
constexpr uint8_t PK_SHORT_BINUNICODE = 0x8c;
constexpr uint8_t PK_BINUNICODE = 0x8d;
constexpr uint8_t PK_SHORT_BINBYTES = 0x43;  // 'C'
constexpr uint8_t PK_BINBYTES = 0x42;        // 'B'
constexpr uint8_t PK_BININT1 = 0x4b;         // 'K'
constexpr uint8_t PK_BININT2 = 0x4d;         // 'M'
constexpr uint8_t PK_BININT = 0x4a;          // 'J'
constexpr uint8_t PK_LONG1 = 0x8a;
constexpr uint8_t PK_MEMOIZE = 0x94;
constexpr uint8_t PK_MARK = 0x28;   // '('
constexpr uint8_t PK_TUPLE = 0x74;  // 't'
constexpr uint8_t PK_TUPLE1 = 0x85;
constexpr uint8_t PK_TUPLE2 = 0x86;
constexpr uint8_t PK_TUPLE3 = 0x87;
constexpr uint8_t PK_EMPTY_TUPLE = 0x29;  // ')'
constexpr uint8_t PK_EMPTY_DICT = 0x7d;   // '}'
constexpr uint8_t PK_STACK_GLOBAL = 0x93;
constexpr uint8_t PK_NEWOBJ = 0x81;
constexpr uint8_t PK_BUILD = 0x62;     // 'b'
constexpr uint8_t PK_SETITEMS = 0x75;  // 'u'
constexpr uint8_t PK_NEWTRUE = 0x88;
constexpr uint8_t PK_NEWFALSE = 0x89;
constexpr uint8_t PK_NONE = 0x4e;  // 'N'
constexpr uint8_t PK_STOP = 0x2e;  // '.'

CudaIpcTensorDesc decode_cuda_ipc_from_pickle(const uint8_t* data, size_t len) {
  CudaIpcTensorDesc desc{};
  std::memset(desc.ipc_handle, 0, 64);
  desc.dtype = DType::Float16;
  desc.storage_offset = 0;
  desc.storage_size_bytes = 0;

  // State for tracking dict key-value pairs
  std::string last_field_name;
  std::string last_string;         // most recent SHORT_BINUNICODE value
  std::string second_last_string;  // for STACK_GLOBAL (module, name)

  // Collect bytes objects found in the handle tuple
  std::vector<std::pair<const uint8_t*, size_t>> bytes_in_handle;
  bool in_handle_tuple = false;  // true between handle's MARK and TUPLE
  int mark_depth = 0;

  // We always collect ints into pending_ints.
  // When a tuple-building opcode fires and last_field_name is
  // "shape" or "stride", we capture the ints.
  // For MARK...TUPLE style, we record where the mark was and
  // take ints from there.  For TUPLE2/TUPLE3 (no mark), we
  // take the last 2 or 3 ints from pending_ints.
  std::vector<int64_t> pending_ints;
  int mark_int_pos = -1;  // index into pending_ints at last MARK

  size_t pos = 0;
  while (pos < len) {
    uint8_t op = data[pos++];

    switch (op) {
      case PK_PROTO:
        pos++;  // skip version byte
        break;

      case PK_FRAME:
        pos += 8;  // skip 8-byte frame length
        break;

      case PK_SHORT_BINUNICODE: {
        if (pos >= len) break;
        uint8_t slen = data[pos++];
        if (pos + slen > len) break;
        second_last_string = last_string;
        last_string =
            std::string(reinterpret_cast<const char*>(data + pos), slen);
        pos += slen;

        // Check if this is a field name we care about
        if (last_string == "handle" || last_string == "dtype" ||
            last_string == "shape" || last_string == "stride" ||
            last_string == "storage_offset" || last_string == "device_uuid") {
          last_field_name = last_string;
        }
        // If the field we're filling is device_uuid and this is the value
        else if (last_field_name == "device_uuid") {
          desc.device_uuid = last_string;
          last_field_name.clear();
        }
        break;
      }

      case PK_BINUNICODE: {
        if (pos + 4 > len) break;
        uint32_t slen = static_cast<uint32_t>(data[pos]) |
                        (static_cast<uint32_t>(data[pos + 1]) << 8) |
                        (static_cast<uint32_t>(data[pos + 2]) << 16) |
                        (static_cast<uint32_t>(data[pos + 3]) << 24);
        pos += 4;
        if (pos + slen > len) break;
        second_last_string = last_string;
        last_string =
            std::string(reinterpret_cast<const char*>(data + pos), slen);
        pos += slen;
        break;
      }

      case PK_SHORT_BINBYTES: {
        if (pos >= len) break;
        uint8_t blen = data[pos++];
        if (pos + blen > len) break;
        if (in_handle_tuple) {
          bytes_in_handle.push_back({data + pos, blen});
        }
        pos += blen;
        break;
      }

      case PK_BINBYTES: {
        if (pos + 4 > len) break;
        uint32_t blen = static_cast<uint32_t>(data[pos]) |
                        (static_cast<uint32_t>(data[pos + 1]) << 8) |
                        (static_cast<uint32_t>(data[pos + 2]) << 16) |
                        (static_cast<uint32_t>(data[pos + 3]) << 24);
        pos += 4;
        if (pos + blen > len) break;
        if (in_handle_tuple) {
          bytes_in_handle.push_back({data + pos, blen});
        }
        pos += blen;
        break;
      }

      case PK_BININT1: {
        if (pos >= len) break;
        int64_t val = data[pos++];
        if (last_field_name == "storage_offset") {
          desc.storage_offset = val;
          last_field_name.clear();
        }
        pending_ints.push_back(val);
        break;
      }

      case PK_BININT2: {
        if (pos + 2 > len) break;
        int64_t val = static_cast<uint16_t>(data[pos]) |
                      (static_cast<uint16_t>(data[pos + 1]) << 8);
        pos += 2;
        if (last_field_name == "storage_offset") {
          desc.storage_offset = val;
          last_field_name.clear();
        }
        pending_ints.push_back(val);
        break;
      }

      case PK_BININT: {
        if (pos + 4 > len) break;
        int32_t val;
        std::memcpy(&val, data + pos, 4);
        pos += 4;
        if (last_field_name == "storage_offset") {
          desc.storage_offset = val;
          last_field_name.clear();
        }
        pending_ints.push_back(val);
        break;
      }

      case PK_LONG1: {
        if (pos >= len) break;
        uint8_t nbytes = data[pos++];
        if (pos + nbytes > len) break;
        int64_t val = 0;
        for (int i = nbytes - 1; i >= 0; --i) {
          val = (val << 8) | data[pos + i];
        }
        if (nbytes > 0 && (data[pos + nbytes - 1] & 0x80)) {
          for (size_t i = nbytes; i < 8; ++i) val |= (0xFFLL << (i * 8));
        }
        pos += nbytes;
        if (last_field_name == "storage_offset") {
          desc.storage_offset = val;
          last_field_name.clear();
        }
        pending_ints.push_back(val);
        break;
      }

      case PK_MARK:
        mark_depth++;
        if (last_field_name == "handle") {
          in_handle_tuple = true;
        }
        mark_int_pos = static_cast<int>(pending_ints.size());
        break;

      case PK_TUPLE: {
        mark_depth--;
        if (in_handle_tuple && last_field_name == "handle") {
          in_handle_tuple = false;
          last_field_name.clear();
          for (const auto& [ptr, sz] : bytes_in_handle) {
            if (sz >= 64) {
              // Store the full blob for libtorch's getIpcDevPtr
              desc.ipc_handle_blob.assign(ptr, ptr + sz);
              // Also copy first 64 bytes into legacy field
              std::memcpy(desc.ipc_handle, ptr, 64);
              break;
            }
          }
          // Extract storage_size from handle ints.
          // handle tuple layout: (device_idx, ipc_bytes, storage_size, ...)
          // The ints collected during handle parsing are:
          //   [device_idx, storage_size, storage_offset_in_handle, 0, ...]
          // pending_ints[0] = device_idx, pending_ints[1] = storage_size
          if (pending_ints.size() >= 2) {
            desc.storage_size_bytes = pending_ints[1];
          }
        }
        // MARK...TUPLE: ints from mark_int_pos onward
        if (last_field_name == "shape" && mark_int_pos >= 0) {
          desc.shape.assign(pending_ints.begin() + mark_int_pos,
                            pending_ints.end());
          last_field_name.clear();
        }
        if (last_field_name == "stride" && mark_int_pos >= 0) {
          desc.stride.assign(pending_ints.begin() + mark_int_pos,
                             pending_ints.end());
          last_field_name.clear();
        }
        pending_ints.clear();
        mark_int_pos = -1;
        break;
      }

      case PK_TUPLE1: {
        // Pop 1 item → 1-element tuple (no MARK)
        if (last_field_name == "shape" && pending_ints.size() >= 1) {
          desc.shape = {pending_ints.back()};
          last_field_name.clear();
        } else if (last_field_name == "stride" && pending_ints.size() >= 1) {
          desc.stride = {pending_ints.back()};
          last_field_name.clear();
        }
        break;
      }

      case PK_TUPLE2: {
        // Pop 2 items → 2-element tuple (no MARK)
        size_t n = pending_ints.size();
        if (last_field_name == "shape" && n >= 2) {
          desc.shape = {pending_ints[n - 2], pending_ints[n - 1]};
          last_field_name.clear();
        } else if (last_field_name == "stride" && n >= 2) {
          desc.stride = {pending_ints[n - 2], pending_ints[n - 1]};
          last_field_name.clear();
        }
        break;
      }

      case PK_TUPLE3: {
        // Pop 3 items → 3-element tuple (no MARK)
        size_t n = pending_ints.size();
        if (last_field_name == "shape" && n >= 3) {
          desc.shape = {pending_ints[n - 3], pending_ints[n - 2],
                        pending_ints[n - 1]};
          last_field_name.clear();
        } else if (last_field_name == "stride" && n >= 3) {
          desc.stride = {pending_ints[n - 3], pending_ints[n - 2],
                         pending_ints[n - 1]};
          last_field_name.clear();
        }
        break;
      }

      case PK_STACK_GLOBAL:
        // The two strings before STACK_GLOBAL form module.name.
        // For dtype: second_last_string="torch", last_string="float16"
        if (last_field_name == "dtype" && second_last_string == "torch") {
          desc.dtype = dtype_from_string(last_string);
          last_field_name.clear();
        }
        break;

      case PK_MEMOIZE:
      case PK_EMPTY_TUPLE:
      case PK_EMPTY_DICT:
      case PK_NEWOBJ:
      case PK_BUILD:
      case PK_SETITEMS:
      case PK_NEWTRUE:
      case PK_NEWFALSE:
      case PK_NONE:
        // No-data opcodes
        break;

      case PK_STOP:
        goto done;

      default:
        // Unknown opcode — skip and hope for the best.
        // This is fragile but works for the known CudaIPCWrapper layout.
        break;
    }
  }

done:
  // Debug logging: report everything we extracted
  {
    // IPC handle: show first 16 bytes hex
    char ipc_hex[129];
    for (int i = 0; i < 64; ++i)
      std::snprintf(ipc_hex + i * 2, 3, "%02x", desc.ipc_handle[i]);
    ipc_hex[128] = '\0';

    // Check if ipc_handle is all zeros
    bool all_zero = true;
    for (int i = 0; i < 64; ++i) {
      if (desc.ipc_handle[i] != 0) {
        all_zero = false;
        break;
      }
    }

    std::fprintf(stderr, "[pickle] device_uuid='%s' dtype=%d shape=[",
                 desc.device_uuid.c_str(), static_cast<int>(desc.dtype));
    for (size_t i = 0; i < desc.shape.size(); ++i) {
      if (i > 0) std::fprintf(stderr, ",");
      std::fprintf(stderr, "%ld", desc.shape[i]);
    }
    std::fprintf(stderr, "] stride=[");
    for (size_t i = 0; i < desc.stride.size(); ++i) {
      if (i > 0) std::fprintf(stderr, ",");
      std::fprintf(stderr, "%ld", desc.stride[i]);
    }
    std::fprintf(stderr,
                 "] storage_offset=%ld ipc_handle=%s%s "
                 "bytes_in_handle=%zu: ",
                 desc.storage_offset, ipc_hex, all_zero ? " (ALL ZEROS!)" : "",
                 bytes_in_handle.size());
    // Print sizes and first 4 bytes of each bytes_in_handle entry
    for (size_t i = 0; i < bytes_in_handle.size(); ++i) {
      auto& [ptr, sz] = bytes_in_handle[i];
      std::fprintf(stderr, "[%zu]=%zu(", i, sz);
      for (size_t j = 0; j < std::min(sz, (size_t)4); ++j)
        std::fprintf(stderr, "%02x", ptr[j]);
      std::fprintf(stderr, "...) ");
    }
    std::fprintf(stderr, "\n");
  }

  if (desc.device_uuid.empty()) {
    std::fprintf(stderr, "WARNING: pickle parser: device_uuid is empty\n");
  }

  // Build "GPU-" prefixed UUID for matching with CUDA device properties
  if (!desc.device_uuid.empty() && desc.device_uuid.substr(0, 4) != "GPU-") {
    desc.device_uuid = "GPU-" + desc.device_uuid;
  }

  return desc;
}

}  // anonymous namespace

// ============================================================================
// Decoder
// ============================================================================

struct Decoder::Impl {
  // Stateless for now; kept for future caching/pooling.
};

Decoder::Decoder() : impl_(new Impl()) {}
Decoder::~Decoder() { delete impl_; }

RequestUID Decoder::decode_request_uid(const uint8_t* data, size_t len) {
  auto oh = unpack_one(data, len);
  return oh.get().as<int64_t>();
}

RequestType Decoder::decode_request_type(const uint8_t* data, size_t len) {
  auto oh = unpack_one(data, len);
  int val = oh.get().as<int>();
  return static_cast<RequestType>(val);
}

RegisterPayload Decoder::decode_register_payload(
    const std::vector<std::vector<uint8_t>>& frames) {
  // Protocol: [instance_id, kv_cache_list, model_name, world_size]
  // kv_cache_list is list[CudaIPCWrapper] encoded via custom encoder
  // (each element is Ext type 1 with pickle bytes)
  if (frames.size() < 4) {
    throw std::runtime_error("RegisterPayload: expected 4 frames, got " +
                             std::to_string(frames.size()));
  }

  RegisterPayload payload;

  // Frame 0: instance_id (int)
  {
    auto oh = unpack_one(frames[0].data(), frames[0].size());
    payload.instance_id = oh.get().as<int32_t>();
  }

  // Frame 1: list[CudaIPCWrapper] — msgpack array of Ext(1, pickle_bytes)
  {
    auto oh = unpack_one(frames[1].data(), frames[1].size());
    const auto& obj = oh.get();
    if (obj.type != msgpack::type::ARRAY) {
      throw std::runtime_error("RegisterPayload: kv_caches expected array");
    }
    const auto& arr = obj.via.array;
    payload.kv_caches.resize(arr.size);
    for (uint32_t i = 0; i < arr.size; ++i) {
      const auto& elem = arr.ptr[i];
      if (elem.type != msgpack::type::EXT) {
        throw std::runtime_error(
            "RegisterPayload: kv_cache element expected Ext type");
      }
      const auto& ext = elem.via.ext;
      if (ext.type() != 1) {
        throw std::runtime_error("RegisterPayload: expected Ext code 1, got " +
                                 std::to_string(ext.type()));
      }
      payload.kv_caches[i] = decode_cuda_ipc_from_pickle(
          reinterpret_cast<const uint8_t*>(ext.data()), ext.size);
    }
  }

  // Frame 2: model_name (str)
  {
    auto oh = unpack_one(frames[2].data(), frames[2].size());
    payload.model_name = oh.get().as<std::string>();
  }

  // Frame 3: world_size (int)
  {
    auto oh = unpack_one(frames[3].data(), frames[3].size());
    payload.world_size = oh.get().as<int32_t>();
  }

  return payload;
}

StorePayload Decoder::decode_store_payload(
    const std::vector<std::vector<uint8_t>>& frames) {
  // Protocol: [key, instance_id, gpu_block_ids, event_ipc_handle]
  if (frames.size() < 4) {
    throw std::runtime_error("StorePayload: expected 4 frames, got " +
                             std::to_string(frames.size()));
  }

  StorePayload payload;
  payload.key = decode_ipc_key(frames[0].data(), frames[0].size());

  {
    auto oh = unpack_one(frames[1].data(), frames[1].size());
    payload.instance_id = oh.get().as<int32_t>();
  }

  payload.gpu_block_ids = decode_int_list(frames[2].data(), frames[2].size());

  payload.event_ipc_handle = decode_bytes(frames[3].data(), frames[3].size());

  return payload;
}

RetrievePayload Decoder::decode_retrieve_payload(
    const std::vector<std::vector<uint8_t>>& frames) {
  // Protocol: [key, instance_id, gpu_block_ids, event_ipc_handle,
  //            skip_first_n_tokens]
  if (frames.size() < 5) {
    throw std::runtime_error("RetrievePayload: expected 5 frames, got " +
                             std::to_string(frames.size()));
  }

  RetrievePayload payload;
  payload.key = decode_ipc_key(frames[0].data(), frames[0].size());

  {
    auto oh = unpack_one(frames[1].data(), frames[1].size());
    payload.instance_id = oh.get().as<int32_t>();
  }

  payload.gpu_block_ids = decode_int_list(frames[2].data(), frames[2].size());

  payload.event_ipc_handle = decode_bytes(frames[3].data(), frames[3].size());

  {
    auto oh = unpack_one(frames[4].data(), frames[4].size());
    payload.skip_first_n_tokens = oh.get().as<int32_t>();
  }

  return payload;
}

LookupPayload Decoder::decode_lookup_payload(
    const std::vector<std::vector<uint8_t>>& frames) {
  // Protocol: [key, tp_size]
  if (frames.size() < 2) {
    throw std::runtime_error("LookupPayload: expected 2 frames, got " +
                             std::to_string(frames.size()));
  }

  LookupPayload payload;
  payload.key = decode_ipc_key(frames[0].data(), frames[0].size());

  {
    auto oh = unpack_one(frames[1].data(), frames[1].size());
    payload.tp_size = oh.get().as<int32_t>();
  }

  return payload;
}

FreeLookupLocksPayload Decoder::decode_free_lookup_locks_payload(
    const std::vector<std::vector<uint8_t>>& frames) {
  // Protocol: [key, tp_size]
  if (frames.size() < 2) {
    throw std::runtime_error("FreeLookupLocksPayload: expected 2 frames, got " +
                             std::to_string(frames.size()));
  }

  FreeLookupLocksPayload payload;
  payload.key = decode_ipc_key(frames[0].data(), frames[0].size());

  {
    auto oh = unpack_one(frames[1].data(), frames[1].size());
    payload.tp_size = oh.get().as<int32_t>();
  }

  return payload;
}

EndSessionPayload Decoder::decode_end_session_payload(
    const std::vector<std::vector<uint8_t>>& frames) {
  // Protocol: [request_id]
  if (frames.size() < 1) {
    throw std::runtime_error("EndSessionPayload: expected 1 frame, got " +
                             std::to_string(frames.size()));
  }

  EndSessionPayload payload;
  {
    auto oh = unpack_one(frames[0].data(), frames[0].size());
    payload.request_id = oh.get().as<std::string>();
  }

  return payload;
}

QueryPrefetchStatusPayload Decoder::decode_query_prefetch_status_payload(
    const std::vector<std::vector<uint8_t>>& frames) {
  // Protocol: [prefetch_job_id]
  if (frames.size() < 1) {
    throw std::runtime_error(
        "QueryPrefetchStatusPayload: expected 1 frame, got " +
        std::to_string(frames.size()));
  }

  QueryPrefetchStatusPayload payload;
  {
    auto oh = unpack_one(frames[0].data(), frames[0].size());
    payload.prefetch_job_id = oh.get().as<int32_t>();
  }

  return payload;
}

QueryPrefetchLookupHitsPayload
Decoder::decode_query_prefetch_lookup_hits_payload(
    const std::vector<std::vector<uint8_t>>& frames) {
  // Protocol: [prefetch_job_id]
  if (frames.size() < 1) {
    throw std::runtime_error(
        "QueryPrefetchLookupHitsPayload: expected 1 frame, got " +
        std::to_string(frames.size()));
  }

  QueryPrefetchLookupHitsPayload payload;
  {
    auto oh = unpack_one(frames[0].data(), frames[0].size());
    payload.prefetch_job_id = oh.get().as<int32_t>();
  }

  return payload;
}

int32_t Decoder::decode_int_payload(const uint8_t* data, size_t len) {
  auto oh = unpack_one(data, len);
  return oh.get().as<int32_t>();
}

CudaIpcTensorDesc Decoder::decode_cuda_ipc_wrapper(const uint8_t* data,
                                                   size_t len) {
  // First, try to unpack as msgpack to see if it's an Ext type
  auto oh = unpack_one(data, len);
  const auto& obj = oh.get();

  if (obj.type == msgpack::type::EXT) {
    const auto& ext = obj.via.ext;
    if (ext.type() == 1) {
      // Ext code 1 = CudaIPCWrapper pickle
      return decode_cuda_ipc_from_pickle(
          reinterpret_cast<const uint8_t*>(ext.data()), ext.size);
    }
  }

  // If it's not an Ext, try to decode as raw pickle bytes
  return decode_cuda_ipc_from_pickle(data, len);
}

}  // namespace server
}  // namespace lmcache
