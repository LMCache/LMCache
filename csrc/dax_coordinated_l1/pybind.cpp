// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dax_coordinated_l1/index_core.h"
#include "dax_coordinated_l1/visibility_x86.h"

namespace py = pybind11;

using namespace lmcache::dax_coordinated_l1;

namespace {

Digest bytes_to_digest(const py::bytes& value, const char* field_name) {
  const auto input = static_cast<std::string>(value);
  if (input.size() != 32) {
    throw std::invalid_argument(std::string(field_name) +
                                " must contain exactly 32 bytes");
  }
  Digest output{};
  std::memcpy(output.data(), input.data(), output.size());
  return output;
}

}  // namespace

PYBIND11_MODULE(lmcache_dax_coordinated_l1, module) {
  module.doc() = "Native DAX-Coordinated L1 lifecycle and visibility core";

  py::enum_<OperationResult>(module, "DaxCoordinatedL1Result")
      .value("SUCCESS", OperationResult::kSuccess)
      .value("NOT_FOUND", OperationResult::kNotFound)
      .value("TARGET_INVALIDATING", OperationResult::kTargetInvalidating)
      .value("WRITER_BUSY", OperationResult::kWriterBusy)
      .value("NO_FREE_BUCKET", OperationResult::kNoFreeBucket)
      .value("NO_LOCAL_PAYLOAD_SLOT", OperationResult::kNoLocalPayloadSlot)
      .value("GENERATION_MISMATCH", OperationResult::kGenerationMismatch)
      .value("ACTIVE_READER", OperationResult::kActiveReader)
      .value("OWNER_MISMATCH", OperationResult::kOwnerMismatch)
      .value("INVALID_STATE", OperationResult::kInvalidState)
      .value("CORRUPT_FORWARD_BACKREF", OperationResult::kCorruptForwardBackref)
      .value("RECOVERY_REQUIRED", OperationResult::kRecoveryRequired);

  py::class_<ReservationResult>(module, "DaxCoordinatedL1Reservation")
      .def_readonly("result", &ReservationResult::result)
      .def_readonly("token", &ReservationResult::token)
      .def_readonly("global_payload_slot_id",
                    &ReservationResult::global_payload_slot_id)
      .def_readonly("bucket_generation", &ReservationResult::bucket_generation)
      .def_readonly("slot_generation", &ReservationResult::slot_generation)
      .def_readonly("payload_offset", &ReservationResult::payload_offset)
      .def_readonly("payload_length", &ReservationResult::payload_length)
      .def_readonly("layout_id", &ReservationResult::layout_id);

  py::class_<CoreStatus>(module, "DaxCoordinatedL1CoreStatus")
      .def_readonly("free_slots_by_rank", &CoreStatus::free_slots_by_rank)
      .def_readonly("free_slots", &CoreStatus::free_slots)
      .def_readonly("used_slots", &CoreStatus::used_slots)
      .def_readonly("active_read_reservations",
                    &CoreStatus::active_read_reservations)
      .def_readonly("active_write_reservations",
                    &CoreStatus::active_write_reservations);

  py::class_<IndexCore>(module, "DevDaxBucketIndexCore")
      .def(py::init<std::uintptr_t, std::uint64_t, std::uint32_t,
                    const FormatParameters&,
                    const std::vector<std::array<std::uint64_t, 4>>&, bool,
                    bool>(),
           py::arg("base_address"), py::arg("region_size"),
           py::arg("participant_id"), py::arg("expected"),
           py::arg("payload_ranges"), py::arg("skip_payload_flush") = false,
           py::arg("memcheck_on_attach") = false)
      .def(
          "reserve_write",
          [](IndexCore& core, const py::bytes& key_digest,
             std::uint32_t payload_length, std::uint32_t layout_id,
             std::uint32_t allocation_rank) {
            const auto digest = bytes_to_digest(key_digest, "key digest");
            py::gil_scoped_release release;
            return core.reserve_write(digest, payload_length, layout_id,
                                      allocation_rank);
          },
          py::arg("key_digest"), py::arg("payload_length"),
          py::arg("layout_id"), py::arg("allocation_rank") = 0)
      .def("finish_writes", &IndexCore::finish_writes,
           py::call_guard<py::gil_scoped_release>())
      .def("finish_write_and_reserve_read",
           &IndexCore::finish_write_and_reserve_read,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "reserve_reads",
          [](IndexCore& core, const std::vector<py::bytes>& key_digests,
             std::uint32_t read_count) {
            std::vector<Digest> digests;
            digests.reserve(key_digests.size());
            for (const auto& digest : key_digests) {
              digests.push_back(bytes_to_digest(digest, "key digest"));
            }
            py::gil_scoped_release release;
            return core.reserve_reads(digests, read_count);
          },
          py::arg("key_digests"), py::arg("read_count"))
      .def("finish_reads", &IndexCore::finish_reads,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "delete_key",
          [](IndexCore& core, const py::bytes& key_digest) {
            const auto digest = bytes_to_digest(key_digest, "key digest");
            py::gil_scoped_release release;
            return core.delete_key(digest);
          },
          py::arg("key_digest"))
      .def("report_status", &IndexCore::report_status,
           py::call_guard<py::gil_scoped_release>())
      .def("memcheck", &IndexCore::memcheck,
           py::call_guard<py::gil_scoped_release>())
      .def("close", &IndexCore::close,
           py::call_guard<py::gil_scoped_release>());

  py::class_<LayoutDescription>(module, "DaxCoordinatedL1Layout")
      .def_readonly("participant_registry_offset",
                    &LayoutDescription::participant_registry_offset)
      .def_readonly("bucket_metadata_offsets",
                    &LayoutDescription::bucket_metadata_offsets)
      .def_readonly("bucket_peterson_offset",
                    &LayoutDescription::bucket_peterson_offset)
      .def_readonly("reader_activity_offset",
                    &LayoutDescription::reader_activity_offset)
      .def_readonly("payload_header_offset",
                    &LayoutDescription::payload_header_offset)
      .def_readonly("required_metadata_bytes",
                    &LayoutDescription::required_metadata_bytes)
      .def_readonly("required_payload_bytes",
                    &LayoutDescription::required_payload_bytes);

  // Immutable Python handle shared by format and attach.
  py::class_<FormatParameters>(module, "DaxCoordinatedL1Parameters")
      .def_property_readonly("layout", &FormatParameters::layout);
  module.def(
      "dax_coordinated_l1_parameters",
      [](std::uint64_t region_epoch,
         const std::vector<std::uint64_t>& buckets_per_level,
         std::uint64_t payload_slot_bytes, std::uint64_t payload_slot_count,
         std::uint64_t payload_alignment, std::uint32_t visibility_mode,
         std::uint64_t participant_0_slot_count, const py::bytes& layout_digest,
         const py::bytes& hardware_digest, const py::bytes& region_id_digest,
         std::uint64_t payload_region_size, std::uint32_t participant_count) {
        FormatParameters parameters{};
        parameters.region_epoch = region_epoch;
        parameters.buckets_per_level = buckets_per_level;
        parameters.payload_slot_bytes = payload_slot_bytes;
        parameters.payload_slot_count = payload_slot_count;
        parameters.payload_alignment = payload_alignment;
        parameters.visibility_mode = visibility_mode;
        parameters.payload_region_size = payload_region_size;
        parameters.participant_0_slot_count = participant_0_slot_count;
        parameters.participant_count = participant_count;
        parameters.layout_profile_digest =
            bytes_to_digest(layout_digest, "layout digest");
        parameters.hardware_qualification_digest =
            bytes_to_digest(hardware_digest, "hardware digest");
        parameters.region_id_digest =
            bytes_to_digest(region_id_digest, "region ID digest");
        return parameters;
      },
      py::arg("region_epoch"), py::arg("buckets_per_level"),
      py::arg("payload_slot_bytes"), py::arg("payload_slot_count"),
      py::arg("payload_alignment"), py::arg("visibility_mode"),
      py::arg("participant_0_slot_count"), py::arg("layout_digest"),
      py::arg("hardware_digest"), py::arg("region_id_digest"),
      py::arg("payload_region_size"), py::arg("participant_count") = 2);

  module.def(
      "dax_coordinated_l1_is_formatted",
      [](std::uintptr_t address) {
        auto* superblock = reinterpret_cast<Superblock*>(address);
        // Refresh every digest-bearing line, not just the publication marker.
        refresh_range(superblock, sizeof(*superblock));
        return superblock->magic == kMagic;
      },
      py::arg("base_address"), py::call_guard<py::gil_scoped_release>());

  module.def(
      "format_dax_coordinated_l1_region",
      [](std::uintptr_t address, std::uint64_t size,
         const FormatParameters& parameters) {
        format_region(reinterpret_cast<void*>(address), size, parameters);
      },
      py::call_guard<py::gil_scoped_release>());

  module.def("dax_coordinated_l1_cpu_profile", []() {
    const auto profile = query_cpu_visibility_profile();
    py::dict output;
    output["is_x86_64"] = profile.is_x86_64;
    output["has_clflush"] = profile.has_clflush;
    output["has_clflushopt"] = profile.has_clflushopt;
    output["cache_line_bytes"] = profile.cache_line_bytes;
    return output;
  });
  module.def(
      "dax_coordinated_l1_publish",
      [](std::uintptr_t address, std::uint64_t length) {
        py::gil_scoped_release release;
        publish_range(reinterpret_cast<void*>(address), length);
      },
      py::arg("address"), py::arg("length"));
  module.def(
      "dax_coordinated_l1_refresh",
      [](std::uintptr_t address, std::uint64_t length) {
        py::gil_scoped_release release;
        refresh_range(reinterpret_cast<void*>(address), length);
      },
      py::arg("address"), py::arg("length"));
}
