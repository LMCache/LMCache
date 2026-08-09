// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttl_lock.h"
#include "bitmap.h"
#include "fold.h"
#include "periodic_event_notifier.h"
#include "utils.h"
#include "engine_kv_format.h"
#include "kv_transfer_types.h"

namespace py = pybind11;

using lmcache::lmcache_native::Bitmap;
using lmcache::lmcache_native::PeriodicEventNotifier;
using lmcache::lmcache_native::TTLLock;
using lmcache::utils::ParallelPatternMatcher;
using lmcache::utils::RangePatternMatcher;

PYBIND11_MODULE(lmcache_native, m) {
  m.doc() = "Native operations for LMCache (device-independent)";

  // Device-independent KV-format and transfer enums. These live in the
  // common (non-CUDA) native module so they are available regardless of the
  // GPU backend, and so they are the single source of truth shared with the
  // pure-Python fallback in ``lmcache.v1.platform.ops_types``.
  py::enum_<EngineKVFormat>(m, "EngineKVFormat")
      .value("NB_NL_TWO_BS_NH_HS", EngineKVFormat::NB_NL_TWO_BS_NH_HS)
      .value("NL_X_TWO_NB_BS_NH_HS", EngineKVFormat::NL_X_TWO_NB_BS_NH_HS)
      .value("NL_X_NB_TWO_BS_NH_HS", EngineKVFormat::NL_X_NB_TWO_BS_NH_HS)
      .value("NL_X_NB_BS_HS", EngineKVFormat::NL_X_NB_BS_HS)
      .value("TWO_X_NL_X_NBBS_NH_HS", EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS)
      .value("NL_X_NBBS_ONE_HS", EngineKVFormat::NL_X_NBBS_ONE_HS)
      .value("NL_X_TWO_NB_NH_BS_HS", EngineKVFormat::NL_X_TWO_NB_NH_BS_HS)
      .value("NL_X_NB_TWO_NH_BS_HS", EngineKVFormat::NL_X_NB_TWO_NH_BS_HS)
      .value("NB_NL_TWO_NH_BS_HS", EngineKVFormat::NB_NL_TWO_NH_BS_HS)
      .value("TWO_X_NL_X_NB_BS_NH_HS", EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS)
      .value("NL_X_NB_NH_BS_TWO_HS", EngineKVFormat::NL_X_NB_NH_BS_TWO_HS)
      .value("NL_X_NB_BS_NH_TWO_HS", EngineKVFormat::NL_X_NB_BS_NH_TWO_HS)
      .value("NL_X_NB_NH_BS_CS", EngineKVFormat::NL_X_NB_NH_BS_CS)
      .value("NL_X_NB_BS_NH_CS", EngineKVFormat::NL_X_NB_BS_NH_CS)
      .value("NL_X_NB_BSV_BSS", EngineKVFormat::NL_X_NB_BSV_BSS);
  m.attr("GPUKVFormat") = m.attr("EngineKVFormat");

  py::enum_<TransferDirection>(m, "TransferDirection")
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H);

  m.def("is_kv_list", &is_kv_list, py::arg("format"),
        "Return whether the format stores KV as a list of per-token KV "
        "tensors (kv_size == 2).");
  m.def("is_layer_list", &is_layer_list, py::arg("format"),
        "Return whether the format stores one list entry per layer.");
  m.def("is_cross_layer", &is_cross_layer, py::arg("format"),
        "Return whether the format stacks KV from different layers into a "
        "single tensor.");
  m.def("is_mla", &is_mla, py::arg("format"),
        "Return whether the format is an MLA variant (single latent KV "
        "head).");

  m.def("fold", &lmcache::lmcache_native::fold, py::arg("found"),
        py::arg("num_chunks"), py::arg("num_ranks"), py::arg("group_windows"),
        "Fold per-(group, chunk, rank) presence into a servable-prefix-lengths "
        "bitmap (size num_chunks + 1); bit L set iff every object group can "
        "serve a length-L prefix.");
  m.def(
      "unfold", &lmcache::lmcache_native::unfold, py::arg("hit_length"),
      py::arg("num_chunks"), py::arg("num_ranks"), py::arg("group_windows"),
      "Expand a model-wide hit length into the per-group retain mask over the "
      "group x chunk x kv_rank layout.");

  py::class_<TTLLock>(m, "TTLLock")
      .def(py::init<uint32_t>(), py::arg("ttl_second") = 300,
           "Construct a TTLLock with the specified TTL duration in "
           "seconds. Default is 300 seconds.")
      .def("lock", &TTLLock::lock,
           "Increment the lock counter by 1 and update the TTL. "
           "If the previous TTL has expired, reset counter to 1.")
      .def("unlock", &TTLLock::unlock,
           "Decrement the lock counter by 1 (minimum 0).")
      .def("is_locked", &TTLLock::is_locked,
           "Check if the lock is held (counter > 0 and TTL not expired).")
      .def("reset", &TTLLock::reset,
           "Reset the lock to initial state (counter = 0, TTL expired).");

  py::class_<Bitmap>(m, "Bitmap")
      .def(py::init<size_t>(), py::arg("size"),
           "Construct a Bitmap with the specified size.")
      .def(py::init<size_t, size_t>(), py::arg("size"), py::arg("prefix_bits"),
           "Construct a Bitmap with the specified size and first N prefix "
           "bits set to 1.")
      .def("set", &Bitmap::set, py::arg("index"),
           "Set the bit at the specified index to 1.")
      .def("clear", &Bitmap::clear, py::arg("index"),
           "Clear the bit at the specified index to 0.")
      .def("test", &Bitmap::test, py::arg("index"),
           "Test the bit at the specified index.")
      .def("popcount", &Bitmap::popcount, "Count the number of bits set to 1.")
      .def("count_leading_zeros", &Bitmap::clz,
           "Count the number of leading zeros.")
      .def("count_leading_ones", &Bitmap::clo,
           "Count the number of leading ones.")
      .def("highest_set_bit", &Bitmap::highest_set_bit,
           "Index of the highest set bit, or -1 if no bit is set (the return "
           "is signed so the empty bitmap is representable).")
      .def("__and__", &Bitmap::operator&, py::arg("other"),
           "Bitwise AND operation between two bitmaps.")
      .def("__or__", &Bitmap::operator|, py::arg("other"),
           "Bitwise OR operation between two bitmaps.")
      .def("__invert__", &Bitmap::operator~,
           "Bitwise NOT operation (flip all bits).")
      .def("get_indices_list", &Bitmap::get_indices,
           "Return a list of indices where the bit is set to 1.")
      .def("get_indices_set", &Bitmap::get_indices_set,
           "Return a set of indices where the bit is set to 1.")
      .def("batched_set", &Bitmap::batched_set, py::arg("indices"),
           "Set every bit in indices to 1 (positions >= size ignored).")
      .def("set_range", &Bitmap::set_range, py::arg("start"), py::arg("end"),
           "Set every bit in the half-open range [start, end) to 1 (end "
           "clamped to size). Fills whole bytes, so far cheaper than per-bit "
           "set for a contiguous span.")
      .def(
          "gather",
          [](const Bitmap& self, const py::sequence& items) {
            auto indices = self.get_indices();
            py::list result;
            for (auto idx : indices) {
              if (idx < static_cast<size_t>(py::len(items))) {
                result.append(items[idx]);
              }
            }
            return result;
          },
          py::arg("items"),
          "Return elements from items at indices where the bit is set to 1.")
      .def("__repr__", &Bitmap::to_string,
           "Convert the bitmap to a string representation.");

  py::class_<ParallelPatternMatcher>(m, "ParallelPatternMatcher")
      .def(py::init<const std::vector<int>&>(), py::arg("pattern"),
           "Construct a ParallelPatternMatcher with the specified pattern.")
      .def("match", &ParallelPatternMatcher::match, py::arg("data"),
           "Match the pattern in the given data and return a sorted list "
           "of positions where the pattern starts.");

  py::class_<RangePatternMatcher>(m, "RangePatternMatcher")
      .def(py::init<const std::vector<int>&, const std::vector<int>&>(),
           py::arg("start_pattern"), py::arg("end_pattern"),
           "Construct a RangePatternMatcher with start and end patterns. ")
      .def("match", &RangePatternMatcher::match, py::arg("data"),
           "Match ranges in the given data. Returns a list of (start_pos, "
           "end_pos) tuples where start_pos is the beginning of the start "
           "pattern and end_pos is the exclusive index after the end pattern. "
           "When multiple end patterns exist after a start pattern, matches "
           "the first one (minimal range).");

  py::class_<PeriodicEventNotifier>(m, "PeriodicEventNotifier")
      .def_static("create", &PeriodicEventNotifier::create,
                  py::call_guard<py::gil_scoped_release>(),
                  py::arg("interval_ms"), py::arg("use_eventfd"),
                  "Create the singleton PeriodicEventNotifier. "
                  "Idempotent -- second call is a no-op.")
      .def_static("get", &PeriodicEventNotifier::get,
                  py::return_value_policy::reference,
                  "Get the singleton instance, or None if not created.")
      .def_static("shutdown", &PeriodicEventNotifier::shutdown,
                  py::call_guard<py::gil_scoped_release>(),
                  "Shut down the singleton. Idempotent.")
      .def("register_fd", &PeriodicEventNotifier::register_fd, py::arg("fd"),
           "Register a file descriptor for periodic signaling.")
      .def("unregister_fd", &PeriodicEventNotifier::unregister_fd,
           py::arg("fd"), "Unregister a file descriptor.")
      .def("set_interval_ms", &PeriodicEventNotifier::set_interval_ms,
           py::arg("interval_ms"),
           "Change the notification interval in milliseconds.");
}
