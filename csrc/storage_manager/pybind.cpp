// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttl_lock.h"
#include "bitmap.h"
#include "utils.h"

namespace py = pybind11;

using lmcache::storage_manager::Bitmap;
using lmcache::storage_manager::TTLLock;
using lmcache::utils::ParallelPatternMatcher;
using lmcache::utils::RangePatternMatcher;

PYBIND11_MODULE(native_storage_ops, m) {
  m.doc() = "Native storage operations for LMCache";

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
      .def(
          "gather",
          [](const Bitmap& self, const py::list& items) {
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
}
