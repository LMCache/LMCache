// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "resp.h"

namespace py = pybind11;

/*
python interface for LMCache Redis client

all methods release the GIL as soon as raw buffer pointers are extracted,
allowing Python to continue executing while C++ workers perform I/O operations.

python surface area:
  client = LMCacheRedisClient(host, port, num_workers)
  fd = client.event_fd()
  asyncio.get_running_loop().add_reader(fd, callback_that_drains_completions)

  # the python caller is responsible for managing the future ids for each
request # and matching them to the completions pending_futures = {}

  future_id = client.submit_get(key, memoryview)
  future_id = client.submit_set(key, memoryview)
  future_id = client.submit_exists(key)
  future_id = client.submit_batch_get(keys, memoryviews)
  future_id = client.submit_batch_set(keys, memoryviews)
  future_id = client.submit_batch_exists(keys)

  completions = client.drain_completions()
  # Returns: [(future_id, ok, error, result_bools), ...]
  # result_bools is None for get/set, a list for exists operations
  # get and set are side-effectful and have no results to extract in the
completion

  client.close()
*/

PYBIND11_MODULE(lmcache_redis, m) {
  py::class_<MultiRESPClient>(m, "LMCacheRedisClient")
      .def(py::init<std::string, int, int, std::string, std::string>(),
           py::arg("host"), py::arg("port"), py::arg("num_workers"),
           py::arg("username") = "", py::arg("password") = "")
      // event_fd: Get file descriptor for async notification
      // fd = client.event_fd()
      .def("event_fd", &MultiRESPClient::event_fd)
      // submit_get: Submit a GET operation for a single key
      // future_id = client.submit_get(key: str, mv: memoryview)
      .def(
          "submit_get",
          [](MultiRESPClient& self, const std::string& key, py::memoryview mv) {
            // extract buffer info under GIL
            py::buffer_info info = py::buffer(mv).request();
            if (info.ndim != 1)
              throw std::runtime_error("memoryview must be 1D");
            if (info.itemsize != 1)
              throw std::runtime_error("memoryview must be byte addressable");

            std::vector<std::string> keys = {key};
            std::vector<void*> bufs = {info.ptr};
            std::vector<size_t> lens = {static_cast<size_t>(info.size)};
            size_t batch_chunk_num_bytes = static_cast<size_t>(info.size);

            // release GIL for C++ operation (python guarantees buffer lifetime)
            py::gil_scoped_release release;
            return self.submit_batch_get(keys, bufs, lens,
                                         batch_chunk_num_bytes);
          },
          py::arg("key"), py::arg("memoryview"))
      // submit_set: Submit a SET operation for a single key
      // future_id = client.submit_set(key: str, mv: memoryview)
      .def(
          "submit_set",
          [](MultiRESPClient& self, const std::string& key, py::memoryview mv) {
            // extract buffer info under GIL
            py::buffer_info info = py::buffer(mv).request();
            if (info.ndim != 1)
              throw std::runtime_error("memoryview must be 1D");
            if (info.itemsize != 1)
              throw std::runtime_error("memoryview must be byte addressable");

            std::vector<std::string> keys = {key};
            std::vector<void*> bufs = {info.ptr};
            std::vector<size_t> lens = {static_cast<size_t>(info.size)};
            size_t batch_chunk_num_bytes = static_cast<size_t>(info.size);

            // release GIL for C++ operation (python guarantees buffer lifetime)
            py::gil_scoped_release release;
            return self.submit_batch_set(keys, bufs, lens,
                                         batch_chunk_num_bytes);
          },
          py::arg("key"), py::arg("memoryview"))
      // submit_exists: Check if a single key exists
      // future_id = client.submit_exists(key: str)
      .def(
          "submit_exists",
          [](MultiRESPClient& self, const std::string& key) {
            std::vector<std::string> keys = {key};
            // release GIL for C++ operation
            py::gil_scoped_release release;
            return self.submit_batch_exists(keys);
          },
          py::arg("key"))
      // submit_batch_get: Submit GET operations for multiple keys
      // future_id = client.submit_batch_get(keys: list[str], memviews:
      // list[memoryview])
      .def(
          "submit_batch_get",
          [](MultiRESPClient& self, const std::vector<std::string>& keys,
             py::list memviews) {
            if (keys.size() != memviews.size()) {
              throw std::runtime_error("keys and memviews size mismatch");
            }
            if (keys.empty()) {
              throw std::runtime_error("keys list is empty");
            }

            // extract all buffer info under GIL
            std::vector<void*> bufs;
            std::vector<size_t> lens;
            bufs.reserve(keys.size());
            lens.reserve(keys.size());

            for (size_t i = 0; i < keys.size(); ++i) {
              py::memoryview mv = memviews[i].cast<py::memoryview>();
              py::buffer_info info = py::buffer(mv).request();

              if (info.ndim != 1) throw std::runtime_error("buffer must be 1D");
              if (info.itemsize != 1)
                throw std::runtime_error("buffer must be byte addressable");

              bufs.push_back(info.ptr);
              lens.push_back(static_cast<size_t>(info.size));
            }

            // use the first buffer's size as batch_chunk_num_bytes (all must
            // match)
            size_t batch_chunk_num_bytes = lens[0];

            // release GIL for C++ operation (python guarantees buffer lifetime)
            py::gil_scoped_release release;
            return self.submit_batch_get(keys, bufs, lens,
                                         batch_chunk_num_bytes);
          },
          py::arg("keys"), py::arg("memoryviews"))
      // submit_batch_set: Submit SET operations for multiple keys
      // future_id = client.submit_batch_set(keys: list[str], memviews:
      // list[memoryview])
      .def(
          "submit_batch_set",
          [](MultiRESPClient& self, const std::vector<std::string>& keys,
             py::list memviews) {
            if (keys.size() != memviews.size()) {
              throw std::runtime_error("keys and memviews size mismatch");
            }
            if (keys.empty()) {
              throw std::runtime_error("keys list is empty");
            }

            // extract all buffer info under GIL
            std::vector<void*> bufs;
            std::vector<size_t> lens;
            bufs.reserve(keys.size());
            lens.reserve(keys.size());

            for (size_t i = 0; i < keys.size(); ++i) {
              py::memoryview mv = memviews[i].cast<py::memoryview>();
              py::buffer_info info = py::buffer(mv).request();

              if (info.ndim != 1) throw std::runtime_error("buffer must be 1D");
              if (info.itemsize != 1)
                throw std::runtime_error("buffer must be byte addressable");

              bufs.push_back(info.ptr);
              lens.push_back(static_cast<size_t>(info.size));
            }

            // use the first buffer's size as batch_chunk_num_bytes (all must
            // match)
            size_t batch_chunk_num_bytes = lens[0];

            // release GIL for C++ operation (python guarantees buffer lifetime)
            py::gil_scoped_release release;
            return self.submit_batch_set(keys, bufs, lens,
                                         batch_chunk_num_bytes);
          },
          py::arg("keys"), py::arg("memoryviews"))
      // submit_batch_exists: Check if multiple keys exist
      // future_id = client.submit_batch_exists(keys: list[str])
      .def(
          "submit_batch_exists",
          [](MultiRESPClient& self, const std::vector<std::string>& keys) {
            // release GIL for C++ operation
            py::gil_scoped_release release;
            return self.submit_batch_exists(keys);
          },
          py::arg("keys"))
      // drain_completions: Drain and convert completed operations to Python
      // tuples completions = client.drain_completions() Returns: [(future_id,
      // ok, error, result_bools), ...] result_bools is None for get/set, a list
      // for exists Note: This also drains the eventfd
      .def("drain_completions",
           [](MultiRESPClient& self) {
             // call C++ method without holding GIL
             std::vector<Completion> completions;
             {
               py::gil_scoped_release release;
               completions = self.drain_completions();
             }

             // convert results to python objects (requires GIL)
             py::list result;
             for (auto& c : completions) {
               // Convert result_bytes to python list of bools if not empty
               py::object bools_obj = py::none();
               if (!c.result_bytes.empty()) {
                 py::list bools_list;
                 for (uint8_t b : c.result_bytes) {
                   bools_list.append(bool(b));
                 }
                 bools_obj = bools_list;
               }

               // Return tuple: (future_id, ok, error, result_bools)
               result.append(
                   py::make_tuple(c.future_id, c.ok, c.error, bools_obj));
             }

             return result;
           })
      // close: Shutdown the client and cleanup resources
      // client.close()
      .def("close", &MultiRESPClient::close);
}
