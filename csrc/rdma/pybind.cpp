// SPDX-License-Identifier: Apache-2.0

#include "rdma_transport.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

namespace py = pybind11;
using lmcache::rdma::RdmaClient;
using lmcache::rdma::RdmaContext;
using lmcache::rdma::WcFailure;
using lmcache::rdma::WcFailureDiagnostics;

namespace {

py::dict wc_failure_to_dict(const WcFailure& failure) {
  py::dict value;
  value["status"] = failure.status;
  value["status_name"] = failure.status_name;
  value["vendor_err"] = failure.vendor_err;
  value["wr_id"] = failure.wr_id;
  value["qp_num"] = failure.qp_num;
  value["device_name"] = failure.device_name;
  value["port_num"] = static_cast<unsigned int>(failure.port_num);
  value["gid_index"] = failure.gid_index;
  return value;
}

py::dict wc_diagnostics_to_dict(const WcFailureDiagnostics& diagnostics) {
  py::dict value;
  value["total"] = diagnostics.total;
  if (diagnostics.last.has_value()) {
    value["last"] = wc_failure_to_dict(*diagnostics.last);
  } else {
    value["last"] = py::none();
  }
  py::list counts;
  for (const auto& count : diagnostics.counts) {
    py::dict item;
    item["status"] = count.status;
    item["status_name"] = count.status_name;
    item["count"] = count.count;
    counts.append(std::move(item));
  }
  value["counts"] = std::move(counts);
  return value;
}

}  // namespace

PYBIND11_MODULE(rdma_l1_ops, module) {
  module.doc() =
      "LMCache-owned direct libibverbs RC transport for host L1 memory";

  py::class_<RdmaClient, std::shared_ptr<RdmaClient>>(module, "RdmaClient")
      .def(
          "submit_read",
          [](RdmaClient& client,
             const std::vector<std::uint64_t>& local_offsets,
             const std::vector<std::uint64_t>& remote_offsets,
             const std::vector<std::uint32_t>& sizes) {
            py::gil_scoped_release release;
            return client.submit_read(local_offsets, remote_offsets, sizes);
          },
          py::arg("local_offsets"), py::arg("remote_offsets"), py::arg("sizes"))
      .def(
          "query_read_status",
          [](RdmaClient& client, std::uint64_t task_id) {
            py::gil_scoped_release release;
            return client.query_read_status(task_id);
          },
          py::arg("task_id"))
      .def("close",
           [](RdmaClient& client) {
             py::gil_scoped_release release;
             client.close();
           })
      .def_property_readonly("healthy", &RdmaClient::healthy)
      .def_property_readonly("outstanding_tasks",
                             &RdmaClient::outstanding_tasks)
      .def_property_readonly("submitted_reads", &RdmaClient::submitted_reads)
      .def_property_readonly("submitted_bytes", &RdmaClient::submitted_bytes)
      .def_property_readonly("completed_reads", &RdmaClient::completed_reads)
      .def_property_readonly("failed_reads", &RdmaClient::failed_reads)
      .def_property_readonly(
          "wc_failure_diagnostics", [](const RdmaClient& client) {
            return wc_diagnostics_to_dict(client.wc_failure_diagnostics());
          });

  py::class_<RdmaContext, std::shared_ptr<RdmaContext>>(module, "RdmaContext")
      .def(py::init([](std::uint64_t base_address, std::uint64_t length,
                       std::string listen_url, std::string advertise_url,
                       std::string device_name, std::uint8_t port_num,
                       int gid_index, std::uint32_t queue_depth,
                       std::uint32_t handshake_timeout_ms) {
             py::gil_scoped_release release;
             return std::make_shared<RdmaContext>(
                 base_address, length, std::move(listen_url),
                 std::move(advertise_url), std::move(device_name), port_num,
                 gid_index, queue_depth, handshake_timeout_ms);
           }),
           py::arg("base_address"), py::arg("length"), py::arg("listen_url"),
           py::arg("advertise_url"), py::arg("device_name"),
           py::arg("port_num") = 1, py::arg("gid_index") = -1,
           py::arg("queue_depth") = 4096,
           py::arg("handshake_timeout_ms") = 10000)
      .def(
          "connect",
          [](RdmaContext& context, const std::string& peer_url) {
            py::gil_scoped_release release;
            return context.connect(peer_url);
          },
          py::arg("peer_url"))
      .def("close",
           [](RdmaContext& context) {
             py::gil_scoped_release release;
             context.close();
           })
      .def_property_readonly("device_name", &RdmaContext::device_name)
      .def_property_readonly("gid_index", &RdmaContext::gid_index)
      .def_property_readonly("port_num", &RdmaContext::port_num)
      .def_property_readonly("queue_depth", &RdmaContext::queue_depth)
      .def_property_readonly("registered_bytes", &RdmaContext::registered_bytes)
      .def_property_readonly("inbound_connection_count",
                             &RdmaContext::inbound_connection_count);
}
