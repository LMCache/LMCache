// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace lmcache::rdma {

class RdmaClient;

struct WcFailure {
  int status = 0;
  std::string status_name;
  std::uint32_t vendor_err = 0;
  std::uint64_t wr_id = 0;
  std::uint32_t qp_num = 0;
  std::string device_name;
  std::uint8_t port_num = 0;
  int gid_index = -1;
};

struct WcFailureCount {
  int status = 0;
  std::string status_name;
  std::uint64_t count = 0;
};

struct WcFailureDiagnostics {
  std::uint64_t total = 0;
  std::optional<WcFailure> last;
  std::vector<WcFailureCount> counts;
};

// Owns one verbs device, one PD, one registration of the LMCache L1 region,
// and the TCP control listener used to establish RC queue pairs.  The data
// plane is libibverbs RDMA READ; TCP never carries KV bytes.
class RdmaContext {
 public:
  RdmaContext(std::uint64_t base_address, std::uint64_t length,
              std::string listen_url, std::string advertise_url,
              std::string device_name, std::uint8_t port_num, int gid_index,
              std::uint32_t queue_depth, std::uint32_t handshake_timeout_ms);
  ~RdmaContext();

  RdmaContext(const RdmaContext&) = delete;
  RdmaContext& operator=(const RdmaContext&) = delete;

  std::shared_ptr<RdmaClient> connect(const std::string& peer_url);
  void close();

  std::string device_name() const;
  int gid_index() const;
  std::uint8_t port_num() const;
  std::uint32_t queue_depth() const;
  std::uint64_t registered_bytes() const;
  std::size_t inbound_connection_count() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

// One active RC connection used only for one-sided reads from a peer L1.
// A completion-channel thread blocks in poll(2) and drains the CQ. There is no
// busy-poll or transfer-worker pool.
class RdmaClient {
 public:
  class Impl;
  explicit RdmaClient(std::shared_ptr<Impl> impl);
  ~RdmaClient();

  RdmaClient(const RdmaClient&) = delete;
  RdmaClient& operator=(const RdmaClient&) = delete;

  std::uint64_t submit_read(const std::vector<std::uint64_t>& local_offsets,
                            const std::vector<std::uint64_t>& remote_offsets,
                            const std::vector<std::uint32_t>& sizes);

  // Returns (finished, succeeded, object_count).  A finished task is consumed;
  // querying the same id again raises.
  std::tuple<bool, bool, std::size_t> query_read_status(std::uint64_t task_id);
  void close();

  bool healthy() const;
  std::size_t outstanding_tasks() const;
  std::uint64_t submitted_reads() const;
  std::uint64_t submitted_bytes() const;
  std::uint64_t completed_reads() const;
  std::uint64_t failed_reads() const;
  WcFailureDiagnostics wc_failure_diagnostics() const;

 private:
  std::shared_ptr<Impl> impl_;
};

}  // namespace lmcache::rdma
