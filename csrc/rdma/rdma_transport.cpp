// SPDX-License-Identifier: Apache-2.0

#include "rdma_transport.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lmcache::rdma {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {'L', 'M', 'C', 'R',
                                                'D', 'M', 'A', '1'};
constexpr std::uint32_t kProtocolVersion = 2;
constexpr std::size_t kHelloBytes = 64;
constexpr std::uint8_t kReadyByte = 0x52;
constexpr std::uint8_t kCloseByte = 0x58;
constexpr std::size_t kWcStatusSlots = 32;
constexpr std::size_t kUnknownWcStatusSlot = kWcStatusSlots - 1;

const char* wc_status_name(int status) {
  switch (static_cast<ibv_wc_status>(status)) {
    case IBV_WC_SUCCESS:
      return "IBV_WC_SUCCESS";
    case IBV_WC_LOC_LEN_ERR:
      return "IBV_WC_LOC_LEN_ERR";
    case IBV_WC_LOC_QP_OP_ERR:
      return "IBV_WC_LOC_QP_OP_ERR";
    case IBV_WC_LOC_EEC_OP_ERR:
      return "IBV_WC_LOC_EEC_OP_ERR";
    case IBV_WC_LOC_PROT_ERR:
      return "IBV_WC_LOC_PROT_ERR";
    case IBV_WC_WR_FLUSH_ERR:
      return "IBV_WC_WR_FLUSH_ERR";
    case IBV_WC_MW_BIND_ERR:
      return "IBV_WC_MW_BIND_ERR";
    case IBV_WC_BAD_RESP_ERR:
      return "IBV_WC_BAD_RESP_ERR";
    case IBV_WC_LOC_ACCESS_ERR:
      return "IBV_WC_LOC_ACCESS_ERR";
    case IBV_WC_REM_INV_REQ_ERR:
      return "IBV_WC_REM_INV_REQ_ERR";
    case IBV_WC_REM_ACCESS_ERR:
      return "IBV_WC_REM_ACCESS_ERR";
    case IBV_WC_REM_OP_ERR:
      return "IBV_WC_REM_OP_ERR";
    case IBV_WC_RETRY_EXC_ERR:
      return "IBV_WC_RETRY_EXC_ERR";
    case IBV_WC_RNR_RETRY_EXC_ERR:
      return "IBV_WC_RNR_RETRY_EXC_ERR";
    case IBV_WC_LOC_RDD_VIOL_ERR:
      return "IBV_WC_LOC_RDD_VIOL_ERR";
    case IBV_WC_REM_INV_RD_REQ_ERR:
      return "IBV_WC_REM_INV_RD_REQ_ERR";
    case IBV_WC_REM_ABORT_ERR:
      return "IBV_WC_REM_ABORT_ERR";
    case IBV_WC_INV_EECN_ERR:
      return "IBV_WC_INV_EECN_ERR";
    case IBV_WC_INV_EEC_STATE_ERR:
      return "IBV_WC_INV_EEC_STATE_ERR";
    case IBV_WC_FATAL_ERR:
      return "IBV_WC_FATAL_ERR";
    case IBV_WC_RESP_TIMEOUT_ERR:
      return "IBV_WC_RESP_TIMEOUT_ERR";
    case IBV_WC_GENERAL_ERR:
      return "IBV_WC_GENERAL_ERR";
    case IBV_WC_TM_ERR:
      return "IBV_WC_TM_ERR";
    case IBV_WC_TM_RNDV_INCOMPLETE:
      return "IBV_WC_TM_RNDV_INCOMPLETE";
    default:
      return "IBV_WC_UNKNOWN";
  }
}

std::size_t wc_status_slot(int status) {
  if (status >= 0 && static_cast<std::size_t>(status) < kUnknownWcStatusSlot) {
    return static_cast<std::size_t>(status);
  }
  return kUnknownWcStatusSlot;
}

[[noreturn]] void throw_errno(const std::string& operation, int error = errno) {
  throw std::runtime_error(operation + ": " + std::strerror(error));
}

void check_zero(int rc, const std::string& operation) {
  if (rc != 0) {
    throw_errno(operation, rc > 0 ? rc : errno);
  }
}

struct HostPort {
  std::string host;
  std::string port;
};

HostPort parse_url(const std::string& url) {
  std::string value = url;
  const auto scheme = value.find("://");
  if (scheme != std::string::npos) value.erase(0, scheme + 3);
  const auto colon = value.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 == value.size()) {
    throw std::invalid_argument("invalid RDMA control URL: " + url);
  }
  return {value.substr(0, colon), value.substr(colon + 1)};
}

void set_nonblocking(int fd, bool enabled) {
  const int old_flags = fcntl(fd, F_GETFL, 0);
  if (old_flags < 0) throw_errno("fcntl(F_GETFL)");
  const int new_flags =
      enabled ? old_flags | O_NONBLOCK : old_flags & ~O_NONBLOCK;
  if (fcntl(fd, F_SETFL, new_flags) != 0) throw_errno("fcntl(F_SETFL)");
}

void set_socket_timeout(int fd, std::uint32_t timeout_ms) {
  timeval timeout{};
  timeout.tv_sec = static_cast<time_t>(timeout_ms / 1000);
  timeout.tv_usec = static_cast<suseconds_t>((timeout_ms % 1000) * 1000);
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0) {
    throw_errno("setsockopt(SO_RCVTIMEO)");
  }
  if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
    throw_errno("setsockopt(SO_SNDTIMEO)");
  }
}

using ControlDeadline = std::chrono::steady_clock::time_point;

ControlDeadline control_deadline(std::uint32_t timeout_ms) {
  return std::chrono::steady_clock::now() +
         std::chrono::milliseconds(timeout_ms);
}

int deadline_poll_timeout_ms(ControlDeadline deadline) {
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) return 0;
  const auto remaining =
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
          .count();
  return static_cast<int>(std::min<std::int64_t>(
      std::numeric_limits<int>::max(), std::max<std::int64_t>(1, remaining)));
}

void wait_for_socket(int fd, short events, ControlDeadline deadline) {
  while (true) {
    const int timeout_ms = deadline_poll_timeout_ms(deadline);
    if (timeout_ms == 0) {
      throw std::runtime_error("RDMA control handshake timed out");
    }
    pollfd pfd{fd, events, 0};
    const int rc = poll(&pfd, 1, timeout_ms);
    if (rc > 0) return;
    if (rc == 0) {
      throw std::runtime_error("RDMA control handshake timed out");
    }
    if (errno == EINTR) continue;
    throw_errno("poll RDMA control socket");
  }
}

void send_all(int fd, const void* data, std::size_t size,
              ControlDeadline deadline) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::size_t sent = 0;
  while (sent < size) {
    wait_for_socket(fd, POLLOUT, deadline);
    const ssize_t rc =
        send(fd, bytes + sent, size - sent, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (rc > 0) {
      sent += static_cast<std::size_t>(rc);
      continue;
    }
    if (rc < 0 && errno == EINTR) continue;
    if (rc < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;
    if (rc == 0) throw std::runtime_error("control socket closed during send");
    throw_errno("send RDMA control message");
  }
}

void recv_all(int fd, void* data, std::size_t size, ControlDeadline deadline) {
  auto* bytes = static_cast<std::uint8_t*>(data);
  std::size_t received = 0;
  while (received < size) {
    wait_for_socket(fd, POLLIN, deadline);
    const ssize_t rc =
        recv(fd, bytes + received, size - received, MSG_DONTWAIT);
    if (rc > 0) {
      received += static_cast<std::size_t>(rc);
      continue;
    }
    if (rc < 0 && errno == EINTR) continue;
    if (rc < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;
    if (rc == 0)
      throw std::runtime_error("control socket closed during receive");
    throw_errno("receive RDMA control message");
  }
}

int create_listener(const std::string& url) {
  const auto endpoint = parse_url(url);
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  addrinfo* addresses = nullptr;
  const char* host =
      endpoint.host == "0.0.0.0" ? nullptr : endpoint.host.c_str();
  const int gai = getaddrinfo(host, endpoint.port.c_str(), &hints, &addresses);
  if (gai != 0) {
    throw std::runtime_error("getaddrinfo(" + url + "): " + gai_strerror(gai));
  }

  int result = -1;
  for (addrinfo* address = addresses; address != nullptr;
       address = address->ai_next) {
    int fd =
        socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (fd < 0) continue;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (bind(fd, address->ai_addr, address->ai_addrlen) == 0 &&
        listen(fd, 16) == 0) {
      result = fd;
      break;
    }
    ::close(fd);
  }
  freeaddrinfo(addresses);
  if (result < 0) throw_errno("bind/listen " + url);
  set_nonblocking(result, true);
  return result;
}

template <typename RegisterSocket, typename UnregisterSocket>
int connect_with_timeout(const std::string& url, std::uint32_t timeout_ms,
                         ControlDeadline deadline, int cancel_fd,
                         RegisterSocket register_socket,
                         UnregisterSocket unregister_socket) {
  const auto endpoint = parse_url(url);
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* addresses = nullptr;
  const int gai = getaddrinfo(endpoint.host.c_str(), endpoint.port.c_str(),
                              &hints, &addresses);
  if (gai != 0) {
    throw std::runtime_error("getaddrinfo(" + url + "): " + gai_strerror(gai));
  }

  int result = -1;
  int last_error = ECONNREFUSED;
  for (addrinfo* address = addresses; address != nullptr;
       address = address->ai_next) {
    if (deadline_poll_timeout_ms(deadline) == 0) {
      last_error = ETIMEDOUT;
      break;
    }
    int fd =
        socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (fd < 0) {
      last_error = errno;
      continue;
    }
    bool registered = false;
    const auto close_candidate = [&] {
      if (registered) {
        unregister_socket(fd);
        registered = false;
      }
      ::close(fd);
    };
    try {
      register_socket(fd);
      registered = true;
      set_nonblocking(fd, true);
      int rc = ::connect(fd, address->ai_addr, address->ai_addrlen);
      if (rc != 0 && errno != EINPROGRESS) {
        last_error = errno;
        close_candidate();
        continue;
      }
      if (rc != 0) {
        pollfd fds[2] = {{fd, POLLOUT, 0}, {cancel_fd, POLLIN, 0}};
        while (true) {
          const int poll_timeout = deadline_poll_timeout_ms(deadline);
          if (poll_timeout == 0) {
            rc = 0;
            break;
          }
          fds[0].revents = 0;
          fds[1].revents = 0;
          rc = poll(fds, cancel_fd >= 0 ? 2 : 1, poll_timeout);
          if (rc < 0 && errno == EINTR) continue;
          break;
        }
        if (cancel_fd >= 0 &&
            (fds[1].revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL))) {
          throw std::runtime_error("RDMA context closed during peer connect");
        }
        if (rc <= 0) {
          last_error = rc == 0 ? ETIMEDOUT : errno;
          close_candidate();
          if (last_error == ETIMEDOUT) break;
          continue;
        }
        int socket_error = 0;
        socklen_t length = sizeof(socket_error);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &length) != 0 ||
            socket_error != 0) {
          last_error = socket_error == 0 ? errno : socket_error;
          close_candidate();
          continue;
        }
      }
      set_nonblocking(fd, false);
      set_socket_timeout(fd, timeout_ms);
      result = fd;
      break;
    } catch (...) {
      close_candidate();
      freeaddrinfo(addresses);
      throw;
    }
  }
  freeaddrinfo(addresses);
  if (result < 0) throw_errno("connect " + url, last_error);
  return result;
}

void put_u16(std::uint8_t* out, std::uint16_t value) {
  const std::uint16_t encoded = htons(value);
  std::memcpy(out, &encoded, sizeof(encoded));
}

void put_u32(std::uint8_t* out, std::uint32_t value) {
  const std::uint32_t encoded = htonl(value);
  std::memcpy(out, &encoded, sizeof(encoded));
}

void put_u64(std::uint8_t* out, std::uint64_t value) {
  put_u32(out, static_cast<std::uint32_t>(value >> 32));
  put_u32(out + 4, static_cast<std::uint32_t>(value));
}

std::uint16_t get_u16(const std::uint8_t* in) {
  std::uint16_t value;
  std::memcpy(&value, in, sizeof(value));
  return ntohs(value);
}

std::uint32_t get_u32(const std::uint8_t* in) {
  std::uint32_t value;
  std::memcpy(&value, in, sizeof(value));
  return ntohl(value);
}

std::uint64_t get_u64(const std::uint8_t* in) {
  return (static_cast<std::uint64_t>(get_u32(in)) << 32) | get_u32(in + 4);
}

struct DeviceState {
  std::uint64_t base_address = 0;
  std::uint64_t length = 0;
  std::string device_name;
  std::uint8_t port_num = 1;
  int gid_index = -1;
  std::uint32_t queue_depth = 0;
  std::uint32_t max_rd_atomic = 1;
  std::uint32_t max_dest_rd_atomic = 1;
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_mr* mr = nullptr;
  ibv_port_attr port_attr{};
  ibv_gid gid{};
  std::atomic<std::size_t> live_qps{0};

  ~DeviceState() {
    // Never deregister the L1 MR underneath a QP that a provider refused to
    // destroy. Leaking at process teardown is safer than permitting DMA into
    // released LMCache memory.
    if (live_qps.load() != 0) return;
    if (mr != nullptr) ibv_dereg_mr(mr);
    if (pd != nullptr) ibv_dealloc_pd(pd);
    if (context != nullptr) ibv_close_device(context);
  }
};

bool gid_matches_ipv4(const ibv_gid& gid, const in_addr& address) {
  static constexpr std::array<std::uint8_t, 12> prefix = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
  return std::memcmp(gid.raw, prefix.data(), prefix.size()) == 0 &&
         std::memcmp(gid.raw + 12, &address, sizeof(address)) == 0;
}

std::shared_ptr<DeviceState> open_device(std::uint64_t base_address,
                                         std::uint64_t length,
                                         const std::string& advertise_url,
                                         const std::string& requested_device,
                                         std::uint8_t port_num,
                                         int requested_gid_index,
                                         std::uint32_t requested_queue_depth) {
  if (base_address == 0 || length == 0) {
    throw std::invalid_argument("RDMA L1 base and length must be nonzero");
  }
  if (length > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("RDMA L1 length exceeds size_t");
  }
  if (base_address > std::numeric_limits<std::uint64_t>::max() - length) {
    throw std::overflow_error("RDMA L1 address range overflow");
  }
  if (port_num == 0) throw std::invalid_argument("RDMA port must be >= 1");
  if (requested_queue_depth == 0) {
    throw std::invalid_argument("RDMA queue depth must be >= 1");
  }

  int device_count = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr) throw_errno("ibv_get_device_list");
  ibv_device* selected = nullptr;
  for (int i = 0; i < device_count; ++i) {
    if (requested_device.empty() ||
        requested_device == ibv_get_device_name(devices[i])) {
      selected = devices[i];
      if (!requested_device.empty()) break;
    }
  }
  if (selected == nullptr) {
    ibv_free_device_list(devices);
    throw std::runtime_error("RDMA device not found: " + requested_device);
  }

  auto state = std::make_shared<DeviceState>();
  state->device_name = ibv_get_device_name(selected);
  state->context = ibv_open_device(selected);
  ibv_free_device_list(devices);
  if (state->context == nullptr) throw_errno("ibv_open_device");

  ibv_device_attr device_attr{};
  if (ibv_query_device(state->context, &device_attr) != 0) {
    throw_errno("ibv_query_device");
  }
  if (requested_queue_depth >
      static_cast<std::uint32_t>(device_attr.max_qp_wr)) {
    throw std::invalid_argument("RDMA queue depth exceeds device max_qp_wr");
  }
  if (device_attr.max_qp_rd_atom == 0 || device_attr.max_qp_init_rd_atom == 0) {
    throw std::runtime_error(
        "RDMA device does not support RC RDMA READ initiator/responder "
        "credits");
  }
  // max_rd_atomic is the initiator depth programmed at RTS and is bounded by
  // max_qp_init_rd_atom.  max_dest_rd_atomic is the responder depth programmed
  // at RTR and is bounded by max_qp_rd_atom.
  state->max_rd_atomic =
      std::min<std::uint32_t>(16, device_attr.max_qp_init_rd_atom);
  state->max_dest_rd_atomic =
      std::min<std::uint32_t>(16, device_attr.max_qp_rd_atom);
  state->queue_depth = requested_queue_depth;
  state->port_num = port_num;
  if (ibv_query_port(state->context, port_num, &state->port_attr) != 0) {
    throw_errno("ibv_query_port");
  }
  if (state->port_attr.state != IBV_PORT_ACTIVE) {
    throw std::runtime_error("RDMA port is not active");
  }

  int gid_index = requested_gid_index;
  if (gid_index < 0) {
    const auto host = parse_url(advertise_url).host;
    in_addr address{};
    if (inet_pton(AF_INET, host.c_str(), &address) != 1) {
      throw std::invalid_argument(
          "automatic GID selection requires an IPv4 advertise address");
    }
    int fallback = -1;
    for (int index = 0; index < state->port_attr.gid_tbl_len; ++index) {
      ibv_gid_entry entry{};
      if (ibv_query_gid_ex(state->context, port_num, index, &entry, 0) != 0) {
        continue;
      }
      if (!gid_matches_ipv4(entry.gid, address)) continue;
      if (fallback < 0) fallback = index;
      if (entry.gid_type == IBV_GID_TYPE_ROCE_V2) {
        gid_index = index;
        break;
      }
    }
    if (gid_index < 0) gid_index = fallback;
    if (gid_index < 0) {
      throw std::runtime_error("no GID matches advertised IPv4 address " +
                               host);
    }
  }
  if (gid_index >= state->port_attr.gid_tbl_len) {
    throw std::invalid_argument("RDMA GID index is outside the port GID table");
  }
  if (ibv_query_gid(state->context, port_num, gid_index, &state->gid) != 0) {
    throw_errno("ibv_query_gid");
  }
  state->gid_index = gid_index;

  state->pd = ibv_alloc_pd(state->context);
  if (state->pd == nullptr) throw_errno("ibv_alloc_pd");
  state->base_address = base_address;
  state->length = length;
  state->mr = ibv_reg_mr(
      state->pd,
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(base_address)),
      static_cast<std::size_t>(length),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
  if (state->mr == nullptr) throw_errno("ibv_reg_mr(L1)");
  return state;
}

struct PeerHello {
  std::uint32_t qpn = 0;
  std::uint32_t psn = 0;
  std::uint32_t rkey = 0;
  std::uint16_t lid = 0;
  ibv_mtu mtu = IBV_MTU_1024;
  std::uint8_t max_dest_rd_atomic = 0;
  std::uint64_t base_address = 0;
  std::uint64_t length = 0;
  ibv_gid gid{};
};

std::array<std::uint8_t, kHelloBytes> encode_hello(const PeerHello& hello) {
  std::array<std::uint8_t, kHelloBytes> out{};
  std::copy(kMagic.begin(), kMagic.end(), out.begin());
  put_u32(out.data() + 8, kProtocolVersion);
  put_u32(out.data() + 12, hello.qpn);
  put_u32(out.data() + 16, hello.psn);
  put_u32(out.data() + 20, hello.rkey);
  put_u16(out.data() + 24, hello.lid);
  out[26] = static_cast<std::uint8_t>(hello.mtu);
  out[27] = hello.max_dest_rd_atomic;
  put_u64(out.data() + 28, hello.base_address);
  put_u64(out.data() + 36, hello.length);
  std::memcpy(out.data() + 44, hello.gid.raw, 16);
  return out;
}

PeerHello decode_hello(const std::array<std::uint8_t, kHelloBytes>& in) {
  if (!std::equal(kMagic.begin(), kMagic.end(), in.begin())) {
    throw std::runtime_error("RDMA control peer sent an invalid magic value");
  }
  if (get_u32(in.data() + 8) != kProtocolVersion) {
    throw std::runtime_error("RDMA control protocol version mismatch");
  }
  PeerHello hello;
  hello.qpn = get_u32(in.data() + 12);
  hello.psn = get_u32(in.data() + 16);
  hello.rkey = get_u32(in.data() + 20);
  hello.lid = get_u16(in.data() + 24);
  const auto mtu = in[26];
  if (mtu < IBV_MTU_256 || mtu > IBV_MTU_4096) {
    throw std::runtime_error("RDMA peer advertised an invalid path MTU");
  }
  hello.mtu = static_cast<ibv_mtu>(mtu);
  hello.max_dest_rd_atomic = in[27];
  hello.base_address = get_u64(in.data() + 28);
  hello.length = get_u64(in.data() + 36);
  std::memcpy(hello.gid.raw, in.data() + 44, 16);
  if (hello.qpn == 0 || hello.rkey == 0 || hello.max_dest_rd_atomic == 0 ||
      hello.base_address == 0 || hello.length == 0) {
    throw std::runtime_error(
        "RDMA peer advertised an incomplete memory/QP tuple");
  }
  if (hello.base_address >
      std::numeric_limits<std::uint64_t>::max() - hello.length) {
    throw std::runtime_error(
        "RDMA peer advertised an overflowing memory range");
  }
  return hello;
}

std::uint32_t random_psn() {
  std::random_device source;
  return source() & 0x00ffffffU;
}

struct Endpoint {
  std::shared_ptr<DeviceState> device;
  int control_fd = -1;
  ibv_comp_channel* completion_channel = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  std::uint32_t local_psn = 0;
  PeerHello remote{};

  ~Endpoint() noexcept { close_noexcept(); }

  void destroy_qp() {
    if (qp == nullptr) return;

    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_ERR;
    // Destroying the QP is the quiescence proof.  A provider may reject the
    // best-effort ERR transition while still successfully destroying the QP.
    ibv_modify_qp(qp, &attr, IBV_QP_STATE);
    const int rc = ibv_destroy_qp(qp);
    if (rc != 0) {
      // Keep the handle rooted so an explicit close can retry.  In particular,
      // do not destroy the CQ or release any memory registrations while the QP
      // may still DMA into them.
      throw_errno("ibv_destroy_qp", rc > 0 ? rc : errno);
    }
    qp = nullptr;
    --device->live_qps;
  }

  void close() {
    destroy_qp();
    if (cq != nullptr) {
      const int rc = ibv_destroy_cq(cq);
      if (rc != 0) {
        throw_errno("ibv_destroy_cq", rc > 0 ? rc : errno);
      }
      cq = nullptr;
    }
    if (completion_channel != nullptr) {
      const int rc = ibv_destroy_comp_channel(completion_channel);
      if (rc != 0) {
        throw_errno("ibv_destroy_comp_channel", rc > 0 ? rc : errno);
      }
      completion_channel = nullptr;
    }
    if (control_fd >= 0) {
      ::close(control_fd);
      control_fd = -1;
    }
  }

  void close_noexcept() noexcept {
    try {
      close();
    } catch (...) {
      // Explicit owners retain this Endpoint and surface/retry close failures.
      // A destructor cannot throw; leave provider resources leaked at process
      // teardown instead of pretending the QP was quiesced.
    }
  }
};

std::shared_ptr<Endpoint> create_endpoint(
    const std::shared_ptr<DeviceState>& device, int control_fd,
    bool with_completion_channel) {
  std::shared_ptr<Endpoint> endpoint;
  try {
    endpoint = std::make_shared<Endpoint>();
    endpoint->device = device;
    endpoint->control_fd = control_fd;
    endpoint->local_psn = random_psn();
    if (with_completion_channel) {
      endpoint->completion_channel = ibv_create_comp_channel(device->context);
      if (endpoint->completion_channel == nullptr) {
        throw_errno("ibv_create_comp_channel");
      }
      set_nonblocking(endpoint->completion_channel->fd, true);
    }
    const std::uint32_t endpoint_depth =
        with_completion_channel ? device->queue_depth : 1;
    endpoint->cq =
        ibv_create_cq(device->context, static_cast<int>(endpoint_depth),
                      nullptr, endpoint->completion_channel, 0);
    if (endpoint->cq == nullptr) throw_errno("ibv_create_cq");

    ibv_qp_init_attr init{};
    init.send_cq = endpoint->cq;
    init.recv_cq = endpoint->cq;
    init.qp_type = IBV_QPT_RC;
    init.sq_sig_all = 0;
    init.cap.max_send_wr = endpoint_depth;
    init.cap.max_recv_wr = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    endpoint->qp = ibv_create_qp(device->pd, &init);
    if (endpoint->qp == nullptr) throw_errno("ibv_create_qp");
    ++device->live_qps;

    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = device->port_num;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ;
    check_zero(ibv_modify_qp(endpoint->qp, &attr,
                             IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                 IBV_QP_ACCESS_FLAGS),
               "ibv_modify_qp(INIT)");
    return endpoint;
  } catch (...) {
    if (endpoint == nullptr) ::close(control_fd);
    throw;
  }
}

PeerHello local_hello(const Endpoint& endpoint) {
  PeerHello hello;
  hello.qpn = endpoint.qp->qp_num;
  hello.psn = endpoint.local_psn;
  hello.rkey = endpoint.device->mr->rkey;
  hello.lid = endpoint.device->port_attr.lid;
  hello.mtu = endpoint.device->port_attr.active_mtu;
  hello.max_dest_rd_atomic =
      static_cast<std::uint8_t>(endpoint.device->max_dest_rd_atomic);
  hello.base_address = endpoint.device->base_address;
  hello.length = endpoint.device->length;
  hello.gid = endpoint.device->gid;
  return hello;
}

void connect_qp(Endpoint& endpoint, const PeerHello& remote) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = static_cast<ibv_mtu>(
      std::min(static_cast<int>(endpoint.device->port_attr.active_mtu),
               static_cast<int>(remote.mtu)));
  attr.dest_qp_num = remote.qpn;
  attr.rq_psn = remote.psn;
  attr.max_dest_rd_atomic = endpoint.device->max_dest_rd_atomic;
  attr.min_rnr_timer = 12;
  attr.ah_attr.port_num = endpoint.device->port_num;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  if (endpoint.device->port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = remote.gid;
    attr.ah_attr.grh.sgid_index = endpoint.device->gid_index;
    attr.ah_attr.grh.hop_limit = 64;
  } else {
    attr.ah_attr.dlid = remote.lid;
  }
  check_zero(
      ibv_modify_qp(endpoint.qp, &attr,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
      "ibv_modify_qp(RTR)");

  attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = endpoint.local_psn;
  // RC initiator depth must not exceed the responder resources programmed by
  // the peer. Identical HCAs often hide this requirement, so negotiate it in
  // the wire hello for heterogeneous deployments.
  attr.max_rd_atomic =
      std::min(endpoint.device->max_rd_atomic,
               static_cast<std::uint32_t>(remote.max_dest_rd_atomic));
  check_zero(ibv_modify_qp(endpoint.qp, &attr,
                           IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                               IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                               IBV_QP_MAX_QP_RD_ATOMIC),
             "ibv_modify_qp(RTS)");
  endpoint.remote = remote;
}

void active_handshake(Endpoint& endpoint, ControlDeadline deadline) {
  const auto local = encode_hello(local_hello(endpoint));
  send_all(endpoint.control_fd, local.data(), local.size(), deadline);
  std::array<std::uint8_t, kHelloBytes> wire{};
  recv_all(endpoint.control_fd, wire.data(), wire.size(), deadline);
  connect_qp(endpoint, decode_hello(wire));
  send_all(endpoint.control_fd, &kReadyByte, 1, deadline);
  std::uint8_t ready = 0;
  recv_all(endpoint.control_fd, &ready, 1, deadline);
  if (ready != kReadyByte) {
    throw std::runtime_error("RDMA peer did not complete the QP handshake");
  }
}

void passive_handshake(Endpoint& endpoint, ControlDeadline deadline) {
  std::array<std::uint8_t, kHelloBytes> wire{};
  recv_all(endpoint.control_fd, wire.data(), wire.size(), deadline);
  const PeerHello remote = decode_hello(wire);
  const auto local = encode_hello(local_hello(endpoint));
  send_all(endpoint.control_fd, local.data(), local.size(), deadline);
  connect_qp(endpoint, remote);
  std::uint8_t ready = 0;
  recv_all(endpoint.control_fd, &ready, 1, deadline);
  if (ready != kReadyByte) {
    throw std::runtime_error("RDMA peer did not complete the QP handshake");
  }
  send_all(endpoint.control_fd, &kReadyByte, 1, deadline);
}

void signal_event_fd(int event_fd) {
  if (event_fd < 0) return;
  const std::uint64_t one = 1;
  ssize_t rc;
  do {
    rc = write(event_fd, &one, sizeof(one));
  } while (rc < 0 && errno == EINTR);
  if (rc < 0 && errno != EAGAIN && errno != EBADF) {
    // A concurrent close can make the shutdown wakeup best-effort.
  }
}

}  // namespace

class RdmaClient::Impl {
 public:
  explicit Impl(std::shared_ptr<Endpoint> endpoint)
      : endpoint_(std::move(endpoint)), stop_fd_(eventfd(0, EFD_CLOEXEC)) {
    if (stop_fd_ < 0) throw_errno("eventfd(RDMA completion stop)");
    if (ibv_req_notify_cq(endpoint_->cq, 0) != 0) {
      ::close(stop_fd_);
      stop_fd_ = -1;
      throw_errno("ibv_req_notify_cq");
    }
    try {
      completion_thread_ = std::thread([this] { completion_loop(); });
    } catch (...) {
      ::close(stop_fd_);
      stop_fd_ = -1;
      throw;
    }
  }

  ~Impl() noexcept { close_noexcept(); }

  std::uint64_t submit_read(const std::vector<std::uint64_t>& local_offsets,
                            const std::vector<std::uint64_t>& remote_offsets,
                            const std::vector<std::uint32_t>& sizes) {
    if (local_offsets.empty()) {
      throw std::invalid_argument("RDMA read batch must not be empty");
    }
    if (local_offsets.size() != remote_offsets.size() ||
        local_offsets.size() != sizes.size()) {
      throw std::invalid_argument(
          "local offsets, remote offsets, and sizes must have equal lengths");
    }
    if (local_offsets.size() > endpoint_->device->queue_depth) {
      throw std::invalid_argument("RDMA read batch exceeds queue depth");
    }

    std::vector<ibv_sge> sges(sizes.size());
    std::vector<ibv_send_wr> wrs(sizes.size());
    std::lock_guard<std::mutex> guard(mutex_);
    if (closing_ || endpoint_->qp == nullptr) {
      throw std::runtime_error("RDMA client is closed");
    }
    if (!healthy_) throw std::runtime_error("RDMA QP is unhealthy");
    if (outstanding_wrs_ + sizes.size() > endpoint_->device->queue_depth) {
      throw std::runtime_error("RDMA send queue is full");
    }

    const std::uint64_t local_base_address = endpoint_->device->base_address;
    const std::uint64_t local_length = endpoint_->device->length;
    const std::uint32_t local_lkey = endpoint_->device->mr->lkey;
    std::uint64_t bytes = 0;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      const auto size = static_cast<std::uint64_t>(sizes[i]);
      if (size > endpoint_->device->port_attr.max_msg_sz) {
        throw std::invalid_argument(
            "RDMA read object exceeds the active port max_msg_sz");
      }
      if (size == 0 || local_offsets[i] > local_length ||
          size > local_length - local_offsets[i]) {
        throw std::invalid_argument(
            "RDMA local address is outside registered L1");
      }
      if (remote_offsets[i] > endpoint_->remote.length ||
          size > endpoint_->remote.length - remote_offsets[i]) {
        throw std::invalid_argument("RDMA remote address is outside peer L1");
      }
      if (bytes > std::numeric_limits<std::uint64_t>::max() - size) {
        throw std::overflow_error("RDMA read byte count overflow");
      }
      bytes += size;
    }

    if (next_task_id_ == std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("RDMA read task id space exhausted");
    }
    const std::uint64_t task_id = next_task_id_++;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      sges[i].addr = local_base_address + local_offsets[i];
      sges[i].length = sizes[i];
      sges[i].lkey = local_lkey;
      wrs[i].wr_id = task_id;
      wrs[i].sg_list = &sges[i];
      wrs[i].num_sge = 1;
      wrs[i].opcode = IBV_WR_RDMA_READ;
      wrs[i].send_flags = i + 1 == sizes.size() ? IBV_SEND_SIGNALED : 0;
      wrs[i].wr.rdma.remote_addr =
          endpoint_->remote.base_address + remote_offsets[i];
      wrs[i].wr.rdma.rkey = endpoint_->remote.rkey;
      wrs[i].next = i + 1 < sizes.size() ? &wrs[i + 1] : nullptr;
    }

    tasks_.emplace(task_id, Task{sizes.size(), sizes.size(), false, false});
    outstanding_wrs_ += sizes.size();
    ibv_send_wr* bad_wr = nullptr;
    const int rc = ibv_post_send(endpoint_->qp, wrs.data(), &bad_wr);
    if (rc != 0) {
      // A provider may have posted the prefix before returning bad_wr. Moving
      // the QP to ERR and destroying it is the only safe point at which the
      // caller may discard destination buffers, so do that before returning.
      if (!fail_endpoint_locked()) {
        std::rethrow_exception(teardown_error_);
      }
      throw_errno("ibv_post_send(RDMA READ)", rc);
    }
    ++submitted_reads_;
    submitted_bytes_ += bytes;
    return task_id;
  }

  std::tuple<bool, bool, std::size_t> query(std::uint64_t task_id) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end())
      throw std::out_of_range("unknown RDMA read task id");
    if (!found->second.finished && teardown_error_ != nullptr) {
      std::rethrow_exception(teardown_error_);
    }
    if (!found->second.finished) {
      return {false, false, found->second.object_count};
    }
    const bool succeeded = found->second.succeeded;
    const std::size_t count = found->second.object_count;
    tasks_.erase(found);
    return {true, succeeded, count};
  }

  void close() {
    {
      std::unique_lock<std::mutex> guard(mutex_);
      completion_cv_.wait(guard, [this] { return !closing_ || closed_; });
      if (closed_) return;
      closing_ = true;
      if (endpoint_->control_fd >= 0) {
        send(endpoint_->control_fd, &kCloseByte, 1, MSG_NOSIGNAL);
        shutdown(endpoint_->control_fd, SHUT_RDWR);
      }
      // Quiesce the QP while the CQ consumer is still alive. Only after QP
      // destruction succeeds is it safe for LMCache to reuse destination
      // buffers or deregister the backing L1 memory.
      if (!fail_endpoint_locked()) {
        const auto error = teardown_error_;
        closing_ = false;
        completion_cv_.notify_all();
        guard.unlock();
        std::rethrow_exception(error);
      }
    }

    if (stop_fd_ >= 0) signal_event_fd(stop_fd_);
    if (completion_thread_.joinable() &&
        completion_thread_.get_id() != std::this_thread::get_id()) {
      completion_thread_.join();
    }

    std::lock_guard<std::mutex> guard(mutex_);
    try {
      // The completion thread acknowledges every event it retrieves. With the
      // QP gone there can be no new CQEs, so consume anything it did not see
      // before destroying the CQ and completion channel.
      acknowledge_pending_cq_events();
      if (endpoint_->cq != nullptr) drain_cq_locked();
      endpoint_->close();
    } catch (...) {
      healthy_ = false;
      teardown_error_ = std::current_exception();
      closing_ = false;
      completion_cv_.notify_all();
      throw;
    }
    for (auto& [_, task] : tasks_) {
      if (!task.finished) {
        task.finished = true;
        task.succeeded = false;
        ++failed_reads_;
      }
    }
    outstanding_wrs_ = 0;
    healthy_ = false;
    if (stop_fd_ >= 0) {
      ::close(stop_fd_);
      stop_fd_ = -1;
    }
    closed_ = true;
    closing_ = false;
    teardown_error_ = nullptr;
    completion_cv_.notify_all();
  }

  void close_noexcept() noexcept {
    try {
      close();
    } catch (...) {
      // The explicit Python/C++ close path reports the teardown failure and
      // keeps this Impl rooted for retry. Destruction itself must not throw or
      // leave a joinable std::thread (which would call std::terminate).
      if (stop_fd_ >= 0) signal_event_fd(stop_fd_);
      if (completion_thread_.joinable() &&
          completion_thread_.get_id() != std::this_thread::get_id()) {
        completion_thread_.join();
      }
    }
  }

  bool healthy() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return healthy_ && !closing_;
  }

  std::size_t outstanding_tasks() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return std::count_if(tasks_.begin(), tasks_.end(), [](const auto& item) {
      return !item.second.finished;
    });
  }

  std::uint64_t submitted_reads() const { return submitted_reads_.load(); }
  std::uint64_t submitted_bytes() const { return submitted_bytes_.load(); }
  std::uint64_t completed_reads() const { return completed_reads_.load(); }
  std::uint64_t failed_reads() const { return failed_reads_.load(); }

  WcFailureDiagnostics wc_failure_diagnostics() const {
    std::lock_guard<std::mutex> guard(mutex_);
    WcFailureDiagnostics diagnostics;
    diagnostics.total = wc_failure_total_;
    diagnostics.last = last_wc_failure_;
    for (std::size_t slot = 0; slot < wc_failure_counts_.size(); ++slot) {
      if (wc_failure_counts_[slot] == 0) continue;
      const int status =
          slot == kUnknownWcStatusSlot ? -1 : static_cast<int>(slot);
      diagnostics.counts.push_back(
          {status, wc_status_name(status), wc_failure_counts_[slot]});
    }
    return diagnostics;
  }

 private:
  struct Task {
    std::size_t object_count;
    std::size_t wr_count;
    bool finished;
    bool succeeded;
  };

  void record_wc_failure_locked(const ibv_wc& completion) {
    const int status = static_cast<int>(completion.status);
    ++wc_failure_counts_[wc_status_slot(status)];
    ++wc_failure_total_;

    WcFailure failure;
    failure.status = status;
    failure.status_name = wc_status_name(status);
    failure.vendor_err = completion.vendor_err;
    failure.wr_id = completion.wr_id;
    failure.qp_num = endpoint_->qp == nullptr ? 0 : endpoint_->qp->qp_num;
    failure.device_name = endpoint_->device->device_name;
    failure.port_num = endpoint_->device->port_num;
    failure.gid_index = endpoint_->device->gid_index;
    last_wc_failure_ = std::move(failure);
  }

  bool fail_endpoint_locked() noexcept {
    healthy_ = false;
    try {
      endpoint_->destroy_qp();
    } catch (...) {
      teardown_error_ = std::current_exception();
      completion_cv_.notify_all();
      return false;
    }
    for (auto& [_, task] : tasks_) {
      if (!task.finished) {
        task.finished = true;
        task.succeeded = false;
        ++failed_reads_;
      }
    }
    outstanding_wrs_ = 0;
    completion_cv_.notify_all();
    teardown_error_ = nullptr;
    return true;
  }

  void drain_cq_locked() {
    std::array<ibv_wc, 32> completions{};
    bool terminal = false;
    while (true) {
      const int count =
          ibv_poll_cq(endpoint_->cq, completions.size(), completions.data());
      if (count < 0) {
        fail_endpoint_locked();
        return;
      }
      if (count == 0) break;
      for (int i = 0; i < count; ++i) {
        // A failed WC does not prove that earlier unsignaled WRs have stopped
        // touching their destination buffers.  Move the QP to ERR and destroy
        // it before publishing any task as terminal so LMCache can safely
        // recycle those buffers after query_read_status() returns.
        if (completions[i].status != IBV_WC_SUCCESS) {
          record_wc_failure_locked(completions[i]);
          fail_endpoint_locked();
          return;
        }
        const auto found = tasks_.find(completions[i].wr_id);
        if (found == tasks_.end() || found->second.finished) continue;
        outstanding_wrs_ -= std::min(outstanding_wrs_, found->second.wr_count);
        found->second.finished = true;
        found->second.succeeded = true;
        terminal = true;
        ++completed_reads_;
      }
    }
    if (terminal) {
      completion_cv_.notify_all();
    }
  }

  void acknowledge_pending_cq_events() {
    if (endpoint_->completion_channel == nullptr) return;
    while (true) {
      ibv_cq* event_cq = nullptr;
      void* event_context = nullptr;
      if (ibv_get_cq_event(endpoint_->completion_channel, &event_cq,
                           &event_context) == 0) {
        ibv_ack_cq_events(event_cq, 1);
        continue;
      }
      if (errno == EINTR) continue;
      if (errno == EAGAIN) return;
      throw_errno("ibv_get_cq_event during RDMA close");
    }
  }

  void completion_loop() {
    pollfd fds[2] = {{endpoint_->completion_channel->fd, POLLIN, 0},
                     {stop_fd_, POLLIN, 0}};
    while (true) {
      int rc;
      do {
        rc = poll(fds, 2, -1);
      } while (rc < 0 && errno == EINTR);
      if (rc < 0) {
        std::lock_guard<std::mutex> guard(mutex_);
        fail_endpoint_locked();
        return;
      }
      if (fds[1].revents & POLLIN) return;
      if (!(fds[0].revents & (POLLIN | POLLERR | POLLHUP))) continue;

      ibv_cq* event_cq = nullptr;
      void* event_context = nullptr;
      while (ibv_get_cq_event(endpoint_->completion_channel, &event_cq,
                              &event_context) == 0) {
        ibv_ack_cq_events(event_cq, 1);
      }
      if (errno != EAGAIN && errno != EINTR) {
        std::lock_guard<std::mutex> guard(mutex_);
        fail_endpoint_locked();
        return;
      }
      if (ibv_req_notify_cq(endpoint_->cq, 0) != 0) {
        std::lock_guard<std::mutex> guard(mutex_);
        fail_endpoint_locked();
        return;
      }
      std::lock_guard<std::mutex> guard(mutex_);
      drain_cq_locked();
      if (!healthy_) return;
    }
  }

  std::shared_ptr<Endpoint> endpoint_;
  mutable std::mutex mutex_;
  std::condition_variable completion_cv_;
  std::unordered_map<std::uint64_t, Task> tasks_;
  std::thread completion_thread_;
  int stop_fd_ = -1;
  std::uint64_t next_task_id_ = 1;
  std::size_t outstanding_wrs_ = 0;
  bool healthy_ = true;
  bool closing_ = false;
  bool closed_ = false;
  std::exception_ptr teardown_error_;
  std::atomic<std::uint64_t> submitted_reads_{0};
  std::atomic<std::uint64_t> submitted_bytes_{0};
  std::atomic<std::uint64_t> completed_reads_{0};
  std::atomic<std::uint64_t> failed_reads_{0};
  std::array<std::uint64_t, kWcStatusSlots> wc_failure_counts_{};
  std::uint64_t wc_failure_total_ = 0;
  std::optional<WcFailure> last_wc_failure_;
};

class RdmaContext::Impl
    : public std::enable_shared_from_this<RdmaContext::Impl> {
 public:
  Impl(std::uint64_t base_address, std::uint64_t length, std::string listen_url,
       std::string advertise_url, std::string device_name,
       std::uint8_t port_num, int gid_index, std::uint32_t queue_depth,
       std::uint32_t handshake_timeout_ms)
      : listen_url_(std::move(listen_url)),
        advertise_url_(std::move(advertise_url)),
        handshake_timeout_ms_(handshake_timeout_ms) {
    if (handshake_timeout_ms == 0) {
      throw std::invalid_argument("RDMA handshake timeout must be >= 1 ms");
    }
    if (handshake_timeout_ms >
        static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("RDMA handshake timeout exceeds INT_MAX ms");
    }
    stop_fd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (stop_fd_ < 0) throw_errno("eventfd(RDMA server stop)");
    try {
      device_ = open_device(base_address, length, advertise_url_, device_name,
                            port_num, gid_index, queue_depth);
      listen_fd_ = create_listener(listen_url_);
      server_thread_ = std::thread([this] { server_loop(); });
    } catch (...) {
      if (listen_fd_ >= 0) ::close(listen_fd_);
      listen_fd_ = -1;
      ::close(stop_fd_);
      stop_fd_ = -1;
      throw;
    }
  }

  ~Impl() noexcept {
    try {
      close();
    } catch (...) {
    }
  }

  std::shared_ptr<RdmaClient> connect(const std::string& peer_url) {
    std::shared_ptr<DeviceState> device;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (closing_) throw std::runtime_error("RDMA context is closing");
      device = device_;
      ++active_connects_;
    }
    bool connect_active = true;
    const auto finish_connect = [this, &connect_active] {
      if (!connect_active) return;
      std::lock_guard<std::mutex> guard(mutex_);
      --active_connects_;
      connect_active = false;
      close_cv_.notify_all();
    };
    const auto deadline = control_deadline(handshake_timeout_ms_);
    int fd = -1;
    bool fd_registered = false;
    std::shared_ptr<Endpoint> endpoint;
    std::shared_ptr<RdmaClient::Impl> impl;
    try {
      fd = connect_with_timeout(
          peer_url, handshake_timeout_ms_, deadline, stop_fd_,
          [this](int socket_fd) { register_outbound_socket(socket_fd); },
          [this](int socket_fd) { unregister_outbound_socket(socket_fd); });
      fd_registered = true;
      endpoint = create_endpoint(device, fd, true);
      active_handshake(*endpoint, deadline);
      impl = std::make_shared<RdmaClient::Impl>(endpoint);

      bool context_closing = false;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        unregister_outbound_socket_locked(fd);
        fd_registered = false;
        context_closing = closing_;
        if (!context_closing) {
          clients_.erase(
              std::remove_if(clients_.begin(), clients_.end(),
                             [](const auto& weak) { return weak.expired(); }),
              clients_.end());
          clients_.push_back(impl);
        }
      }
      if (context_closing) {
        impl->close();
        throw std::runtime_error("RDMA context closed during peer handshake");
      }
      auto result = std::make_shared<RdmaClient>(std::move(impl));
      endpoint.reset();
      finish_connect();
      return result;
    } catch (...) {
      if (fd_registered) unregister_outbound_socket(fd);
      impl.reset();
      endpoint.reset();
      finish_connect();
      throw;
    }
  }

  void close() {
    std::vector<std::shared_ptr<RdmaClient::Impl>> clients;
    {
      std::unique_lock<std::mutex> guard(mutex_);
      close_cv_.wait(guard, [this] { return !close_in_progress_; });
      if (closed_) return;
      close_in_progress_ = true;
      closing_ = true;
      if (server_handshake_cancel_fd_ >= 0) {
        shutdown(server_handshake_cancel_fd_, SHUT_RDWR);
      }
      for (const auto& [_, cancel_fd] : outbound_sockets_) {
        shutdown(cancel_fd, SHUT_RDWR);
      }
    }

    try {
      // Stop and join the listener before any fallible endpoint cleanup. A
      // destructor must never reach the std::thread member while it is
      // joinable.
      signal_event_fd(stop_fd_);
      if (server_thread_.joinable()) server_thread_.join();

      {
        std::unique_lock<std::mutex> guard(mutex_);
        close_cv_.wait(guard, [this] { return active_connects_ == 0; });
        clients.reserve(clients_.size() + retry_clients_.size());
        for (const auto& weak : clients_) {
          if (auto client = weak.lock()) clients.push_back(std::move(client));
        }
        for (const auto& client : retry_clients_) {
          if (std::find(clients.begin(), clients.end(), client) ==
              clients.end()) {
            clients.push_back(client);
          }
        }
      }

      std::exception_ptr first_error;
      std::vector<std::shared_ptr<RdmaClient::Impl>> retry_clients;
      retry_clients.reserve(clients.size());
      for (const auto& client : clients) {
        try {
          client->close();
        } catch (...) {
          if (first_error == nullptr) first_error = std::current_exception();
          retry_clients.push_back(client);
        }
      }
      for (auto& endpoint : inbound_) {
        try {
          endpoint->close();
        } catch (...) {
          if (first_error == nullptr) first_error = std::current_exception();
        }
      }

      {
        std::lock_guard<std::mutex> guard(mutex_);
        retry_clients_ = std::move(retry_clients);
        close_in_progress_ = false;
        if (first_error == nullptr) {
          inbound_.clear();
          for (const auto& [_, cancel_fd] : outbound_sockets_) {
            ::close(cancel_fd);
          }
          outbound_sockets_.clear();
          if (listen_fd_ >= 0) {
            ::close(listen_fd_);
            listen_fd_ = -1;
          }
          if (stop_fd_ >= 0) {
            ::close(stop_fd_);
            stop_fd_ = -1;
          }
          device_.reset();
          clients_.clear();
          retry_clients_.clear();
          closed_ = true;
        }
      }
      close_cv_.notify_all();
      if (first_error != nullptr) std::rethrow_exception(first_error);
    } catch (...) {
      {
        std::lock_guard<std::mutex> guard(mutex_);
        close_in_progress_ = false;
      }
      close_cv_.notify_all();
      throw;
    }
  }
  std::string device_name() const { return require_device()->device_name; }
  int gid_index() const { return require_device()->gid_index; }
  std::uint8_t port_num() const { return require_device()->port_num; }
  std::uint32_t queue_depth() const { return require_device()->queue_depth; }
  std::uint64_t registered_bytes() const { return require_device()->length; }

  std::size_t inbound_connection_count() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return inbound_.size();
  }

 private:
  void register_outbound_socket(int fd) {
    const int cancel_fd = dup(fd);
    if (cancel_fd < 0) throw_errno("dup(RDMA control socket)");
    std::lock_guard<std::mutex> guard(mutex_);
    if (closing_) {
      ::close(cancel_fd);
      throw std::runtime_error("RDMA context is closing");
    }
    try {
      outbound_sockets_.emplace_back(fd, cancel_fd);
    } catch (...) {
      ::close(cancel_fd);
      throw;
    }
  }

  void unregister_outbound_socket_locked(int fd) noexcept {
    const auto found =
        std::find_if(outbound_sockets_.begin(), outbound_sockets_.end(),
                     [fd](const auto& item) { return item.first == fd; });
    if (found == outbound_sockets_.end()) return;
    ::close(found->second);
    outbound_sockets_.erase(found);
  }

  void unregister_outbound_socket(int fd) noexcept {
    try {
      std::lock_guard<std::mutex> guard(mutex_);
      unregister_outbound_socket_locked(fd);
    } catch (...) {
    }
  }

  std::shared_ptr<DeviceState> require_device() const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (device_ == nullptr) {
      throw std::runtime_error("RDMA context is closed");
    }
    return device_;
  }

  void accept_one() {
    sockaddr_storage address{};
    socklen_t length = sizeof(address);
    const int fd =
        accept(listen_fd_, reinterpret_cast<sockaddr*>(&address), &length);
    if (fd < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return;
      throw_errno("accept RDMA control connection");
    }
    const auto deadline = control_deadline(handshake_timeout_ms_);
    const int cancel_fd = dup(fd);
    if (cancel_fd < 0) {
      const int error = errno;
      ::close(fd);
      throw_errno("dup(accepted RDMA control socket)", error);
    }
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (closing_) {
        ::close(cancel_fd);
        ::close(fd);
        return;
      }
      server_handshake_cancel_fd_ = cancel_fd;
    }
    std::shared_ptr<Endpoint> endpoint;
    bool control_fd_transferred = false;
    try {
      set_socket_timeout(fd, handshake_timeout_ms_);
      control_fd_transferred = true;
      endpoint = create_endpoint(device_, fd, false);
      passive_handshake(*endpoint, deadline);
      {
        std::lock_guard<std::mutex> guard(mutex_);
        ::close(server_handshake_cancel_fd_);
        server_handshake_cancel_fd_ = -1;
      }
      set_nonblocking(fd, true);
      std::lock_guard<std::mutex> guard(mutex_);
      // Keep a completed handshake rooted during shutdown; context::close()
      // owns the retryable endpoint cleanup after the listener has joined.
      inbound_.push_back(std::move(endpoint));
    } catch (...) {
      {
        std::lock_guard<std::mutex> guard(mutex_);
        if (server_handshake_cancel_fd_ >= 0) {
          ::close(server_handshake_cancel_fd_);
          server_handshake_cancel_fd_ = -1;
        }
      }
      if (!control_fd_transferred) ::close(fd);
      // A malformed or timed-out handshake is isolated to this connection.
    }
  }

  void reap_closed_inbound(
      const std::vector<pollfd>& fds,
      const std::vector<std::shared_ptr<Endpoint>>& snapshot) {
    std::vector<std::shared_ptr<Endpoint>> closed;
    for (std::size_t i = 0; i < snapshot.size(); ++i) {
      const short events = fds[i + 2].revents;
      if (!(events & (POLLIN | POLLERR | POLLHUP | POLLNVAL))) continue;
      std::uint8_t message = 0;
      const ssize_t rc = recv(snapshot[i]->control_fd, &message, 1, 0);
      if (rc <= 0 || message == kCloseByte ||
          events & (POLLERR | POLLHUP | POLLNVAL)) {
        closed.push_back(snapshot[i]);
      }
    }
    if (closed.empty()) return;
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& endpoint : closed) {
      const auto found = std::find(inbound_.begin(), inbound_.end(), endpoint);
      if (found != inbound_.end()) {
        try {
          (*found)->close();
        } catch (...) {
          if ((*found)->control_fd >= 0) {
            ::close((*found)->control_fd);
            (*found)->control_fd = -1;
          }
          // Keep provider resources rooted for Context::close() to retry.
          continue;
        }
        inbound_.erase(found);
      }
    }
  }

  void server_loop() {
    while (true) {
      std::vector<std::shared_ptr<Endpoint>> snapshot;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        if (closing_) break;
        snapshot = inbound_;
      }
      std::vector<pollfd> fds;
      fds.reserve(snapshot.size() + 2);
      fds.push_back({listen_fd_, POLLIN, 0});
      fds.push_back({stop_fd_, POLLIN, 0});
      for (const auto& endpoint : snapshot) {
        fds.push_back({endpoint->control_fd,
                       static_cast<short>(POLLIN | POLLERR | POLLHUP), 0});
      }
      int rc;
      do {
        rc = poll(fds.data(), fds.size(), -1);
      } while (rc < 0 && errno == EINTR);
      if (rc < 0) break;
      if (fds[1].revents & POLLIN) break;
      if (fds[0].revents & POLLIN) {
        try {
          accept_one();
        } catch (...) {
          // Keep the listener alive; the next coordinator reconciliation may
          // establish a healthy connection without restarting the engine.
        }
      }
      try {
        reap_closed_inbound(fds, snapshot);
      } catch (...) {
        // Keep the inbound Endpoint rooted. Context::close() will retry and
        // surface a persistent QP-destroy failure without terminating this
        // noexcept-owned server thread.
      }
    }
  }

  std::string listen_url_;
  std::string advertise_url_;
  std::uint32_t handshake_timeout_ms_;
  std::shared_ptr<DeviceState> device_;
  int listen_fd_ = -1;
  int stop_fd_ = -1;
  int server_handshake_cancel_fd_ = -1;
  std::thread server_thread_;
  mutable std::mutex mutex_;
  std::condition_variable close_cv_;
  std::vector<std::pair<int, int>> outbound_sockets_;
  std::vector<std::shared_ptr<Endpoint>> inbound_;
  std::vector<std::weak_ptr<RdmaClient::Impl>> clients_;
  std::vector<std::shared_ptr<RdmaClient::Impl>> retry_clients_;
  std::size_t active_connects_ = 0;
  bool closing_ = false;
  bool close_in_progress_ = false;
  bool closed_ = false;
};

RdmaContext::RdmaContext(std::uint64_t base_address, std::uint64_t length,
                         std::string listen_url, std::string advertise_url,
                         std::string device_name, std::uint8_t port_num,
                         int gid_index, std::uint32_t queue_depth,
                         std::uint32_t handshake_timeout_ms)
    : impl_(std::make_shared<Impl>(base_address, length, std::move(listen_url),
                                   std::move(advertise_url),
                                   std::move(device_name), port_num, gid_index,
                                   queue_depth, handshake_timeout_ms)) {}

RdmaContext::~RdmaContext() {
  try {
    close();
  } catch (...) {
  }
}

std::shared_ptr<RdmaClient> RdmaContext::connect(const std::string& peer_url) {
  return impl_->connect(peer_url);
}

void RdmaContext::close() {
  if (impl_ != nullptr) impl_->close();
}

std::string RdmaContext::device_name() const { return impl_->device_name(); }
int RdmaContext::gid_index() const { return impl_->gid_index(); }
std::uint8_t RdmaContext::port_num() const { return impl_->port_num(); }
std::uint32_t RdmaContext::queue_depth() const { return impl_->queue_depth(); }
std::uint64_t RdmaContext::registered_bytes() const {
  return impl_->registered_bytes();
}
std::size_t RdmaContext::inbound_connection_count() const {
  return impl_->inbound_connection_count();
}

RdmaClient::RdmaClient(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
RdmaClient::~RdmaClient() {
  if (impl_ != nullptr) impl_->close_noexcept();
}

std::uint64_t RdmaClient::submit_read(
    const std::vector<std::uint64_t>& local_offsets,
    const std::vector<std::uint64_t>& remote_offsets,
    const std::vector<std::uint32_t>& sizes) {
  return impl_->submit_read(local_offsets, remote_offsets, sizes);
}

std::tuple<bool, bool, std::size_t> RdmaClient::query_read_status(
    std::uint64_t task_id) {
  return impl_->query(task_id);
}

void RdmaClient::close() {
  if (impl_ != nullptr) impl_->close();
}

bool RdmaClient::healthy() const { return impl_->healthy(); }
std::size_t RdmaClient::outstanding_tasks() const {
  return impl_->outstanding_tasks();
}
std::uint64_t RdmaClient::submitted_reads() const {
  return impl_->submitted_reads();
}
std::uint64_t RdmaClient::submitted_bytes() const {
  return impl_->submitted_bytes();
}
std::uint64_t RdmaClient::completed_reads() const {
  return impl_->completed_reads();
}
std::uint64_t RdmaClient::failed_reads() const { return impl_->failed_reads(); }
WcFailureDiagnostics RdmaClient::wc_failure_diagnostics() const {
  return impl_->wc_failure_diagnostics();
}

}  // namespace lmcache::rdma
