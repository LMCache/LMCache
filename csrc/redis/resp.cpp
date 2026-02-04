// SPDX-License-Identifier: Apache-2.0

#include "resp.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/eventfd.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

/*
There are two sources of overhead in a python integration:
1. on submission, we want to make sure that:
  i. the submitting python thread isn't blocked
  ii. we make as few submissions to the event loop as possible
  because we don't know when they will be scheduled

2. on the callback, we want to make sure that:
  i. we don't have to poll for completions
  ii. we have as few completions as possible because we don't
  know when the event loop will run our callbacks

Even if the RESP client is very fast, these sources of python overhead
can make the client very slow.

Therefore, we have:
1. eventfd allows us to make submission non-blocking (and no polling)
2. threading on the C++ layer (avoiding GIL) and tiling work among threads
3. generates a single future/completion per batch operation
*/

// tiling refers to dividing work for batched operations between threads
// beforehand
enum class Op : uint8_t { GET, SET, EXISTS, BATCH_TILE_GET, BATCH_TILE_SET };

// shared communication state between threads executing a single batch operation
// all threads need to complete before the completion is sent
struct BatchState {
  // the number of tiles left to finish
  // we can only send a completion when the last tile is finished
  std::atomic<uint32_t> remaining_tiles{0};
  std::atomic<bool> any_failed{false};

  std::mutex err_mu;
  std::string first_error;
};

struct Request {
  // the completion also has a future_id
  // the caller is responsible for matching the request to the completion
  uint64_t future_id = 0;
  Op op;
  std::string key;

  void* buf_ptr = nullptr;
  size_t buf_len = 0;

  // python passes in a memoryview or bytearray
  // following the buffer protocol
  py::object buf_owner;

  /*
  for tiling batched operations
  */

  // these will be the keys and buffers that this Request is responsible for
  std::vector<std::string> keys;
  std::vector<void*> buf_ptrs;
  std::vector<py::object> buf_owners;

  uint64_t batch_future_id = 0;
  // shared batch state between threads executing a single batch operation
  // so that they can coordinate when to send the completion
  std::shared_ptr<BatchState> batch;
};

struct Completion {
  // the request also has a future_id
  // the caller is responsible for matching the completion to the request
  uint64_t future_id = 0;

  // did the operation succeed?
  bool ok = true;
  // for EXISTS only (no result in the completion for SET and GET)
  bool result_bool = false;

  // error string if operation failed
  std::string error;
};

// a TCP session (one per thread) implementing RESP2
/*
Key Optimizations include:
1. preset chunk_size (allows not parsing for \r\n byte-by-byte)
2. scatter/gather sending of data (with pre-allocated buffers)
3. zero copy (no bounce bufferse)
*/
struct WorkerConn {
  int fd = -1;
  std::string host;
  int port;
  size_t chunk_size;

  // pre-computed headers
  std::string size_header;
  std::string get_prefix;
  std::string set_prefix;
  std::string exists_prefix;

  // pre-computed constants (for comparisons)
  const char* crlf = "\r\n";
  static constexpr size_t crlf_len = 2;

  const char* ok_response = "+OK\r\n";
  static constexpr size_t ok_response_len = 5;

  const char* exists_one = ":1\r\n";
  const char* exists_zero = ":0\r\n";
  static constexpr size_t exists_response_len = 4;

  WorkerConn() = default;

  void connect(const std::string& host, int port, size_t chunk_size) {
    this->host = host;
    this->port = port;
    this->chunk_size = chunk_size;

    // reusable headers (for scatter/gather)

    {
      std::ostringstream oss;
      oss << "$" << chunk_size << "\r\n";
      size_header = oss.str();
    }

    get_prefix = "*2\r\n$3\r\nGET\r\n";
    set_prefix = "*3\r\n$3\r\nSET\r\n";
    exists_prefix = "*2\r\n$6\r\nEXISTS\r\n";

    // 1. create socket
    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      throw std::runtime_error("failed to create socket");
    }

    // 2. resolve host
    struct addrinfo hints = {}, *result = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    int err = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);

    if (err != 0) {
      ::close(fd);
      throw std::runtime_error(std::string("getaddrinfo failed: ") +
                               gai_strerror(err));
    }

    // 3. connect to host
    if (::connect(fd, result->ai_addr, result->ai_addrlen) < 0) {
      freeaddrinfo(result);
      ::close(fd);
      throw std::runtime_error("connection failed");
    }

    freeaddrinfo(result);
  }

  ~WorkerConn() {
    if (fd >= 0) {
      ::close(fd);
    }
  }

  // call send multiple times until all the data is sent
  void send_all(const void* data, size_t len) {
    size_t sent_so_far = 0;
    const char* ptr = static_cast<const char*>(data);
    while (sent_so_far < len) {
      ssize_t n = ::send(fd, ptr + sent_so_far, len - sent_so_far, 0);
      if (n < 0) {
        if (errno == EINTR) {
          continue;  // retry on EINTR
        }
        throw std::runtime_error("socket send failed");
      }
      if (n == 0) {
        throw std::runtime_error("socket send failed: connection closed");
      }
      sent_so_far += n;
    }
  }

  // scatter gather send
  void send_multipart(
      const std::vector<std::pair<const void*, size_t>>& parts) {
    if (parts.empty()) return;

    // writev
    std::vector<struct iovec> iov;
    iov.reserve(parts.size());
    for (const auto& part : parts) {
      iov.push_back({const_cast<void*>(part.first), part.second});
    }

    size_t total_to_send = 0;
    for (const auto& part : parts) {
      total_to_send += part.second;
    }

    size_t sent_so_far = 0;
    size_t iov_idx = 0;

    while (sent_so_far < total_to_send) {
      ssize_t n = ::writev(fd, &iov[iov_idx], iov.size() - iov_idx);
      if (n < 0) {
        if (errno == EINTR) {
          continue;  // retry on EINTR
        }
        throw std::runtime_error("socket writev failed");
      }
      if (n == 0) {
        throw std::runtime_error("socket writev failed: connection closed");
      }

      sent_so_far += n;

      // adjust iovec for partial writes
      size_t remaining = n;
      while (remaining > 0 && iov_idx < iov.size()) {
        if (remaining >= iov[iov_idx].iov_len) {
          // this iovec was fully consumed
          remaining -= iov[iov_idx].iov_len;
          iov_idx++;
        } else {
          // this iovec was partially consumed
          iov[iov_idx].iov_base =
              static_cast<char*>(iov[iov_idx].iov_base) + remaining;
          iov[iov_idx].iov_len -= remaining;
          remaining = 0;
        }
      }
    }
  }

  void recv_exactly(void* buf, size_t len) {
    size_t recv_so_far = 0;
    char* ptr = static_cast<char*>(buf);
    while (recv_so_far < len) {
      ssize_t n = ::recv(fd, ptr + recv_so_far, len - recv_so_far, 0);
      if (n < 0) {
        if (errno == EINTR) {
          continue;  // retry on EINTR
        }
        throw std::runtime_error("socket recv failed");
      }
      if (n == 0) {
        throw std::runtime_error("socket recv failed: connection closed");
      }
      recv_so_far += n;
    }
  }

  std::string make_key_header(const std::string& key) {
    std::ostringstream oss;
    oss << "$" << key.size() << "\r\n" << key << "\r\n";
    return oss.str();
  }
};

/*
RESP SET, GET, EXISTS are very fragile here since we make hard assumptions about
the RESP responses. A single error could break our assumptions. We can mitigate
this by actually parsing the headers and trailers.
*/

// RESP GET
static void do_get_into(WorkerConn& conn, const std::string& key, void* buf,
                        size_t len) {
  // we only read exactly chunk_size bytes (save_unfull_chunk must be off)
  if (len != conn.chunk_size) {
    throw std::runtime_error("buffer size mismatch");
  }

  // build key header (can't pre-allocate)
  std::string key_header = conn.make_key_header(key);

  // send GET cmd
  // iovec let's us combine pre-built parts and dynamic strings
  conn.send_multipart({{conn.get_prefix.data(), conn.get_prefix.size()},
                       {key_header.data(), key_header.size()}});

  // parse in 3 steps

  // 1. parse the size header and validate
  std::vector<char> size_header_buf(conn.size_header.size());
  conn.recv_exactly(size_header_buf.data(), size_header_buf.size());
  if (std::memcmp(size_header_buf.data(), conn.size_header.data(),
                  conn.size_header.size()) != 0) {
    throw std::runtime_error("GET: size header mismatch");
  }

  // 2. parse the payload (KV Cache)
  conn.recv_exactly(buf, len);

  // 3. parse the trailer and validate
  char trailer[WorkerConn::crlf_len];
  conn.recv_exactly(trailer, WorkerConn::crlf_len);
  if (std::memcmp(trailer, conn.crlf, WorkerConn::crlf_len) != 0) {
    throw std::runtime_error("GET: trailer mismatch");
  }
}

// RESP SET
static void do_set_from(WorkerConn& conn, const std::string& key,
                        const void* buf, size_t len) {
  // we only write exactly chunk_size bytes (save_unfull_chunk must be off)
  if (len != conn.chunk_size) {
    throw std::runtime_error("buffer size mismatch");
  }

  // build key header (can't pre-allocate)
  std::string key_header = conn.make_key_header(key);

  // send GET cmd
  // iovec let's us combine pre-built parts and dynamic strings
  conn.send_multipart({{conn.set_prefix.data(), conn.set_prefix.size()},
                       {key_header.data(), key_header.size()},
                       {conn.size_header.data(), conn.size_header.size()},
                       {buf, len},
                       {conn.crlf, WorkerConn::crlf_len}});

  // parse response which should be exactly +OK\r\n
  char response[WorkerConn::ok_response_len];
  conn.recv_exactly(response, WorkerConn::ok_response_len);

  if (std::memcmp(response, conn.ok_response, WorkerConn::ok_response_len) !=
      0) {
    throw std::runtime_error("SET: response was not OK");
  }
}

// RESP EXISTS
static bool do_exists(WorkerConn& conn, const std::string& key) {
  // build key header (can't pre-allocate)
  std::string key_header = conn.make_key_header(key);

  // send EXISTS cmd
  // iovec let's us combine pre-built parts and dynamic strings
  conn.send_multipart({{conn.exists_prefix.data(), conn.exists_prefix.size()},
                       {key_header.data(), key_header.size()}});

  // parse response (either :0\r\n or :1\r\n for non-batched EXISTS)
  char response[WorkerConn::exists_response_len];
  conn.recv_exactly(response, WorkerConn::exists_response_len);

  if (std::memcmp(response, conn.exists_one, WorkerConn::exists_response_len) ==
      0) {
    return true;
  } else if (std::memcmp(response, conn.exists_zero,
                         WorkerConn::exists_response_len) == 0) {
    return false;
  } else {
    throw std::runtime_error(
        "EXISTS returned invalid response that wasn't :0\r\n or :1\r\n");
  }
}

// MultiRESP means multi-threaded RESP (multiple workers)
class MultiRESPClient {
 public:
  // python interface
  // client = MultiRESPClient(host: str, port: int, chunk_size: int,
  // num_workers: int)
  MultiRESPClient(std::string host, int port, size_t chunk_size,
                  int num_workers)
      : host_(std::move(host)),
        port_(port),
        chunk_size_(chunk_size),
        num_workers_(num_workers) {
    if (num_workers_ <= 0) {
      throw std::runtime_error("num threads must > 0");
    }

    // default behavior of eventfd:
    // calling read on it returns the counter to 0 or blocks until the counter
    // is non-0 calling write on it increments the counter
    // flags:
    // EFD_NONBLOCK: read() and write() are both non-blocking
    // if no events are available on read(), return -1 instead
    // worker needs to poll / drain the fd without blocking
    efd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);

    if (efd_ < 0) {
      throw std::runtime_error("failed to create eventfd");
    }

    // start workers
    workers_.reserve((size_t)num_workers_);
    for (int i = 0; i < num_workers_; i++) {
      workers_.emplace_back([this]() { this->worker_loop(); });
    }
  }

  ~MultiRESPClient() { close(); }

  // python interface
  // fd = client.event_fd()
  // asyncio.get_running_loop().add_reader(fd, callback_that_drains_completions)
  int event_fd() const { return efd_; }

  // python interface (non-blocking)
  // future_id = client.submit_get(key: str, mv: memoryview)
  uint64_t submit_get(const std::string& key, py::memoryview mv) {
    return submit_with_buffer(Op::GET, key, mv);
  }

  // python interface (non-blocking)
  // future_id = client.submit_set(key: str, mv: memoryview)
  uint64_t submit_set(const std::string& key, py::memoryview mv) {
    return submit_with_buffer(Op::SET, key, mv);
  }

  // python interface (non-blocking)
  // future_id = client.submit_exists(key: str)
  uint64_t submit_exists(const std::string& key) {
    Request req;
    req.future_id = next_future_id_.fetch_add(1, std::memory_order_relaxed);
    req.op = Op::EXISTS;
    req.key = key;

    enqueue_request(std::move(req));
    return req.future_id;
  }

  // python interface (non-blocking)
  // future_id = client.submit_batch_get(keys: list[str], memviews:
  // list[memoryview])
  uint64_t submit_batch_get(const std::vector<std::string>& keys,
                            py::list memviews) {
    if (keys.size() != memviews.size()) {
      throw std::runtime_error("keys and memviews size mismatch");
    }

    // divide work evenly between workers into tiles (round up, the last tile
    // will be clipped)
    size_t num_items = keys.size();
    size_t num_tiles =
        std::min<size_t>(num_workers_, num_items);  // avoid empty tiles
    size_t tile_size = (num_items + num_tiles - 1) / num_tiles;  // round up

    // create shared batch state
    uint64_t batch_future_id =
        next_future_id_.fetch_add(1, std::memory_order_relaxed);
    auto batch_state = std::make_shared<BatchState>();
    batch_state->remaining_tiles.store(num_tiles, std::memory_order_relaxed);

    // fan out
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      size_t start = tile_idx * tile_size;
      size_t end = std::min(start + tile_size, num_items);  // clip last tile

      Request tile_req;
      tile_req.op = Op::BATCH_TILE_GET;
      tile_req.future_id = batch_future_id;
      tile_req.batch = batch_state;

      for (size_t i = start; i < end; ++i) {
        py::memoryview mv = memviews[i].cast<py::memoryview>();
        py::buffer_info info = py::buffer(mv).request();

        if (info.ndim != 1) throw std::runtime_error("buffer must be 1D");
        if (info.itemsize != 1)
          throw std::runtime_error("buffer must be byte addressable");
        if ((size_t)info.size != chunk_size_)
          throw std::runtime_error("buffer size != chunk_size");

        tile_req.keys.push_back(keys[i]);
        tile_req.buf_ptrs.push_back(info.ptr);
        tile_req.buf_owners.push_back(mv);
      }

      enqueue_request(std::move(tile_req));
    }

    return batch_future_id;
  }

  // python interface (non-blocking)
  // future_id = client.submit_batch_set(keys: list[str], memviews:
  // list[memoryview])
  uint64_t submit_batch_set(const std::vector<std::string>& keys,
                            py::list memviews) {
    if (keys.size() != memviews.size()) {
      throw std::runtime_error("keys and memviews size mismatch");
    }

    // divide work evenly between workers into tiles (round up, the last tile
    // will be clipped)
    size_t num_items = keys.size();
    size_t num_tiles =
        std::min<size_t>(num_workers_, num_items);  // avoid empty tiles
    size_t tile_size = (num_items + num_tiles - 1) / num_tiles;  // round up

    // create shared batch state
    uint64_t batch_future_id =
        next_future_id_.fetch_add(1, std::memory_order_relaxed);
    auto batch_state = std::make_shared<BatchState>();
    batch_state->remaining_tiles.store(num_tiles, std::memory_order_relaxed);

    // fan out
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      size_t start = tile_idx * tile_size;
      size_t end = std::min(start + tile_size, num_items);  // clip last tile

      Request tile_req;
      tile_req.op = Op::BATCH_TILE_SET;
      tile_req.future_id = batch_future_id;
      tile_req.batch = batch_state;

      for (size_t i = start; i < end; ++i) {
        py::memoryview mv = memviews[i].cast<py::memoryview>();
        py::buffer_info info = py::buffer(mv).request();

        if (info.ndim != 1) throw std::runtime_error("buffer must be 1D");
        if (info.itemsize != 1)
          throw std::runtime_error("buffer must be byte addressable");
        if ((size_t)info.size != chunk_size_)
          throw std::runtime_error("buffer size != chunk_size");

        tile_req.keys.push_back(keys[i]);
        tile_req.buf_ptrs.push_back(info.ptr);
        tile_req.buf_owners.push_back(mv);
      }

      enqueue_request(std::move(tile_req));
    }

    return batch_future_id;
  }

  /*
  Crucial: drain_completions *ALSO* drains the eventfd
  */
  // python interface (non-blocking)
  //  the python caller needs to manage its own futures
  //  [[future_id, ok, result_bool, error]] = client.drain_completions()
  //  fut = pending_futures[future_id]
  //  fut.set_result(result_bool or None)
  py::list drain_completions() {
    // drain the eventfd that cause this drain_completions callback to be
    // invoked
    drain_eventfd_();

    py::list completions_list;

    for (;;) {
      Completion c;
      {
        // we should prevent CQ submissions while we are consuming
        std::lock_guard<std::mutex> lk(comp_mu_);
        if (completions_.empty()) {
          signaled_.store(false, std::memory_order_release);
          // re-check: if someone raced and pushed after we decided empty but
          // before store
          if (!completions_.empty() &&
              !signaled_.exchange(true, std::memory_order_acq_rel)) {
            uint64_t x = 1;
            ::write(efd_, &x, sizeof(x));
          }
          break;
        }

        c = std::move(completions_.front());
        completions_.pop();
      }
      // convert Completion to python tuple: (future_id, ok, result_bool, error)
      completions_list.append(
          py::make_tuple(c.future_id, c.ok, c.result_bool, c.error));
    }

    return completions_list;
  }

  // python interface (blocking)
  // client.close()
  void close() {
    if (closed_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }

    // kick the stop signal to all the worker threads
    stop_.store(true, std::memory_order_release);
    // wake them all up to shutdown gracefully
    req_cv_.notify_all();

    // shutdown all worker sockets to unblock any recv()/writev() calls
    {
      std::lock_guard<std::mutex> lk(worker_fds_mu_);
      for (int fd : worker_fds_) {
        if (fd >= 0) {
          // SHUT_RDWR causes both recv and send to return immediately
          ::shutdown(fd, SHUT_RDWR);
        }
      }
    }

    for (auto& worker : workers_) {
      worker.join();
    }

    if (efd_ >= 0) {
      ::close(efd_);
      efd_ = -1;
    }

    // clear queues
    {
      py::gil_scoped_acquire gil;
      {
        std::lock_guard<std::mutex> lk(req_mu_);
        while (!requests_.empty()) {
          requests_.front().buf_owner = py::none();
          for (auto& owner : requests_.front().buf_owners) {
            owner = py::none();
          }
          requests_.pop();
        }
      }
      {
        std::lock_guard<std::mutex> lk(comp_mu_);
        while (!completions_.empty()) {
          completions_.pop();
        }
      }
    }
  }

 private:
  void enqueue_request(Request&& req) {
    {
      std::lock_guard<std::mutex> lk(req_mu_);
      requests_.push(std::move(req));
    }
    req_cv_.notify_one();
  }

  uint64_t submit_with_buffer(Op op, const std::string& key,
                              py::memoryview mv) {
    py::buffer_info info = py::buffer(mv).request();
    if (info.ndim != 1) throw std::runtime_error("memoryview must be 1D");
    if (info.itemsize != 1)
      throw std::runtime_error("memoryview must be byte addressable");
    if ((size_t)info.size != chunk_size_)
      throw std::runtime_error("buffer size != chunk_size");

    Request req;
    req.future_id = next_future_id_.fetch_add(1, std::memory_order_relaxed);
    req.op = op;
    req.key = key;
    req.buf_ptr = info.ptr;
    req.buf_len = (size_t)info.size;
    // need to ref count down later under GIL
    req.buf_owner = mv;

    enqueue_request(std::move(req));
    return req.future_id;
  }

  // the first completion after the eventfd signal is consumed will send another
  // signal
  void push_completion(Completion&& c) {
    {
      std::lock_guard<std::mutex> lk(comp_mu_);
      completions_.push(std::move(c));
    }
    signal_eventfd_();  // might not signal if not the first completion since
                        // last eventfd read
  }

  void drain_eventfd_() {
    // loop to consume all writes that happened since last drain/read
    for (;;) {
      uint64_t x;
      ssize_t r = ::read(efd_, &x, sizeof(x));
      // if counter > 0, r == 8 == sizeof(uint64_t)
      if (r == (ssize_t)sizeof(x)) {
        continue;  // keep draining (more writes might race in)
      }
      if (r < 0) {
        if (errno == EINTR) {
          continue;  // retry on EINTR
        }
        // we set EFD_NONBLOCK in the beginning
        // instead of blocking, r == -1 and errno == EAGAIN when no writes to
        // drain
        if (errno == EAGAIN) {
          break;  // drained
        }
      }
      break;
    }
  }

  // NOTE: only the first signal eventfd will write (before a read)
  /*
  the mutex on the completions queue prevents the following corner case:
  signal event fd, submit completion 1
  python drains eventfd and completions (simultaneously, submit completion 2)
  no signal is sent for completion 2 because we thought the signal was already
  sent only until completion 3 is sent will the eventfd be signaled again and
  python will then drain both completion 2 and completion 3
  */
  void signal_eventfd_() {
    bool already_signaled = signaled_.exchange(true, std::memory_order_acq_rel);
    // ensure only one write at once
    if (already_signaled) return;

    // now actually write to eventfd
    uint64_t x = 1;
    for (;;) {
      ssize_t w = ::write(efd_, &x, sizeof(x));
      if (w == (ssize_t)sizeof(x)) {
        return;  // success
      }
      if (w < 0) {
        if (errno == EINTR) {
          continue;  // retry on EINTR
        }
        // this shouldn't happen
        throw std::runtime_error(
            "assumption that eventfd is atomic was somehow broken");
      }
      // partial write shouldn't happen for eventfd
      throw std::runtime_error("partial write to eventfd");
    }
  }

  // background daemon per thread
  // minimize GIL access
  void worker_loop() {
    try {
      WorkerConn conn;
      conn.connect(host_, port_,
                   chunk_size_);  // one RESP session per worker/thread

      // register socket fd so close() can shutdown this socket
      {
        std::lock_guard<std::mutex> lk(worker_fds_mu_);
        worker_fds_.push_back(conn.fd);
      }

      for (;;) {
        Request req;  // make sure req is scoped

        // 1. grab a request from the SQ
        {
          // wait for a request to be available and grab it
          std::unique_lock<std::mutex> lk(req_mu_);
          req_cv_.wait(lk, [&] {
            return stop_.load(std::memory_order_acquire) || !requests_.empty();
          });
          if (stop_.load(std::memory_order_acquire) && requests_.empty()) {
            break;  // exit loop
          }
          req = std::move(requests_.front());
          requests_.pop();
        }

        Completion comp;
        // coupling between request and completion also needs to be handled on
        // the caller side
        comp.future_id = req.future_id;

        // 2. do the requested operation
        try {
          switch (req.op) {
            case Op::GET:
              do_get_into(conn, req.key, req.buf_ptr, req.buf_len);
              comp.ok = true;
              break;
            case Op::SET:
              do_set_from(conn, req.key, req.buf_ptr, req.buf_len);
              comp.ok = true;
              break;
            case Op::EXISTS:
              comp.result_bool = do_exists(conn, req.key);
              comp.ok = true;
              break;
            case Op::BATCH_TILE_GET:
              for (size_t i = 0; i < req.keys.size(); ++i) {
                do_get_into(conn, req.keys[i], req.buf_ptrs[i], chunk_size_);
              }
              comp.ok = true;
              break;
            case Op::BATCH_TILE_SET:
              for (size_t i = 0; i < req.keys.size(); ++i) {
                do_set_from(conn, req.keys[i], req.buf_ptrs[i], chunk_size_);
              }
              comp.ok = true;
              break;
          }
        } catch (const std::exception& e) {
          comp.ok = false;
          comp.error = e.what();
          // if we're shutting down, socket errors are expected
          if (stop_.load(std::memory_order_acquire)) {
            // cleanup and exit without pushing completion
            if (req.op == Op::BATCH_TILE_GET || req.op == Op::BATCH_TILE_SET) {
              py::gil_scoped_acquire gil;
              for (auto& owner : req.buf_owners) {
                owner = py::none();
              }
            } else {
              py::gil_scoped_acquire gil;
              req.buf_owner = py::none();
            }
            break;  // exit loop
          }
        }

        // 3. push completion to CQ

        // batch completions need to be "joined"
        if (req.op == Op::BATCH_TILE_GET || req.op == Op::BATCH_TILE_SET) {
          if (!comp.ok) {
            req.batch->any_failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(req.batch->err_mu);
            if (req.batch->first_error.empty()) {
              req.batch->first_error = comp.error;
            }
          }
          // release Python refs for this tile under GIL
          {
            py::gil_scoped_acquire gil;
            for (auto& owner : req.buf_owners) {
              owner = py::none();
            }
          }

          uint32_t tiles_left = req.batch->remaining_tiles.fetch_sub(
                                    1, std::memory_order_relaxed) -
                                1;
          if (tiles_left == 0) {
            // last tile to finish -- emit single completion for batch
            Completion batch_comp;
            batch_comp.future_id = req.future_id;
            batch_comp.ok =
                !req.batch->any_failed.load(std::memory_order_relaxed);
            if (!batch_comp.ok) {
              std::lock_guard<std::mutex> lk(req.batch->err_mu);
              batch_comp.error = req.batch->first_error;
            }
            push_completion(std::move(batch_comp));
          }
        }

        // single completions
        else {
          {
            py::gil_scoped_acquire gil;
            req.buf_owner = py::none();
          }

          push_completion(std::move(comp));
        }
      }
    } catch (const std::exception& e) {
      // catch any exception from connect() or other unexpected errors
      // log error and exit thread gracefully
      // we can't throw from here as it would call std::terminate
      (void)e;  // suppress unused warning
    } catch (...) {
      // catch any non-standard exception
    }
  }

 private:
  std::string host_;
  int port_;
  size_t chunk_size_;
  int num_workers_;

  int efd_ = -1;

  std::atomic<bool> stop_{false};
  std::atomic<bool> closed_{false};
  std::atomic<uint64_t> next_future_id_{1};

  // we treat eventfd not as a counter, but as a binary wakeup flag.
  // true: Python has been signaled (or will be)
  // false: Python is asleep, no wakeup pending
  std::atomic<bool> signaled_{false};

  /*
  SQ/CQ Design
  */
  std::mutex req_mu_;
  std::condition_variable req_cv_;
  // SUBMISSION QUEUE
  std::queue<Request> requests_;

  std::mutex comp_mu_;
  // COMPLETION QUEUE
  std::queue<Completion> completions_;

  std::vector<std::thread> workers_;

  // track worker socket fds so we can shutdown during close()
  std::mutex worker_fds_mu_;
  std::vector<int> worker_fds_;
};