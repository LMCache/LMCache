// SPDX-License-Identifier: Apache-2.0

#include "resp.h"
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

// a TCP session (one per thread) implementing RESP2
/*
Key Optimizations include:
1. preset chunk_bytes (allows not parsing for \r\n byte-by-byte)
2. scatter/gather sending of data (with pre-allocated buffers)
3. zero copy (no bounce bufferse)
*/
struct WorkerConn {
  int fd = -1;
  std::string host;
  int port;
  size_t chunk_bytes;

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

  void connect(const std::string& host, int port, size_t chunk_bytes) {
    this->host = host;
    this->port = port;
    this->chunk_bytes = chunk_bytes;

    // reusable headers (for scatter/gather)

    {
      std::ostringstream oss;
      oss << "$" << chunk_bytes << "\r\n";
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
  // we only read exactly chunk_bytes bytes (save_unfull_chunk must be off)
  if (len != conn.chunk_bytes) {
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
  // we only write exactly chunk_bytes bytes (save_unfull_chunk must be off)
  if (len != conn.chunk_bytes) {
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

  // parse response (either :0\r\n or :1\r\n)
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
// constructor
MultiRESPClient::MultiRESPClient(std::string host, int port, size_t chunk_bytes,
                                 int num_workers)
    : host_(std::move(host)),
      port_(port),
      chunk_bytes_(chunk_bytes),
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

// destructor
MultiRESPClient::~MultiRESPClient() { close(); }

int MultiRESPClient::event_fd() const { return efd_; }

uint64_t MultiRESPClient::submit_batch_get(const std::vector<std::string>& keys,
                                           const std::vector<void*>& bufs,
                                           const std::vector<size_t>& lens) {
  if (keys.size() != bufs.size() || keys.size() != lens.size()) {
    throw std::runtime_error("keys, bufs, and lens size mismatch");
  }

  size_t num_items = keys.size();
  if (num_items == 0) {
    throw std::runtime_error("keys list is empty");
  }

  // divide work evenly between workers into tiles (round up, the last tile
  // will be clipped)
  size_t num_tiles =
      std::min<size_t>(num_workers_, num_items);  // avoid empty tiles
  size_t tile_size = (num_items + num_tiles - 1) / num_tiles;  // round up

  // create shared batch state
  uint64_t batch_future_id =
      next_future_id_.fetch_add(1, std::memory_order_relaxed);
  auto batch_state = std::make_shared<BatchState>();
  batch_state->remaining_tiles.store(num_tiles, std::memory_order_relaxed);
  batch_state->batch_op = Op::BATCH_TILE_GET;

  // fan out
  for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    size_t start = tile_idx * tile_size;
    size_t end = std::min(start + tile_size, num_items);  // clip last tile

    Request tile_req;
    tile_req.op = Op::BATCH_TILE_GET;
    tile_req.future_id = batch_future_id;
    tile_req.batch = batch_state;

    for (size_t i = start; i < end; ++i) {
      tile_req.keys.push_back(keys[i]);
      tile_req.buf_ptrs.push_back(bufs[i]);
      tile_req.buf_lens.push_back(lens[i]);
    }

    enqueue_request(std::move(tile_req));
  }

  return batch_future_id;
}

uint64_t MultiRESPClient::submit_batch_set(const std::vector<std::string>& keys,
                                           const std::vector<void*>& bufs,
                                           const std::vector<size_t>& lens) {
  if (keys.size() != bufs.size() || keys.size() != lens.size()) {
    throw std::runtime_error("keys, bufs, and lens size mismatch");
  }

  size_t num_items = keys.size();
  if (num_items == 0) {
    throw std::runtime_error("keys list is empty");
  }

  // divide work evenly between workers into tiles (round up, the last tile
  // will be clipped)
  size_t num_tiles =
      std::min<size_t>(num_workers_, num_items);  // avoid empty tiles
  size_t tile_size = (num_items + num_tiles - 1) / num_tiles;  // round up

  // create shared batch state
  uint64_t batch_future_id =
      next_future_id_.fetch_add(1, std::memory_order_relaxed);
  auto batch_state = std::make_shared<BatchState>();
  batch_state->remaining_tiles.store(num_tiles, std::memory_order_relaxed);
  batch_state->batch_op = Op::BATCH_TILE_SET;

  // fan out
  for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    size_t start = tile_idx * tile_size;
    size_t end = std::min(start + tile_size, num_items);  // clip last tile

    Request tile_req;
    tile_req.op = Op::BATCH_TILE_SET;
    tile_req.future_id = batch_future_id;
    tile_req.batch = batch_state;

    for (size_t i = start; i < end; ++i) {
      tile_req.keys.push_back(keys[i]);
      tile_req.buf_ptrs.push_back(bufs[i]);
      tile_req.buf_lens.push_back(lens[i]);
    }

    enqueue_request(std::move(tile_req));
  }

  return batch_future_id;
}

uint64_t MultiRESPClient::submit_batch_exists(
    const std::vector<std::string>& keys) {
  size_t num_items = keys.size();
  if (num_items == 0) {
    throw std::runtime_error("keys list is empty");
  }

  // divide work evenly between workers into tiles (round up, the last tile
  // will be clipped)
  size_t num_tiles =
      std::min<size_t>(num_workers_, num_items);  // avoid empty tiles
  size_t tile_size = (num_items + num_tiles - 1) / num_tiles;  // round up

  // create shared batch state
  uint64_t batch_future_id =
      next_future_id_.fetch_add(1, std::memory_order_relaxed);
  auto batch_state = std::make_shared<BatchState>();
  batch_state->remaining_tiles.store(num_tiles, std::memory_order_relaxed);
  batch_state->batch_op = Op::BATCH_TILE_EXISTS;
  // Use uint8_t instead of bool to avoid vector<bool> concurrent write data
  // race
  batch_state->exists_results.assign(num_items, 0);

  // fan out
  for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    size_t start = tile_idx * tile_size;
    size_t end = std::min(start + tile_size, num_items);  // clip last tile

    Request tile_req;
    tile_req.op = Op::BATCH_TILE_EXISTS;
    tile_req.future_id = batch_future_id;
    tile_req.batch = batch_state;
    tile_req.start_idx = start;

    for (size_t i = start; i < end; ++i) {
      tile_req.keys.push_back(keys[i]);
    }

    enqueue_request(std::move(tile_req));
  }

  return batch_future_id;
}

// crucial: drain_completions *ALSO* drains the eventfd
std::vector<Completion> MultiRESPClient::drain_completions() {
  // drain the eventfd that cause this drain_completions callback to be
  // invoked
  drain_eventfd_();

  std::vector<Completion> completions_list;

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
    completions_list.push_back(std::move(c));
  }

  return completions_list;
}

void MultiRESPClient::close() {
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

  // clear queues (no GIL needed - python guarantees buffers stay alive)
  {
    std::lock_guard<std::mutex> lk(req_mu_);
    while (!requests_.empty()) {
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

// private methods
void MultiRESPClient::enqueue_request(Request&& req) {
  {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_.push(std::move(req));
  }
  req_cv_.notify_one();
}

// the first completion after the eventfd signal is consumed will send another
// signal
void MultiRESPClient::push_completion(Completion&& c) {
  {
    std::lock_guard<std::mutex> lk(comp_mu_);
    completions_.push(std::move(c));
  }
  signal_eventfd_();  // might not signal if not the first completion since
                      // last eventfd read
}

void MultiRESPClient::drain_eventfd_() {
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
void MultiRESPClient::signal_eventfd_() {
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
void MultiRESPClient::worker_loop() {
  try {
    WorkerConn conn;
    conn.connect(host_, port_,
                 chunk_bytes_);  // one RESP session per worker/thread

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
          case Op::BATCH_TILE_GET:
            for (size_t i = 0; i < req.keys.size(); ++i) {
              do_get_into(conn, req.keys[i], req.buf_ptrs[i], req.buf_lens[i]);
            }
            comp.ok = true;
            break;
          case Op::BATCH_TILE_SET:
            for (size_t i = 0; i < req.keys.size(); ++i) {
              do_set_from(conn, req.keys[i], req.buf_ptrs[i], req.buf_lens[i]);
            }
            comp.ok = true;
            break;
          case Op::BATCH_TILE_EXISTS:
            for (size_t i = 0; i < req.keys.size(); ++i) {
              bool exists = do_exists(conn, req.keys[i]);
              // write result as uint8_t (0/1) to avoid vector<bool> data race
              req.batch->exists_results[req.start_idx + i] = exists ? 1 : 0;
            }
            comp.ok = true;
            break;
        }
      } catch (const std::exception& e) {
        comp.ok = false;
        comp.error = e.what();
        // if we're shutting down, socket errors are expected
        if (stop_.load(std::memory_order_acquire)) {
          // exit without pushing completion (no cleanup needed - Python
          // guarantees lifetime)
          break;  // exit loop
        }
      }

      // 3. push completion to CQ

      // All operations are batched tiles that need to be "joined"
      // (multiple worker threads coordinate to complete a single batch
      // operation)
      if (!comp.ok) {
        req.batch->any_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(req.batch->err_mu);
        if (req.batch->first_error.empty()) {
          req.batch->first_error = comp.error;
        }
      }
      // No GIL or python ref cleanup needed - python guarantees lifetime

      uint32_t tiles_left =
          req.batch->remaining_tiles.fetch_sub(1, std::memory_order_relaxed) -
          1;
      if (tiles_left == 0) {
        // last tile to finish -- emit single completion for batch
        Completion batch_comp;
        batch_comp.future_id = req.future_id;
        batch_comp.ok = !req.batch->any_failed.load(std::memory_order_relaxed);
        if (!batch_comp.ok) {
          std::lock_guard<std::mutex> lk(req.batch->err_mu);
          batch_comp.error = req.batch->first_error;
        }
        // for batch exists, copy the results (use batch_op to determine type)
        if (req.batch->batch_op == Op::BATCH_TILE_EXISTS) {
          batch_comp.result_bytes = std::move(req.batch->exists_results);
        }
        push_completion(std::move(batch_comp));
      }
    }
  } catch (const std::exception& e) {
    // catch any exception from connect() or other unexpected errors
    // log error and exit thread gracefully
    // we can't throw from here as it would call std::terminate
    fprintf(stderr, "[LMCache RESP Worker Error] Caught exception: %s\n",
            e.what());
  } catch (...) {
    // catch any non-standard exception
    fprintf(stderr, "[LMCache RESP Worker Error] Caught unknown exception\n");
  }
}