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
  std::atomic<uint32_t> remaining{0};
  std::atomic<bool> any_failed{false};

  std::mutex err_mu;
  std::string first_error;
};

struct Request {
  uint64_t future_id = 0;
  Op op;
  std::string key;

  void* buf_ptr = nullptr;
  size_t buf_len = 0;

  // python passes in a memoryview or bytearray
  // following the buffer protocol
  py_object buf_owner;

  // for tiling batched operations
  std::vector<std::string> keys;
  std::vector<void*> buf_ptrs;
  std::vector<py::object> buf_owners;

  uint64_t batch_token = 0;
  // shared batch state between threads executing a single batch operation
  std::shared_ptr<BatchState> batch;
}

struct Completion {
  // the caller should index a future by this id
  uint64_t future_id = 0;

  bool ok = true;
  bool result_bool = false;

  std::string error;
}

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
    hints.ai_familiy = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    int err = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);

    if (err != 0) {
      ::close(fd);
      throw std :;
      runtime_error(std::string("getaddrinfo failed: ") + gai_strerror(err));
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
      if (n <= 0) {
        throw std::runtime_error("socket send failed");
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
      iov.push_back({const_cast<void*>(p.first), p.second});
    }

    size_t total_to_send = 0;
    for (const auto& part : parts) {
      total_to_send += p.second;
    }

    size_t sent_so_far = 0;
    size_t iov_idx = 0;
    size_t iov_offset = 0;

    while (sent < total_to_send) {
      ssize_t n = ::writev(fd, &iov[iov_idx], iov.size() - iov_idx);
      if (n <= 0) {
        throw std::runtime_error("socket writev failed");
      }

      sent_so_far += n;

      // adjust iovec for partial writes
      size_t remaining = n;
      while (remaining > 0 && iov_idx < iov.size()) {
        if (remaining >= iov[iov_idx].iov_len - iov_offset) {
          // the part was fully consumed
          remaining -= (iov[iov_idx].iov_len - iov_offset);
          iov_offset = 0;
          iov_idx++;
        } else {
          // the part was partially consumed
          iov_offset += remaining;
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
      if (n <= 0) {
        throw std::runtime_error("socket recv failed");
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
the RESP responses a single error could break our assumptions we can mitigate
this by actually parsing the headers and trailers at a minimum
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
static bool do_exists(WorkerConn& conn, const std::string* key) {
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
  // python interface
  // client = MultiRESPClient(host: str, port: int, chunk_size: int,
  // num_workers: int)
 public:
  MultiRESPClient(std::string host, int port, size_t chunk_size,
                  int num_workers)
      : host_(std::move(host)), port_(port), chunk_size_(chunk_size) {
    if (num_threads <= 0) {
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
    workers_.reserve((size_t)num_threads);
    for (int i = 0; i < num_threads; i++) {
      workers_.emplace_back([this]() { this->worker_loop(); });
    }
  }

  ~MultiRESPClient() { close(); }

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
}