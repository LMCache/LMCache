// SPDX-License-Identifier: Apache-2.0
// HF3FS SDK double: payload I/O goes through POSIX pread/pwrite on the
// registered fd, and every ring/buffer handle is tracked so tests can detect
// leaks, double destroys, and I/O through destroyed handles.
#include "hf3fs_mock.h"
#include <hf3fs_usrbio.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace hf3fs_mock {
std::string mount_point;
std::atomic<int> created{0};
std::atomic<int> destroyed{0};
std::atomic<int> duplicate_destroys{0};
std::atomic<bool> hold_io{false};
std::atomic<bool> io_entered{false};
std::atomic<bool> io_released{false};
std::atomic<bool> exit_on_cleanup{false};
}  // namespace hf3fs_mock

namespace {
std::mutex resource_mutex;
std::unordered_set<void*> resources;
// The connector waits for each I/O right after submitting it on the same
// thread, so one pending result per thread is enough.
thread_local hf3fs_cqe pending;

void* create_resource(size_t size) {
  auto* ptr = new char[size];
  std::lock_guard<std::mutex> lock(resource_mutex);
  resources.insert(ptr);
  ++hf3fs_mock::created;
  return ptr;
}

void destroy_resource(void* ptr) {
  // The shutdown test stops at worker-local resource destruction, after
  // the worker has finished using the connector's members.
  if (hf3fs_mock::exit_on_cleanup) ::_exit(0);
  std::lock_guard<std::mutex> lock(resource_mutex);
  if (resources.erase(ptr) == 0) {
    ++hf3fs_mock::duplicate_destroys;
    return;
  }
  ++hf3fs_mock::destroyed;
  delete[] static_cast<char*>(ptr);
}

bool is_live(void* ptr) {
  std::lock_guard<std::mutex> lock(resource_mutex);
  return resources.count(ptr) != 0;
}
}  // namespace

extern "C" {
int hf3fs_extract_mount_point(char* out, int size, const char*) {
  const auto& path = hf3fs_mock::mount_point;
  if (path.size() + 1 <= static_cast<size_t>(size)) {
    std::memcpy(out, path.c_str(), path.size() + 1);
  }
  return static_cast<int>(path.size() + 1);
}

int hf3fs_iovcreate(hf3fs_iov* iov, const char*, size_t size, size_t, int) {
  // Like the SDK, map the full requested size; tests pick a small iov_size.
  iov->base = static_cast<uint8_t*>(create_resource(size));
  iov->size = size;
  return 0;
}

void hf3fs_iovdestroy(hf3fs_iov* iov) { destroy_resource(iov->base); }

int hf3fs_iorcreate4(hf3fs_ior* ior, const char*, int, bool, int, int, int,
                     uint64_t) {
  ior->iorh = create_resource(1);
  return 0;
}

void hf3fs_iordestroy(hf3fs_ior* ior) { destroy_resource(ior->iorh); }

int hf3fs_reg_fd(int, uint64_t) { return 0; }

void hf3fs_dereg_fd(int) {}

int hf3fs_prep_io(const hf3fs_ior* ior, const hf3fs_iov* iov, bool read,
                  void* ptr, int fd, size_t off, uint64_t len, const void*) {
  if (!is_live(ior->iorh) || !is_live(iov->base)) return -EFAULT;
  // I/O must stay inside the registered buffer.
  // Compare as integers: ptr may point outside the buffer.
  auto begin = reinterpret_cast<uintptr_t>(ptr);
  auto base = reinterpret_cast<uintptr_t>(iov->base);
  if (begin < base || len > iov->size || begin - base > iov->size - len) {
    return -EINVAL;
  }
  pending.result =
      read ? ::pread(fd, ptr, len, off) : ::pwrite(fd, ptr, len, off);
  if (pending.result < 0) pending.result = -errno;
  return 0;
}

int hf3fs_submit_ios(const hf3fs_ior*) { return 0; }

int hf3fs_wait_for_ios(const hf3fs_ior*, hf3fs_cqe* cqes, int, int,
                       const timespec*) {
  if (hf3fs_mock::hold_io) {
    hf3fs_mock::io_entered = true;
    while (!hf3fs_mock::io_released) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  cqes[0] = pending;
  return 1;
}
}
