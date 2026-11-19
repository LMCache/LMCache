// SPDX-License-Identifier: Apache-2.0
// Test-only declarations of the HF3FS USRBIO SDK subset that
// csrc/storage_backends/hf3fs uses, so its regressions build without the
// SDK. hf3fs_mock.cpp implements these functions; production builds use the
// real SDK header.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

struct hf3fs_iov {
  uint8_t* base;
  size_t size;
};

struct hf3fs_ior {
  void* iorh;
};

struct hf3fs_cqe {
  int64_t result;
};

int hf3fs_extract_mount_point(char* hf3fs_mount_point, int size,
                              const char* path);
int hf3fs_iovcreate(struct hf3fs_iov* iov, const char* hf3fs_mount_point,
                    size_t size, size_t block_size, int numa);
void hf3fs_iovdestroy(struct hf3fs_iov* iov);
int hf3fs_iorcreate4(struct hf3fs_ior* ior, const char* hf3fs_mount_point,
                     int entries, bool for_read, int io_depth, int timeout,
                     int numa, uint64_t flags);
void hf3fs_iordestroy(struct hf3fs_ior* ior);
int hf3fs_reg_fd(int fd, uint64_t flags);
void hf3fs_dereg_fd(int fd);
int hf3fs_prep_io(const struct hf3fs_ior* ior, const struct hf3fs_iov* iov,
                  bool read, void* ptr, int fd, size_t off, uint64_t len,
                  const void* userdata);
int hf3fs_submit_ios(const struct hf3fs_ior* ior);
int hf3fs_wait_for_ios(const struct hf3fs_ior* ior, struct hf3fs_cqe* cqes,
                       int cqec, int min_results,
                       const struct timespec* abs_timeout);

#ifdef __cplusplus
}
#endif
