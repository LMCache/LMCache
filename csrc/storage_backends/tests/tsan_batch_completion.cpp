// SPDX-License-Identifier: Apache-2.0
//
// Regression test for the batch-completion handoff in ConnectorBase.
//
// The tiles of one batch write disjoint slots of BatchState::per_key_results
// and fill the caller's read buffers, then decrement remaining_tiles; the tile
// that observes zero moves per_key_results into the Completion that
// drain_completions() hands back. That decrement must be acq_rel so each tile
// releases its writes and the last tile acquires them. With a relaxed
// decrement there is no happens-before edge between tiles and
// ThreadSanitizer reports the accesses as racing.
//
// This test drives the FS connector -- the only built-in connector with no
// external service dependency -- across every batched op and reads back every
// per-key result byte and payload byte, so the racing accesses are executed.
// It asserts the functional outcome too, but the ordering defect is what
// ThreadSanitizer observes: build without -fsanitize=thread and this test
// passes either way.
//
// Build and run:
//
//   g++ -std=c++17 -fsanitize=thread -g -O1 -I csrc/storage_backends \
//       csrc/storage_backends/tests/tsan_batch_completion.cpp \
//       csrc/storage_backends/fs/connector.cpp -o tsan_batch_completion \
//       -lpthread
//   TSAN_OPTIONS=halt_on_error=1 ./tsan_batch_completion <scratch-dir>
//
// On kernels that place the binary outside the TSan shadow mapping, run it
// under `setarch $(uname -m) -R` to disable ASLR.

#include "fs/connector.h"
#include <poll.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using lmcache::connector::Completion;
using lmcache::connector::FSConnector;

constexpr int kNumWorkers = 4;
constexpr size_t kNumKeys = 64;
constexpr size_t kValueBytes = 4096;
constexpr int kPollTimeoutMs = 30000;

int g_failures = 0;

void expect_eq(size_t actual, size_t expected, const char* what) {
  if (actual == expected) return;
  fprintf(stderr, "FAIL %s: expected %zu, got %zu\n", what, expected, actual);
  ++g_failures;
}

// Distinct per-key payload so a GET that returns another key's bytes, or no
// bytes at all, is caught rather than silently passing.
char payload_byte(size_t key_idx, size_t offset) {
  return static_cast<char>((key_idx * 31 + offset) & 0xff);
}

// Block until the connector signals, drain once, and return the completions.
// A batch emits exactly one completion, so one drain is enough per op.
std::vector<Completion> await_completions(FSConnector& conn, const char* what) {
  struct pollfd pfd = {conn.event_fd(), POLLIN, 0};
  for (;;) {
    int ready = ::poll(&pfd, 1, kPollTimeoutMs);
    if (ready <= 0) {
      fprintf(stderr, "FAIL %s: timed out waiting for completion\n", what);
      ++g_failures;
      return {};
    }
    std::vector<Completion> completions = conn.drain_completions();
    if (!completions.empty()) return completions;
  }
}

// Read every per-key result byte. This is the access that races with the tile
// threads' writes when the tile counter is decremented with relaxed ordering.
size_t count_successes(const std::vector<Completion>& completions,
                       const char* what) {
  expect_eq(completions.size(), 1, what);
  size_t successes = 0;
  for (const Completion& c : completions) {
    if (!c.ok) {
      fprintf(stderr, "FAIL %s: batch reported failure: %s\n", what,
              c.error.c_str());
      ++g_failures;
    }
    for (uint8_t b : c.result_bytes) {
      if (b) ++successes;
    }
  }
  return successes;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <scratch-dir>\n", argv[0]);
    return 2;
  }

  FSConnector conn(argv[1], kNumWorkers);

  std::vector<std::string> keys;
  std::vector<std::vector<char>> write_bufs;
  std::vector<void*> write_ptrs;
  std::vector<size_t> lens;
  keys.reserve(kNumKeys);
  write_bufs.reserve(kNumKeys);
  for (size_t i = 0; i < kNumKeys; ++i) {
    // Wire format: <model>@<kv_rank_hex>@<group_hex>@<chunk_hash_hex>
    keys.push_back("tsan-model@00000001@0@" + std::to_string(100000 + i));
    std::vector<char> buf(kValueBytes);
    for (size_t off = 0; off < kValueBytes; ++off) {
      buf[off] = payload_byte(i, off);
    }
    write_bufs.push_back(std::move(buf));
    lens.push_back(kValueBytes);
  }
  for (size_t i = 0; i < kNumKeys; ++i) {
    write_ptrs.push_back(write_bufs[i].data());
  }

  conn.submit_batch_set(keys, write_ptrs, lens, kValueBytes);
  std::vector<Completion> set_done = await_completions(conn, "SET");
  expect_eq(set_done.size(), 1, "SET completion count");
  for (const Completion& c : set_done) {
    if (!c.ok) {
      fprintf(stderr, "FAIL SET: %s\n", c.error.c_str());
      ++g_failures;
    }
  }

  conn.submit_batch_exists(keys);
  expect_eq(count_successes(await_completions(conn, "EXISTS"), "EXISTS"),
            kNumKeys, "EXISTS hits");

  std::vector<std::vector<char>> read_bufs(kNumKeys,
                                           std::vector<char>(kValueBytes, 0));
  std::vector<void*> read_ptrs;
  read_ptrs.reserve(kNumKeys);
  for (size_t i = 0; i < kNumKeys; ++i) {
    read_ptrs.push_back(read_bufs[i].data());
  }
  conn.submit_batch_get(keys, read_ptrs, lens, kValueBytes);
  expect_eq(count_successes(await_completions(conn, "GET"), "GET"), kNumKeys,
            "GET hits");

  // Read back every payload byte a worker thread wrote through ::read().
  size_t mismatched = 0;
  for (size_t i = 0; i < kNumKeys; ++i) {
    for (size_t off = 0; off < kValueBytes; ++off) {
      if (read_bufs[i][off] != payload_byte(i, off)) {
        ++mismatched;
        break;
      }
    }
  }
  expect_eq(mismatched, 0, "GET payload mismatches");

  conn.submit_batch_delete(keys);
  expect_eq(count_successes(await_completions(conn, "DELETE"), "DELETE"),
            kNumKeys, "DELETE hits");

  conn.submit_batch_exists(keys);
  expect_eq(count_successes(await_completions(conn, "EXISTS after DELETE"),
                            "EXISTS after DELETE"),
            0, "EXISTS hits after DELETE");

  conn.close();

  if (g_failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  printf("ok: %zu keys across set/exists/get/delete with %d workers\n",
         kNumKeys, kNumWorkers);
  return 0;
}
