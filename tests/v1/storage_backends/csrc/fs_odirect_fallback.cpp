// SPDX-License-Identifier: Apache-2.0
//
// Regression test for O_DIRECT handling in the native fs connector (#4364).
//
// O_DIRECT constrains the file offset, the transfer length and the buffer
// address. The connector used to gate only on length, so a host buffer that
// was length aligned but not address aligned passed the gate, open() accepted
// O_DIRECT and write() then answered EINVAL. Nothing classified that errno, so
// every store failed and the L2 tier stayed empty while reporting healthy.
//
// This test drives exactly that shape, a length aligned buffer at a
// deliberately unaligned address, and asserts the store still lands and reads
// back byte identical.

#include "fs/connector.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using lmcache::connector::Completion;
using lmcache::connector::FSConnector;

namespace {

constexpr size_t kAlign = 4096;
constexpr size_t kPayload = 8192;  // multiple of kAlign, so the length gate passes
constexpr size_t kSkew = 512;      // shifts the address off a kAlign boundary
constexpr int kPollAttempts = 300;

std::vector<Completion> await_completion(FSConnector& connector) {
  for (int attempt = 0; attempt < kPollAttempts; ++attempt) {
    std::vector<Completion> completions = connector.drain_completions();
    if (!completions.empty()) {
      return completions;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return {};
}

// Returns a pointer that is kSkew bytes past a kAlign boundary.
char* skewed_buffer(std::vector<char>* backing) {
  backing->assign(kPayload + 2 * kAlign, 0);
  auto base = reinterpret_cast<uintptr_t>(backing->data());
  uintptr_t aligned = (base + kAlign - 1) & ~(uintptr_t)(kAlign - 1);
  return reinterpret_cast<char*>(aligned + kSkew);
}

std::string make_temp_dir() {
  std::string tmpl =
      (std::filesystem::temp_directory_path() / "lmc_odirect_XXXXXX").string();
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  const char* dir = ::mkdtemp(buf.data());
  if (dir == nullptr) {
    throw std::runtime_error("mkdtemp failed");
  }
  return std::string(dir);
}

bool unaligned_address_still_stores_and_loads() {
  const std::string base = make_temp_dir();

  std::vector<char> store_backing;
  char* store_buf = skewed_buffer(&store_backing);
  for (size_t i = 0; i < kPayload; ++i) {
    store_buf[i] = static_cast<char>(i % 251);
  }

  if (reinterpret_cast<uintptr_t>(store_buf) % kAlign == 0) {
    fprintf(stderr, "test setup failed, buffer came out aligned\n");
    return false;
  }

  const std::vector<std::string> keys{"m@00000000@abcdef"};
  const std::vector<size_t> lens{kPayload};

  {
    FSConnector connector(base, 2, "", /*use_odirect=*/true, 0);
    std::vector<void*> bufs{store_buf};
    const uint64_t store_id = connector.submit_batch_set(keys, bufs, lens, 0);

    std::vector<Completion> done = await_completion(connector);
    if (done.size() != 1 || done[0].future_id != store_id) {
      fprintf(stderr, "store produced %zu completions\n", done.size());
      return false;
    }
    if (!done[0].ok) {
      fprintf(stderr, "store failed: %s\n", done[0].error.c_str());
      return false;
    }
  }

  std::vector<char> load_backing;
  char* load_buf = skewed_buffer(&load_backing);
  {
    FSConnector connector(base, 2, "", /*use_odirect=*/true, 0);
    std::vector<void*> bufs{load_buf};
    const uint64_t load_id = connector.submit_batch_get(keys, bufs, lens, 0);

    std::vector<Completion> done = await_completion(connector);
    if (done.size() != 1 || done[0].future_id != load_id) {
      fprintf(stderr, "load produced %zu completions\n", done.size());
      return false;
    }
    if (!done[0].ok) {
      fprintf(stderr, "load failed: %s\n", done[0].error.c_str());
      return false;
    }
  }

  if (std::memcmp(store_buf, load_buf, kPayload) != 0) {
    fprintf(stderr, "payload came back different\n");
    return false;
  }

  std::error_code ec;
  std::filesystem::remove_all(base, ec);
  return true;
}

}  // namespace

int main() {
#ifndef O_DIRECT
  // The round trip still runs, it just cannot exercise the refusal path, so
  // say so rather than reporting a pass that proves less than it looks like.
  printf("NOTE: no O_DIRECT on this platform, buffered path only\n");
#endif
  const bool passed = unaligned_address_still_stores_and_loads();
  printf("%s: unaligned_address_still_stores_and_loads\n",
         passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
