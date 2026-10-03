// SPDX-License-Identifier: Apache-2.0
//
// Regression test for the ConnectorBase completion contract (see #4342).
//
// ConnectorBase hands every submitted batch a future_id and the L2 adapter
// keeps that id in its pending maps until a completion arrives. A worker that
// leaves worker_loop_for_queue without completing the requests it owns strands
// the id forever, and for stores it also pins the L1 read locks the task holds.
//
// Each case below drives one of the paths a worker can leave on and asserts
// that exactly one completion still comes back.

#include "connector_base.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using lmcache::connector::Completion;
using lmcache::connector::ConnectorBase;

namespace {

constexpr int kPollAttempts = 200;
constexpr auto kPollInterval = std::chrono::milliseconds(10);

// Collects completions until at least one arrives or the budget runs out.
std::vector<Completion> await_completions(ConnectorBase<int>& connector) {
  for (int attempt = 0; attempt < kPollAttempts; ++attempt) {
    std::vector<Completion> completions = connector.drain_completions();
    if (!completions.empty()) {
      return completions;
    }
    std::this_thread::sleep_for(kPollInterval);
  }
  return {};
}

// Minimal connector: every operation succeeds unless a subclass says otherwise.
class TestConnector : public ConnectorBase<int> {
 public:
  explicit TestConnector(int num_workers) : ConnectorBase(num_workers) {}
  ~TestConnector() override { close(); }

 protected:
  int create_connection() override { return 0; }
  void do_single_get(int&, const std::string&, void*, size_t,
                     size_t) override {}
  void do_single_set(int&, const std::string&, const void*, size_t,
                     size_t) override {}
  bool do_single_exists(int&, const std::string&) override { return true; }
};

// Path 1: create_connection() throws, so no worker ever reaches the request
// loop and the queue is left with nobody to drain it.
class UnreachableConnector : public TestConnector {
 public:
  UnreachableConnector() : TestConnector(2) { start_workers(); }

 protected:
  int create_connection() override {
    throw std::runtime_error("connection refused");
  }
};

// Path 2: a non-std::exception escapes the per-tile handler. It must fail the
// tile without killing the worker, so later requests are still served.
class ThrowsNonStandardOnce : public TestConnector {
 public:
  ThrowsNonStandardOnce() : TestConnector(1) { start_workers(); }

 protected:
  bool do_single_exists(int&, const std::string&) override {
    if (!already_threw_.exchange(true)) {
      throw 42;  // deliberately not derived from std::exception
    }
    return true;
  }

 private:
  std::atomic<bool> already_threw_{false};
};

bool expect_single_completion(const std::vector<Completion>& completions,
                              uint64_t future_id, bool expected_ok,
                              const char* case_name) {
  if (completions.size() != 1) {
    fprintf(stderr, "[%s] expected 1 completion, got %zu\n", case_name,
            completions.size());
    return false;
  }
  if (completions[0].future_id != future_id) {
    fprintf(stderr, "[%s] completion carried future_id %llu, expected %llu\n",
            case_name,
            static_cast<unsigned long long>(completions[0].future_id),
            static_cast<unsigned long long>(future_id));
    return false;
  }
  if (completions[0].ok != expected_ok) {
    fprintf(stderr, "[%s] completion ok=%d, expected %d (error: %s)\n",
            case_name, static_cast<int>(completions[0].ok),
            static_cast<int>(expected_ok), completions[0].error.c_str());
    return false;
  }
  return true;
}

// A batch submitted to a connector whose workers all died still completes.
bool dead_workers_still_complete_the_batch() {
  UnreachableConnector connector;
  // Let both workers fail create_connection() before anything is submitted.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  const std::vector<std::string> keys{"a", "b", "c"};
  const uint64_t future_id = connector.submit_batch_exists(keys);

  return expect_single_completion(await_completions(connector), future_id,
                                  false, "dead_workers");
}

// An unknown exception fails its own batch and leaves the worker able to serve
// the next one.
bool unknown_exception_fails_only_its_batch() {
  ThrowsNonStandardOnce connector;

  const std::vector<std::string> keys{"a"};
  const uint64_t failing_id = connector.submit_batch_exists(keys);
  if (!expect_single_completion(await_completions(connector), failing_id, false,
                                "unknown_exception/first")) {
    return false;
  }

  const uint64_t following_id = connector.submit_batch_exists(keys);
  return expect_single_completion(await_completions(connector), following_id,
                                  true, "unknown_exception/second");
}

struct TestCase {
  const char* name;
  bool (*run)();
};

}  // namespace

int main() {
  const TestCase cases[] = {
      {"dead_workers_still_complete_the_batch",
       dead_workers_still_complete_the_batch},
      {"unknown_exception_fails_only_its_batch",
       unknown_exception_fails_only_its_batch},
  };

  int failures = 0;
  for (const TestCase& test_case : cases) {
    const bool passed = test_case.run();
    printf("%s: %s\n", passed ? "PASS" : "FAIL", test_case.name);
    if (!passed) {
      ++failures;
    }
  }

  return failures == 0 ? 0 : 1;
}
