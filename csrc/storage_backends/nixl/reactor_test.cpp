// SPDX-License-Identifier: Apache-2.0
// Deterministic behavior tests for the internal CrossCallReactor admission
// and load-fallback state machine (no NIXL agent, no Ceph).
// Build:
//   g++ -std=c++20 -O1 -pthread -I csrc/storage_backends -I
//   csrc/storage_backends/nixl -I "$NIXL_INCLUDE_DIR"
//   csrc/storage_backends/nixl/reactor_test.cpp -L "$NIXL_LIB_DIR" -lnixl
//   -Wl,-rpath,"$NIXL_LIB_DIR" -o reactor_test
// Then run ./reactor_test. libnixl is linked only for
// nixlEnumStrings::statusStr; no live NIXL behavior is exercised or claimed.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "reactor.h"

namespace {

#define CHECK(cond)                                                  \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::fprintf(stderr, "FAILED: %s at %s:%d\n", #cond, __FILE__, \
                   __LINE__);                                        \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

using namespace lmcache::connector;

template <typename Pred>
bool wait_until(Pred pred, int timeout_ms = 5000) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (!pred()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

struct FakeState {
  std::atomic<int> live{0};
  std::atomic<int> post_calls{0};
  std::atomic<bool> release{false};
  std::atomic<bool> posted{false};
  std::atomic<bool> released{false};
};

// Fallback outcomes for a failed first load transfer.
constexpr int kFallbackHardThrow = 0;  // query error: raw throw, tile fails
constexpr int kFallbackSoft = 1;       // all keys present: false bits, ok
constexpr int kFallbackEmpty = 2;      // none present: empty retry
constexpr int kFallbackRetry = 3;      // present subset: real retry

// Ordered event log shared by the driver and its fake operations.
struct TraceLog {
  std::mutex mu;
  std::vector<std::string> events;
  void add(const std::string& event) {
    std::lock_guard<std::mutex> lock(mu);
    events.push_back(event);
  }
};

struct Script {
  bool posts = true;
  int prepare_mode = 0;  // 0 ok, 1 throw hard, 2 throw load-soft
  bool throw_on_post = false;
  bool stop_in_complete = false;
  int hold_polls = 0;  // >0: stay IN_PROG for this many polls even released
  nixl_status_t post_status = NIXL_IN_PROG;
  nixl_status_t terminal_status = NIXL_SUCCESS;
  std::vector<uint8_t> soft_results;
  std::vector<uint8_t> finish_results;
  std::string name = "op";
  std::shared_ptr<FakeState> state = std::make_shared<FakeState>();
  int fallback_mode = kFallbackRetry;
  std::vector<uint8_t> fallback_soft_results;
  std::shared_ptr<Script> retry_script;
  // Stamped by FakeDriver::set.
  TraceLog* trace = nullptr;
  std::atomic<bool>* stop_flag = nullptr;
};

class FakeOperation final : public NixlPendingOperation {
 public:
  explicit FakeOperation(Script script) : script_(std::move(script)) {
    script_.state->live.fetch_add(1);
  }
  ~FakeOperation() override { script_.state->live.fetch_add(-1); }

  nixl_status_t post() override {
    script_.state->post_calls.fetch_add(1);
    if (script_.throw_on_post)
      throw std::runtime_error("fake submission failure");
    script_.state->posted.store(true);
    if (script_.post_status != NIXL_IN_PROG) latched_ = true;
    return script_.post_status;
  }

  nixl_status_t poll() override {
    ++polls_;
    if (script_.post_status == NIXL_IN_PROG && !script_.state->release.load() &&
        (script_.hold_polls == 0 || polls_ <= script_.hold_polls)) {
      return NIXL_IN_PROG;
    }
    latched_ = true;
    return script_.terminal_status;
  }

  std::vector<uint8_t> complete() override {
    release();
    if (latched_ && script_.terminal_status != NIXL_SUCCESS) {
      // Mirrors the real operation: failed transfers keep false bits.
      return std::vector<uint8_t>(script_.finish_results.size(), 0);
    }
    return std::move(script_.finish_results);
  }

  void discard() noexcept override { release(); }

 private:
  void release() {
    script_.state->released.store(true);
    if (script_.stop_in_complete) script_.stop_flag->store(true);
    script_.trace->add(script_.name + ":released");
  }

  Script script_;
  int polls_ = 0;
  bool latched_ = false;
};

class FakeDriver final : public NixlReactorDriver {
 public:
  explicit FakeDriver(std::atomic<bool>* stop_flag) : stop_flag_(stop_flag) {}

  void set(uint64_t future_id, Script script) {
    script.trace = &trace_log_;
    script.stop_flag = stop_flag_;
    if (script.retry_script) {
      script.retry_script->trace = &trace_log_;
      script.retry_script->stop_flag = stop_flag_;
    }
    scripts_[future_id] = std::move(script);
  }

  std::shared_ptr<FakeState> state(uint64_t future_id) {
    return scripts_.at(future_id).state;
  }

  std::shared_ptr<FakeState> retry_state(uint64_t future_id) {
    return scripts_.at(future_id).retry_script->state;
  }

  bool posts_tile(const Request& request) const override {
    return scripts_.at(request.future_id).posts;
  }

  void execute_sync(const Request&) override { ++sync_calls_; }

  std::unique_ptr<NixlPendingOperation> prepare(
      const Request& request) override {
    const Script& script = scripts_.at(request.future_id);
    {
      std::lock_guard<std::mutex> lock(trace_mu_);
      prepare_order_.push_back(request.future_id);
    }
    if (script.prepare_mode == 1) {
      throw std::runtime_error("fake hard prepare failure");
    }
    if (script.prepare_mode == 2) {
      throw NixlLoadPrepareError("fake load prepare failure",
                                 script.soft_results);
    }
    return std::make_unique<FakeOperation>(script);
  }

  std::unique_ptr<NixlPendingOperation> prepare_load_fallback(
      const Request& request) override {
    const Script& script = scripts_.at(request.future_id);
    ++fallback_calls_;
    trace_log_.add("fallback:" + script.name);
    switch (script.fallback_mode) {
      case kFallbackHardThrow:
        throw std::runtime_error("fake fallback query failure");
      case kFallbackSoft:
        throw NixlLoadPrepareError("fake load retry skipped",
                                   script.fallback_soft_results);
      case kFallbackEmpty: {
        Script empty;
        empty.post_status = NIXL_SUCCESS;
        empty.finish_results.assign(request.keys.size(), 0);
        empty.trace = &trace_log_;
        empty.stop_flag = stop_flag_;
        return std::make_unique<FakeOperation>(std::move(empty));
      }
      default:
        return std::make_unique<FakeOperation>(*script.retry_script);
    }
  }

  void log_load_error(const std::string&) override {}

  size_t trace_index(const std::string& event) {
    std::lock_guard<std::mutex> lock(trace_log_.mu);
    for (size_t index = 0; index < trace_log_.events.size(); ++index) {
      if (trace_log_.events[index] == event) return index;
    }
    return 1000;
  }

  std::vector<uint64_t> prepare_order() {
    std::lock_guard<std::mutex> lock(trace_mu_);
    return prepare_order_;
  }

  size_t fallback_calls() const { return fallback_calls_.load(); }
  size_t sync_calls() const { return sync_calls_.load(); }

 private:
  std::atomic<bool>* stop_flag_;
  std::unordered_map<uint64_t, Script> scripts_;  // frozen before run
  TraceLog trace_log_;
  std::mutex trace_mu_;
  std::vector<uint64_t> prepare_order_;
  std::atomic<size_t> fallback_calls_{0};
  std::atomic<size_t> sync_calls_{0};
};

std::shared_ptr<BatchState> make_batch(Op op, size_t num_items) {
  auto batch = std::make_shared<BatchState>();
  batch->batch_op = op;
  batch->remaining_tiles.store(1);
  batch->per_key_results.assign(num_items, 0);
  return batch;
}

Request make_request(uint64_t future_id, Op op, size_t num_keys,
                     std::shared_ptr<BatchState> batch, size_t start_idx) {
  Request request;
  request.future_id = future_id;
  request.op = op;
  for (size_t index = 0; index < num_keys; ++index) {
    request.keys.push_back("k" + std::to_string(index));
  }
  request.batch = std::move(batch);
  request.start_idx = start_idx;
  return request;
}

class Harness {
 public:
  void start() {
    worker_ = std::thread([this] {
      CrossCallReactor reactor(driver);
      try {
        reactor.run(
            queue_mu_, queue_cv, queue_, [this] { return stop.load(); },
            [this](const Request& request, const Completion& completion) {
              std::lock_guard<std::mutex> lock(emit_mu_);
              ++emit_attempts_;
              if (throw_next_emit_) {
                throw_next_emit_ = false;
                throw std::runtime_error("fake emit failure");
              }
              emits_.push_back({request.future_id, completion.ok});
            });
      } catch (...) {
        run_threw.store(true);
      }
    });
  }

  void push(Request request) {
    {
      std::lock_guard<std::mutex> lock(queue_mu_);
      queue_.push(std::move(request));
    }
    queue_cv.notify_one();
  }

  void stop_and_join() {
    {
      // Serialize with the worker's predicate evaluation: a store outside
      // queue_mu_ can land between check and wait and lose the wakeup.
      std::lock_guard<std::mutex> lock(queue_mu_);
      stop.store(true);
    }
    queue_cv.notify_all();
    worker_.join();
  }

  void join() { worker_.join(); }

  size_t emit_count() {
    std::lock_guard<std::mutex> lock(emit_mu_);
    return emits_.size();
  }

  // The next completion callback throws once; later callbacks record.
  void fail_next_emit() {
    std::lock_guard<std::mutex> lock(emit_mu_);
    throw_next_emit_ = true;
  }

  size_t emit_attempts() {
    std::lock_guard<std::mutex> lock(emit_mu_);
    return emit_attempts_;
  }

  // The future_id completed exactly once with the given status.
  bool emitted(uint64_t future_id, bool ok) {
    std::lock_guard<std::mutex> lock(emit_mu_);
    size_t count = 0;
    for (const Emit& emit : emits_) {
      if (emit.future_id != future_id) continue;
      ++count;
      if (emit.ok != ok) return false;
    }
    return count == 1;
  }

  FakeDriver driver{&stop};
  std::atomic<bool> stop{false};
  std::atomic<bool> run_threw{false};
  std::condition_variable queue_cv;

 private:
  struct Emit {
    uint64_t future_id;
    bool ok;
  };

  std::mutex queue_mu_;
  std::queue<Request> queue_;
  std::mutex emit_mu_;
  std::vector<Emit> emits_;
  size_t emit_attempts_ = 0;
  bool throw_next_emit_ = false;
  std::thread worker_;
};

Script held_script(std::string name = "op") {
  Script script;
  script.name = std::move(name);
  return script;  // state->release stays false: pending until released
}

Script done_script(std::vector<uint8_t> finish_results = {},
                   std::string name = "op",
                   nixl_status_t terminal = NIXL_SUCCESS) {
  Script script;
  script.name = std::move(name);
  script.finish_results = std::move(finish_results);
  script.terminal_status = terminal;
  script.state->release.store(true);
  return script;
}

Script failed_load_script(int fallback_mode) {
  Script script = done_script({}, "first", NIXL_ERR_NOT_FOUND);
  script.fallback_mode = fallback_mode;
  return script;
}

void run_until_emits(Harness& harness, size_t expected) {
  harness.start();
  CHECK(wait_until([&] { return harness.emit_count() == expected; }));
  harness.stop_and_join();
}

void test_overlap_admission_and_exact_once() {
  Harness harness;
  harness.driver.set(1, held_script("held"));
  for (uint64_t id = 2; id <= 4; ++id) harness.driver.set(id, done_script());
  Script sync;
  sync.posts = false;
  harness.driver.set(5, sync);
  for (uint64_t id = 1; id <= 5; ++id) {
    const Op op = id == 5 ? Op::BATCH_TILE_EXISTS : Op::BATCH_TILE_SET;
    harness.push(make_request(id, op, 1, make_batch(op, 1), 0));
  }
  harness.start();
  CHECK(wait_until([&] { return harness.emit_count() == 4; }));
  CHECK(harness.driver.prepare_order() == std::vector<uint64_t>({1, 2, 3, 4}));
  CHECK(!harness.driver.state(1)->released.load());
  harness.driver.state(1)->release.store(true);
  CHECK(wait_until([&] { return harness.emit_count() == 5; }));
  harness.stop_and_join();
  for (uint64_t id = 1; id <= 5; ++id) CHECK(harness.emitted(id, true));
  CHECK(harness.driver.sync_calls() == 1);
  CHECK(!harness.driver.state(5)->posted.load());
  std::printf("PASS overlap_admission_and_exact_once\n");
}

void test_retry_slot_and_release_order() {
  Harness harness;
  Script first = held_script("first");
  first.terminal_status = NIXL_ERR_NOT_FOUND;
  first.retry_script =
      std::make_shared<Script>(done_script({0, 1, 0}, "retry"));
  first.retry_script->state->release.store(false);  // pending until released
  harness.driver.set(1, first);
  auto batch = make_batch(Op::BATCH_TILE_GET, 4);
  batch->per_key_results = {9, 9, 9, 9};
  harness.push(make_request(1, Op::BATCH_TILE_GET, 3, batch, 1));
  harness.start();
  CHECK(wait_until([&] { return harness.driver.state(1)->posted.load(); }));
  harness.driver.state(1)->release.store(true);
  CHECK(
      wait_until([&] { return harness.driver.retry_state(1)->posted.load(); }));
  CHECK(harness.driver.state(1)->live.load() == 0);
  CHECK(harness.driver.trace_index("first:released") <
        harness.driver.trace_index("fallback:first"));
  CHECK(harness.emit_count() == 0);  // the retry replaces the same slot
  harness.driver.retry_state(1)->release.store(true);
  CHECK(wait_until([&] { return harness.emit_count() == 1; }));
  harness.stop_and_join();
  CHECK(harness.emitted(1, true));
  CHECK(batch->per_key_results == std::vector<uint8_t>({9, 0, 1, 0}));
  CHECK(harness.driver.fallback_calls() == 1);
  std::printf("PASS retry_slot_and_release_order\n");
}

struct LoadCase {
  const char* name;
  Script first;
  bool expect_ok;
  std::vector<uint8_t> expect_bits;  // compared when expect_ok
  size_t expect_fallbacks;
};

void test_load_outcomes_table() {
  Script prepare_soft;
  prepare_soft.prepare_mode = 2;
  prepare_soft.soft_results = {0, 0};
  Script prepare_hard;
  prepare_hard.prepare_mode = 1;

  Script retry_failed = failed_load_script(kFallbackRetry);
  retry_failed.retry_script =
      std::make_shared<Script>(done_script({0, 0}, "retry", NIXL_ERR_BACKEND));
  Script all_present = failed_load_script(kFallbackSoft);
  all_present.fallback_soft_results = {0, 0};

  const LoadCase cases[] = {
      {"prepare-soft", prepare_soft, true, {0, 0}, 0},
      {"prepare-hard", prepare_hard, false, {}, 0},
      {"none-present-empty",
       failed_load_script(kFallbackEmpty),
       true,
       {0, 0},
       1},
      {"all-present-soft", all_present, true, {0, 0}, 1},
      {"query-error-hard",
       failed_load_script(kFallbackHardThrow),
       false,
       {},
       1},
      {"retry-failed-soft", retry_failed, true, {0, 0}, 1},
  };
  for (const LoadCase& test : cases) {
    Harness harness;
    harness.driver.set(1, test.first);
    auto batch = make_batch(Op::BATCH_TILE_GET, 2);
    batch->per_key_results = {9, 9};
    harness.push(make_request(1, Op::BATCH_TILE_GET, 2, batch, 0));
    run_until_emits(harness, 1);
    CHECK(harness.emitted(1, test.expect_ok));
    CHECK(batch->per_key_results ==
          (test.expect_ok ? test.expect_bits : std::vector<uint8_t>({9, 9})));
    CHECK(harness.driver.fallback_calls() == test.expect_fallbacks);
    std::printf("PASS load_outcomes[%s]\n", test.name);
  }
}

void test_post_throws() {
  {
    Harness harness;
    Script first = held_script("first");
    first.throw_on_post = true;
    first.retry_script = std::make_shared<Script>(done_script({0, 1}, "retry"));
    harness.driver.set(1, first);
    auto batch = make_batch(Op::BATCH_TILE_GET, 2);
    batch->per_key_results = {9, 9};
    harness.push(make_request(1, Op::BATCH_TILE_GET, 2, batch, 0));
    run_until_emits(harness, 1);
    CHECK(harness.emitted(1, true));
    CHECK(batch->per_key_results == std::vector<uint8_t>({0, 1}));
    CHECK(harness.driver.fallback_calls() == 1);
  }
  {
    Harness harness;
    Script store = done_script({}, "store");
    store.throw_on_post = true;
    harness.driver.set(1, store);
    harness.push(make_request(1, Op::BATCH_TILE_SET, 1,
                              make_batch(Op::BATCH_TILE_SET, 1), 0));
    run_until_emits(harness, 1);
    CHECK(harness.emitted(1, false));
    CHECK(harness.driver.fallback_calls() == 0);
  }
  std::printf("PASS post_throws\n");
}

void test_stop_drains_without_completion() {
  Harness harness;
  for (uint64_t id = 1; id <= 2; ++id) {
    harness.driver.set(id, held_script());
    harness.push(make_request(id, Op::BATCH_TILE_SET, 1,
                              make_batch(Op::BATCH_TILE_SET, 1), 0));
  }
  harness.driver.set(3, held_script());
  harness.start();
  CHECK(wait_until([&] {
    return harness.driver.state(1)->posted.load() &&
           harness.driver.state(2)->posted.load();
  }));
  harness.stop.store(true);
  harness.queue_cv.notify_all();
  harness.push(  // queued after stop: admission halted
      make_request(3, Op::BATCH_TILE_SET, 1, make_batch(Op::BATCH_TILE_SET, 1),
                   0));
  harness.driver.state(1)->release.store(true);
  harness.driver.state(2)->release.store(true);
  harness.join();
  CHECK(harness.emit_count() == 0);
  CHECK(harness.driver.state(1)->released.load());
  CHECK(harness.driver.state(2)->released.load());
  const auto order = harness.driver.prepare_order();
  CHECK(std::find(order.begin(), order.end(), 3) == order.end());
  std::printf("PASS stop_drains_without_completion\n");
}

void test_stop_before_fallback() {
  Harness harness;
  Script first = done_script({}, "first", NIXL_ERR_NOT_FOUND);
  first.stop_in_complete = true;  // stop begins during first-op release
  first.retry_script = std::make_shared<Script>(done_script({0}, "retry"));
  harness.driver.set(1, first);
  harness.push(make_request(1, Op::BATCH_TILE_GET, 1,
                            make_batch(Op::BATCH_TILE_GET, 1), 0));
  harness.start();
  harness.join();  // the run exits on its own after the stop is observed
  CHECK(harness.driver.fallback_calls() == 0);
  CHECK(harness.driver.retry_state(1)->post_calls.load() == 0);
  CHECK(harness.emit_count() == 0);
  std::printf("PASS stop_before_fallback\n");
}

void test_emit_throw_drains_survivor() {
  Harness harness;
  Script survivor = held_script("survivor");
  survivor.hold_polls = 2;  // stays posted past the throwing emit
  harness.driver.set(1, survivor);
  Script first = done_script({}, "first", NIXL_ERR_NOT_FOUND);
  Script retry = done_script({0, 1}, "retry");
  retry.post_status = NIXL_SUCCESS;  // completes without ever staying posted
  first.retry_script = std::make_shared<Script>(retry);
  harness.driver.set(2, first);
  harness.push(make_request(1, Op::BATCH_TILE_GET, 1,
                            make_batch(Op::BATCH_TILE_GET, 1), 0));
  harness.push(make_request(2, Op::BATCH_TILE_GET, 1,
                            make_batch(Op::BATCH_TILE_GET, 1), 0));
  harness.fail_next_emit();
  harness.start();
  CHECK(wait_until([&] { return harness.run_threw.load(); }));
  harness.join();
  CHECK(harness.emit_attempts() == 1);
  CHECK(harness.emit_count() == 0);
  CHECK(harness.driver.state(1)->released.load());  // drained in the catch
  CHECK(harness.driver.state(1)->live.load() == 0);
  CHECK(harness.driver.state(2)->live.load() == 0);
  CHECK(harness.driver.retry_state(2)->live.load() == 0);
  CHECK(harness.driver.fallback_calls() == 1);
  std::printf("PASS emit_throw_drains_survivor\n");
}

void test_reactor_eligibility_matrix() {
  NixlStorageCapabilities object_caps;
  object_caps.pending_transfers = true;
  NixlStorageCapabilities file_caps;
  CHECK(!file_caps.pending_transfers);

  CHECK(nixl_reactor_eligible("RADOS", object_caps, true));
  CHECK(nixl_reactor_eligible("REDIS", object_caps, true));
  CHECK(!nixl_reactor_eligible("RADOS", object_caps, false));
  CHECK(!nixl_reactor_eligible("REDIS", object_caps, false));
  CHECK(!nixl_reactor_eligible("OBJ", object_caps, true));
  CHECK(!nixl_reactor_eligible("S3", object_caps, true));
  CHECK(!nixl_reactor_eligible("rados", object_caps, true));
  CHECK(!nixl_reactor_eligible("RAD", object_caps, true));
  CHECK(!nixl_reactor_eligible("RADOS", file_caps, true));
  CHECK(!nixl_reactor_eligible("POSIX", file_caps, true));
  std::printf("PASS reactor_eligibility_matrix\n");
}

}  // namespace

int main() {
  test_overlap_admission_and_exact_once();
  test_retry_slot_and_release_order();
  test_load_outcomes_table();
  test_post_throws();
  test_stop_drains_without_completion();
  test_stop_before_fallback();
  test_emit_throw_drains_survivor();
  test_reactor_eligibility_matrix();
  std::printf("all reactor tests passed\n");
  return 0;
}
