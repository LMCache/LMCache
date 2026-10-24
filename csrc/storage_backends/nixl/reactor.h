// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <condition_variable>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../connector_types.h"
#include "storage.h"

namespace lmcache {
namespace connector {

class NixlReactorDriver {
 public:
  virtual ~NixlReactorDriver() = default;

  virtual bool posts_tile(const Request& request) const = 0;
  // Runs a non-posted tile inline; a throw fails the tile.
  virtual void execute_sync(const Request& request) = 0;
  // Validates and prepares before the tile becomes active. A generic throw
  // fails the tile; NixlLoadPrepareError completes a load tile successfully
  // with the carried (false) per-key results.
  virtual std::unique_ptr<NixlPendingOperation> prepare(
      const Request& request) = 0;
  // One-shot fallback after a failed first load transfer (the failed
  // operation is already released when this is called). A raw throw fails
  // the tile (hard query error); NixlLoadPrepareError completes it
  // successfully with false bits; otherwise returns the retry operation.
  virtual std::unique_ptr<NixlPendingOperation> prepare_load_fallback(
      const Request& request) = 0;
  // Records a load-path message that keeps the serial soft-failure
  // semantics (false bits, successful tile).
  virtual void log_load_error(const std::string& message) = 0;
};

// Centralized reactor eligibility: enable_reactor admits allowlisted backends
// only.
inline bool nixl_reactor_eligible(const std::string& backend,
                                  const NixlStorageCapabilities& capabilities,
                                  bool enable_reactor) {
  const bool allowlisted = backend == "RADOS" || backend == "REDIS";
  return enable_reactor && allowlisted && capabilities.pending_transfers;
}

// Internal scheduler for one single-threaded agent on pending-capable
// storage: every transition after post allocates nothing, and stop or an
// escaping exception drains posted tiles to terminal with completions
// suppressed. One run() per instance.
class CrossCallReactor {
 public:
  explicit CrossCallReactor(NixlReactorDriver& driver) : driver_(driver) {}

  void run(std::mutex& req_mu, std::condition_variable& req_cv,
           std::queue<Request>& requests, std::function<bool()> stopping,
           std::function<void(const Request&, const Completion&)> emit) {
    stopping_ = std::move(stopping);
    emit_ = std::move(emit);
    try {
      for (;;) {
        if (stopping_()) break;

        bool consumed = false;
        bool taken = false;
        Request candidate;
        {
          std::unique_lock<std::mutex> lock(req_mu);
          if (!requests.empty()) {
            candidate = std::move(requests.front());
            requests.pop();
            taken = true;
          }
        }
        if (taken) {
          if (stopping_()) break;  // stop observed after dequeue: discard
          admit(std::move(candidate));
          consumed = true;
        }

        if (!active_.empty()) {
          const bool terminal = poll_one();
          if (!consumed && !terminal) std::this_thread::yield();
          continue;
        }
        if (consumed) continue;

        std::unique_lock<std::mutex> lock(req_mu);
        req_cv.wait(lock, [&] { return stopping_() || !requests.empty(); });
      }
    } catch (...) {
      drain_active();
      throw;
    }
    drain_active();
  }

 private:
  struct Tile {
    Request request;
    std::unique_ptr<NixlPendingOperation> operation;
    // A failed load gets exactly one query-confirmed retry.
    bool fallback_used = false;
  };

  using ActiveIt = std::list<Tile>::iterator;

  void emit_tile(const Request& request, const Completion& completion) {
    if (stopping_()) return;  // shutdown suppresses completions
    emit_(request, completion);
  }

  void fail_tile(const Request& request, const std::string& error) {
    Completion completion;
    completion.future_id = request.future_id;
    completion.ok = false;
    completion.error = error;
    emit_tile(request, completion);
  }

  void complete_load_tile(const Request& request,
                          const std::vector<uint8_t>& results) {
    for (size_t index = 0; index < results.size(); ++index) {
      request.batch->per_key_results[request.start_idx + index] =
          results[index];
    }
    Completion completion;
    completion.future_id = request.future_id;
    completion.ok = true;
    emit_tile(request, completion);
  }

  void admit(Request request) {
    if (!driver_.posts_tile(request)) {
      Completion completion;
      completion.future_id = request.future_id;
      try {
        driver_.execute_sync(request);
        completion.ok = true;
      } catch (const std::exception& error) {
        completion.ok = false;
        completion.error = error.what();
      }
      emit_tile(request, completion);
      return;
    }

    // Preparation and its failures happen before any active node exists.
    std::unique_ptr<NixlPendingOperation> operation;
    try {
      operation = driver_.prepare(request);
      if (stopping_()) return;  // posted nothing; the operation destructs
    } catch (const NixlLoadPrepareError& prepared) {
      driver_.log_load_error(prepared.what());
      complete_load_tile(request, prepared.results);
      return;
    } catch (const std::exception& error) {
      fail_tile(request, error.what());
      return;
    }

    // Last allocation point, still before post.
    active_.emplace_back(Tile{std::move(request), std::move(operation), false});
    const ActiveIt it = std::prev(active_.end());
    nixl_status_t posted = NIXL_SUCCESS;
    try {
      posted = it->operation->post();
    } catch (const std::exception& error) {
      if (it->request.op == Op::BATCH_TILE_GET) {
        handle_failed_load(it);
        return;
      }
      it->operation->discard();
      Request tile_request = std::move(it->request);
      active_.erase(it);
      fail_tile(tile_request, error.what());
      return;
    }

    if (posted == NIXL_IN_PROG) return;
    if (posted != NIXL_SUCCESS && !it->fallback_used &&
        it->request.op == Op::BATCH_TILE_GET) {
      handle_failed_load(it);
      return;
    }
    finish_terminal(it, posted);  // synchronous completion (incl empty loads)
  }

  std::string terminal_failure_message(nixl_status_t status) {
    return "NIXL transfer progress failed: " +
           nixlEnumStrings::statusStr(status);
  }

  // Polls one tile (round-robin order); returns true once it is terminal.
  bool poll_one() {
    const ActiveIt it = active_.begin();
    const nixl_status_t status = it->operation->poll();
    if (status == NIXL_IN_PROG) {
      // Allocation-free rotation: the live operation never leaves its node.
      active_.splice(active_.end(), active_, it);
      return false;
    }
    if (status != NIXL_SUCCESS && it->request.op == Op::BATCH_TILE_GET &&
        !it->fallback_used) {
      handle_failed_load(it);
      return true;
    }
    finish_terminal(it, status);
    return true;
  }

  // The tile is spliced out of the active list before anything can throw:
  // completions never run under a handler holding an active iterator, and
  // a node without an operation never survives into a drain. Stop
  // suppresses the fallback.
  void handle_failed_load(ActiveIt it) {
    it->operation->discard();
    it->operation.reset();
    if (stopping_()) {
      active_.erase(it);
      return;
    }
    std::list<Tile> detached;
    detached.splice(detached.end(), active_, it);
    Tile& tile = detached.front();

    bool hard_failure = false;
    std::vector<uint8_t> results;
    std::string error;
    std::unique_ptr<NixlPendingOperation> retry;
    try {
      retry = driver_.prepare_load_fallback(tile.request);
      if (stopping_()) return;
      const nixl_status_t posted = retry->post();
      if (posted == NIXL_IN_PROG) {
        tile.operation = std::move(retry);
        tile.fallback_used = true;
        active_.splice(active_.end(), detached, detached.begin());
        return;
      }
      results = retry->complete();  // moves: no alloc
    } catch (const NixlLoadPrepareError& soft) {
      driver_.log_load_error(soft.what());
      results = soft.results;
    } catch (const std::exception& failure) {
      if (retry) {
        // Retry preparation posted nothing; its bits stay false.
        driver_.log_load_error(failure.what());
        results = retry->complete();  // moves: no alloc
      } else {
        hard_failure = true;
        error = failure.what();
      }
    }
    if (hard_failure) {
      fail_tile(tile.request, error);
    } else {
      complete_load_tile(tile.request, results);
    }
  }

  // Single terminal path: releases the operation, unnodes the tile before
  // emitting, then emits exactly one completion unless stop has begun.
  void finish_terminal(ActiveIt it, nixl_status_t status) {
    const std::vector<uint8_t> results = it->operation->complete();
    const Op op = it->request.op;
    Request tile_request = std::move(it->request);
    active_.erase(it);
    if (op == Op::BATCH_TILE_GET) {
      if (status != NIXL_SUCCESS) {
        driver_.log_load_error(terminal_failure_message(status));
      }
      complete_load_tile(tile_request, results);
      return;
    }
    if (status == NIXL_SUCCESS) {
      Completion completion;
      completion.future_id = tile_request.future_id;
      completion.ok = true;
      emit_tile(tile_request, completion);
      return;
    }
    fail_tile(tile_request, terminal_failure_message(status));
  }

  // Drain for posted operations: poll, rotate, release, and unnode every
  // active tile without allocating. Every node owns a non-null operation.
  void drain_active() {
    while (!active_.empty()) {
      const ActiveIt it = active_.begin();
      if (it->operation->poll() == NIXL_IN_PROG) {
        active_.splice(active_.end(), active_, it);
        std::this_thread::yield();
        continue;
      }
      it->operation->discard();
      active_.erase(it);
    }
  }

  NixlReactorDriver& driver_;
  std::list<Tile> active_;
  std::function<bool()> stopping_;
  std::function<void(const Request&, const Completion&)> emit_;
};

}  // namespace connector
}  // namespace lmcache
