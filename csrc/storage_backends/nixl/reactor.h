// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
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

// Centralized reactor eligibility: explicit enablement and pending support.
inline bool nixl_reactor_eligible(const NixlStorageCapabilities& capabilities,
                                  bool enable_reactor) {
  return enable_reactor && capabilities.pending_transfers;
}

// Logical tile bytes: the sum of the per-key destination lengths.
inline size_t nixl_tile_bytes(const Request& request) {
  size_t total = 0;
  for (const size_t length : request.buf_lens) {
    if (length > std::numeric_limits<size_t>::max() - total) {
      throw std::overflow_error("NIXL reactor tile byte sum overflow");
    }
    total += length;
  }
  return total;
}

// One worker's request queue pair, registered for credit-release wakes.
struct ReactorWake {
  std::mutex* queue_mu;
  std::condition_variable* queue_cv;
};

// Connector-wide in-flight byte window shared by all reactor workers.
class ReactorWindow final {
 public:
  explicit ReactorWindow(size_t window_bytes) : window_(window_bytes) {}

  // Registers one running reactor's queue until it detaches.
  void attach(ReactorWake& wake) {
    std::lock_guard<std::mutex> lock(wakes_mu_);
    wakes_.push_back(&wake);
  }

  // Unregisters before the queue pair's lifetime can end.
  void detach(ReactorWake& wake) {
    std::lock_guard<std::mutex> lock(wakes_mu_);
    std::erase(wakes_, &wake);
  }

  // Never blocks: false means the caller keeps the tile queued.
  bool try_reserve(size_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!admissible_locked(bytes)) return false;
    if (bytes > std::numeric_limits<size_t>::max() - reserved_) {
      throw std::overflow_error("NIXL reactor window reservation overflow");
    }
    // The only permitted overshoot: an oversized tile from an empty window.
    if (window_ != 0 && reserved_ == 0 && bytes > window_ &&
        !oversized_warning_emitted_) {
      std::fprintf(stderr,
                   "[LMCache NIXL Reactor] admitting oversized tile alone: "
                   "%zu bytes > window %zu bytes\n",
                   bytes, window_);
      oversized_warning_emitted_ = true;
    }
    reserved_ += bytes;
    return true;
  }

  // Gate probe for wait predicates; reserves nothing.
  bool can_reserve(size_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    return admissible_locked(bytes);
  }

  // Returns credit, then allocation-free wakes every registered queue.
  void release(size_t bytes) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (bytes > reserved_) {
        throw std::underflow_error("NIXL reactor window credit underflow");
      }
      reserved_ -= bytes;
    }
    std::lock_guard<std::mutex> lock(wakes_mu_);
    for (const ReactorWake* wake : wakes_) {
      {
        std::lock_guard<std::mutex> queue_lock(*wake->queue_mu);
      }
      wake->queue_cv->notify_all();
    }
  }

 private:
  // A window of 0 is unrestricted; an oversized tile admits only alone.
  bool admissible_locked(size_t bytes) const {
    if (window_ == 0 || reserved_ == 0) return true;
    return reserved_ <= window_ && bytes <= window_ - reserved_;
  }

  size_t window_;
  size_t reserved_ = 0;
  bool oversized_warning_emitted_ = false;
  std::mutex mu_;
  std::mutex wakes_mu_;
  std::vector<ReactorWake*> wakes_;
};

// Internal scheduler for one single-threaded agent on pending-capable
// storage: every transition after post allocates nothing, and stop or an
// escaping exception drains posted tiles to terminal with completions
// suppressed. One run() per instance.
class CrossCallReactor {
 public:
  CrossCallReactor(NixlReactorDriver& driver, ReactorWindow& window)
      : driver_(driver), window_(window) {}

  void run(std::mutex& req_mu, std::condition_variable& req_cv,
           std::queue<Request>& requests, std::function<bool()> stopping,
           std::function<void(const Request&, const Completion&)> emit) {
    stopping_ = std::move(stopping);
    emit_ = std::move(emit);
    auto wake = std::make_unique<ReactorWake>(ReactorWake{&req_mu, &req_cv});
    // Both exits detach before the owned wake node dies.
    window_.attach(*wake);
    try {
      for (;;) {
        if (stopping_()) break;

        bool consumed = false;
        bool taken = false;
        bool posted_tile = false;
        Request candidate;
        size_t charge = 0;
        {
          std::unique_lock<std::mutex> lock(req_mu);
          if (!requests.empty()) {
            Request& front = requests.front();
            posted_tile = driver_.posts_tile(front);
            const size_t bytes = nixl_tile_bytes(front);
            // A gated posted tile stays queued: no reordering, no blocking.
            if (!posted_tile || window_.try_reserve(bytes)) {
              candidate = std::move(front);
              requests.pop();
              taken = true;
              charge = posted_tile ? bytes : 0;
            }
          }
        }
        if (taken) {
          if (stopping_()) {  // stop observed after dequeue: discard
            window_.release(charge);
            break;
          }
          admit(std::move(candidate), posted_tile, charge);
          consumed = true;
        }

        if (!active_.empty()) {
          const bool terminal = poll_one();
          if (!consumed && !terminal) std::this_thread::yield();
          continue;
        }
        if (consumed) continue;

        std::unique_lock<std::mutex> lock(req_mu);
        // A gated queue front sleeps here; a credit release wakes this wait.
        req_cv.wait(lock, [&] {
          if (stopping_() || requests.empty()) return true;
          const Request& front = requests.front();
          return !driver_.posts_tile(front) ||
                 window_.can_reserve(nixl_tile_bytes(front));
        });
      }
    } catch (...) {
      drain_active();
      window_.detach(*wake);
      throw;
    }
    drain_active();
    window_.detach(*wake);
  }

 private:
  struct Tile {
    Request request;
    std::unique_ptr<NixlPendingOperation> operation;
    // A failed load gets exactly one query-confirmed retry.
    bool fallback_used = false;
    // Window bytes held from activation until the tile finally leaves active.
    size_t charge_bytes = 0;
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

  // The caller's charge (0 for non-posted) transfers to the active node.
  void admit(Request request, bool posted, size_t charge) {
    if (!posted) {
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
      if (stopping_()) {  // posted nothing; the operation destructs
        window_.release(charge);
        return;
      }
    } catch (const NixlLoadPrepareError& prepared) {
      window_.release(charge);
      driver_.log_load_error(prepared.what());
      complete_load_tile(request, prepared.results);
      return;
    } catch (const std::exception& error) {
      window_.release(charge);
      fail_tile(request, error.what());
      return;
    }

    // Last allocation point, still before post.
    active_.emplace_back(
        Tile{std::move(request), std::move(operation), false, charge});
    const ActiveIt it = std::prev(active_.end());
    nixl_status_t posted_status = NIXL_SUCCESS;
    try {
      posted_status = it->operation->post();
    } catch (const std::exception& error) {
      if (it->request.op == Op::BATCH_TILE_GET) {
        handle_failed_load(it);
        return;
      }
      it->operation->discard();
      Request tile_request = std::move(it->request);
      const size_t bytes = it->charge_bytes;
      active_.erase(it);
      window_.release(bytes);
      fail_tile(tile_request, error.what());
      return;
    }

    if (posted_status == NIXL_IN_PROG) return;
    if (posted_status != NIXL_SUCCESS && !it->fallback_used &&
        it->request.op == Op::BATCH_TILE_GET) {
      handle_failed_load(it);
      return;
    }
    // Synchronous completion (incl empty loads).
    finish_terminal(it, posted_status);
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
    // The charge rides the node: a re-splice keeps it, every exit returns it.
    it->operation->discard();
    it->operation.reset();
    if (stopping_()) {
      const size_t bytes = it->charge_bytes;
      active_.erase(it);
      window_.release(bytes);
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
      if (stopping_()) {
        window_.release(tile.charge_bytes);
        return;
      }
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
    window_.release(tile.charge_bytes);
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
    const size_t bytes = it->charge_bytes;
    Request tile_request = std::move(it->request);
    active_.erase(it);
    window_.release(bytes);
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
      const size_t bytes = it->charge_bytes;
      active_.erase(it);
      window_.release(bytes);
    }
  }

  NixlReactorDriver& driver_;
  ReactorWindow& window_;
  std::list<Tile> active_;
  std::function<bool()> stopping_;
  std::function<void(const Request&, const Completion&)> emit_;
};

}  // namespace connector
}  // namespace lmcache
