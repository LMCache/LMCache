// SPDX-License-Identifier: Apache-2.0

#include "phase_timing_recorder.cuh"

#include "event_recorder.h"

#include <exception>

namespace {

// Destroy an event pair (either handle may be null) and clear the CUDA
// error state, so a timing failure never surfaces on an unrelated call.
void destroy_phase_timing_events(cudaEvent_t start, cudaEvent_t end) {
  if (start != nullptr) {
    cudaEventDestroy(start);
  }
  if (end != nullptr) {
    cudaEventDestroy(end);
  }
  (void)cudaGetLastError();
}

void destroy_anchor(PhaseTimingContext& ctx) {
  destroy_phase_timing_events(ctx.anchor, nullptr);
  ctx.anchor = nullptr;
}

// Drop one record's reference to its context; the last reference destroys
// the anchor event.
void release_context(PhaseTimingRecord& record) {
  if (record.ctx->pending_records.fetch_sub(1) == 1) {
    destroy_anchor(*record.ctx);
  }
  record.ctx.reset();
}

// Host callback enqueued right after the anchor event: stamps the wall
// clock. Owns and frees the shared_ptr copy it was handed.
void
#ifndef USE_ROCM
    CUDART_CB
#endif
    stamp_anchor_wall_time(void* data) {
  auto* holder = static_cast<std::shared_ptr<PhaseTimingContext>*>(data);
  (*holder)->anchor_wall_time_s.store(lmcache_wall_clock_time());
  delete holder;
}

// Record the anchor event and its wall-clock stamp on the stream. On
// failure the context keeps a null anchor and its samples carry no wall
// clock.
void open_anchor(cudaStream_t stream,
                 const std::shared_ptr<PhaseTimingContext>& ctx) {
  if (cudaEventCreate(&ctx->anchor) != cudaSuccess ||
      cudaEventRecord(ctx->anchor, stream) != cudaSuccess) {
    destroy_anchor(*ctx);
    return;
  }
  auto* holder = new std::shared_ptr<PhaseTimingContext>(ctx);
  if (LMCACHE_LAUNCH_HOST_FUNC(stream, stamp_anchor_wall_time, holder) !=
      cudaSuccess) {
    delete holder;
    destroy_anchor(*ctx);
  }
}

// Build the sample of a completed record. False if the elapsed time could
// not be read (the record is then dropped).
bool measure(const PhaseTimingRecord& record, PhaseTimingSample& sample) {
  float elapsed_ms = 0.0f;
  if (cudaEventElapsedTime(&elapsed_ms, record.start, record.end) !=
      cudaSuccess) {
    return false;
  }
  const PhaseTimingContext& ctx = *record.ctx;
  sample.phase = static_cast<int>(record.phase);
  sample.direction = ctx.direction;
  sample.device_index = ctx.device_index;
  sample.elapsed_ms = elapsed_ms;
  sample.nbytes = record.nbytes;
  sample.session_id = ctx.session_id;
  sample.start_time_s = 0.0;
  sample.end_time_s = 0.0;
  float offset_ms = 0.0f;
  if (ctx.anchor != nullptr &&
      cudaEventElapsedTime(&offset_ms, ctx.anchor, record.start) ==
          cudaSuccess) {
    sample.start_time_s = ctx.anchor_wall_time_s.load() + offset_ms / 1e3;
    sample.end_time_s = sample.start_time_s + elapsed_ms / 1e3;
  }
  return true;
}

}  // namespace

PhaseTimer::PhaseTimer(bool enabled, cudaStream_t stream, int direction,
                       int device_index, size_t max_sections,
                       std::string session_id)
    : enabled_(enabled), stream_(stream) {
  if (!enabled_) {
    return;
  }
  records_.reserve(max_sections);
  ctx_ = std::make_shared<PhaseTimingContext>();
  ctx_->session_id = std::move(session_id);
  ctx_->direction = direction;
  ctx_->device_index = device_index;
  open_anchor(stream, ctx_);
}

PhaseTimer::~PhaseTimer() {
  // Destroyed without commit(): the transfer failed or returned early, so
  // the collected samples are abandoned.
  if (committed_) {
    return;
  }
  for (auto& record : records_) {
    destroy_phase_timing_events(record.start, record.end);
  }
  if (ctx_) {
    destroy_anchor(*ctx_);
  }
}

void PhaseTimer::commit() {
  // Set the flag first: if push_batch throws, the destructor must not
  // destroy events whose copies may already sit in the recorder.
  committed_ = true;
  if (!enabled_) {
    return;
  }
  if (records_.empty()) {
    destroy_anchor(*ctx_);
    return;
  }
  ctx_->pending_records.store(static_cast<int>(records_.size()));
  PhaseTimingRecorder::instance().push_batch(records_);
}

void PhaseTimer::add(const PhaseTimingRecord& record) {
  // records_ was reserved for every possible section up front; the guard
  // keeps this path allocation-free and therefore unable to throw.
  if (records_.size() == records_.capacity()) {
    destroy_phase_timing_events(record.start, record.end);
    return;
  }
  records_.push_back(record);
}

PhaseTimer::Section::Section(PhaseTimer& timer, TransferPhase phase,
                             int64_t nbytes)
    : timer_(nullptr),
      start_(nullptr),
      end_(nullptr),
      phase_(phase),
      nbytes_(nbytes),
      base_exceptions_(std::uncaught_exceptions()) {
  // Stay inert (timer_ == nullptr) when timing is off or the section moves
  // no data.
  if (!timer.enabled_ || nbytes <= 0) {
    return;
  }

  // Open the interval: create the event pair and stamp its start onto the
  // transfer stream.
  const bool opened = cudaEventCreate(&start_) == cudaSuccess &&
                      cudaEventCreate(&end_) == cudaSuccess &&
                      cudaEventRecord(start_, timer.stream_) == cudaSuccess;
  if (!opened) {
    // Degrade to an inert section; the transfer itself is unaffected.
    destroy_phase_timing_events(start_, end_);
    start_ = nullptr;
    end_ = nullptr;
    return;
  }

  timer_ = &timer;
}

PhaseTimer::Section::~Section() {
  // Inert section: nothing was opened, nothing to close.
  if (timer_ == nullptr) {
    return;
  }

  // An exception is unwinding through the scope: the section did not run
  // to completion, so its sample would be garbage.
  if (std::uncaught_exceptions() > base_exceptions_) {
    destroy_phase_timing_events(start_, end_);
    return;
  }

  // Close the interval: stamp the end onto the stream. A never-recorded
  // event queries as "complete", so on failure the pair must be dropped
  // rather than kept as a bogus sample.
  if (cudaEventRecord(end_, timer_->stream_) != cudaSuccess) {
    destroy_phase_timing_events(start_, end_);
    return;
  }

  timer_->add({phase_, nbytes_, start_, end_, timer_->ctx_});
}

PhaseTimingRecorder& PhaseTimingRecorder::instance() {
  static PhaseTimingRecorder recorder;
  return recorder;
}

void PhaseTimingRecorder::push_batch(
    const std::vector<PhaseTimingRecord>& records) {
  if (records.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& record : records) {
    evict_until_below_cap(1);
    pending_.push_back(record);
  }
}

std::vector<PhaseTimingSample> PhaseTimingRecorder::pop_completed() {
  // Take the whole queue, then run every CUDA call unlocked so executor
  // pushes never wait.
  std::deque<PhaseTimingRecord> pending;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending.swap(pending_);
  }

  std::vector<PhaseTimingSample> samples;
  std::deque<PhaseTimingRecord> not_ready;
  for (auto& record : pending) {
    // Ready once the end event has completed and, with an anchor, its
    // wall-clock stamp has landed.
    const cudaError_t status = cudaEventQuery(record.end);
    const bool anchor_pending = record.ctx->anchor != nullptr &&
                                record.ctx->anchor_wall_time_s.load() == 0.0;
    if (status == cudaErrorNotReady ||
        (status == cudaSuccess && anchor_pending)) {
      (void)cudaGetLastError();  // clear cudaErrorNotReady
      not_ready.push_back(record);
      continue;
    }
    PhaseTimingSample sample;
    if (status == cudaSuccess && measure(record, sample)) {
      samples.push_back(std::move(sample));
    }
    destroy_phase_timing_events(record.start, record.end);
    release_context(record);
  }

  // Not-yet-completed records go back at the front: they predate anything
  // pushed while the queue was unlocked, keeping eviction oldest-first.
  if (!not_ready.empty()) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.insert(pending_.begin(), not_ready.begin(), not_ready.end());
    evict_until_below_cap(0);
  }
  return samples;
}

void PhaseTimingRecorder::evict_until_below_cap(size_t headroom) {
  while (pending_.size() + headroom > kMaxPending) {
    PhaseTimingRecord& oldest = pending_.front();
    destroy_phase_timing_events(oldest.start, oldest.end);
    release_context(oldest);
    pending_.pop_front();
  }
}

std::vector<PhaseTimingSample> pop_completed_phase_timings() {
  return PhaseTimingRecorder::instance().pop_completed();
}
