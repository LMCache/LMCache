use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const SHARD_COUNT: usize = 32;
const READ_ATTEMPTS: usize = 0;
const WRITE_ATTEMPTS: usize = 1;
const READ_BYTES: usize = 2;
const WRITE_BYTES: usize = 3;
const COMPLETED: usize = 4;
const FAILED: usize = 5;
const BOUNCE_ATTEMPTS: usize = 6;
const BOUNCE_BYTES: usize = 7;
const FIXED_ATTEMPTS: usize = 8;
const FIXED_BYTES: usize = 9;
const COUNTER_NAMES: [&str; 10] = [
    "read_attempts",
    "write_attempts",
    "read_submitted_bytes",
    "write_submitted_bytes",
    "completed_attempts",
    "failed_attempts",
    "bounce_attempts",
    "bounce_submitted_bytes",
    "fixed_buffer_attempts",
    "fixed_buffer_submitted_bytes",
];

static NEXT_THREAD_SHARD: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static THREAD_SHARD: usize = NEXT_THREAD_SHARD.fetch_add(1, Ordering::Relaxed) % SHARD_COUNT;
}

fn observe_max(counter: &AtomicU64, value: u64) {
    if value > counter.load(Ordering::Relaxed) {
        counter.fetch_max(value, Ordering::Relaxed);
    }
}

#[repr(align(128))]
#[derive(Default)]
struct CounterShard {
    counters: [AtomicU64; 10],
    peak_outstanding: AtomicU64,
    peak_queued: AtomicU64,
    queue_full: AtomicU64,
}

#[repr(align(128))]
#[derive(Default)]
struct QueueGauge(AtomicU64);

/// Preallocated observation shards; lifecycle synchronization lives elsewhere.
pub(crate) struct RawBlockIoStats {
    shards: [CounterShard; SHARD_COUNT],
    worker: CounterShard,
    worker_claimed: AtomicBool,
    queue: QueueGauge,
}

impl Default for RawBlockIoStats {
    fn default() -> Self {
        Self {
            shards: std::array::from_fn(|_| CounterShard::default()),
            worker: CounterShard::default(),
            worker_claimed: AtomicBool::new(false),
            queue: QueueGauge::default(),
        }
    }
}

impl RawBlockIoStats {
    /// Borrow this thread's preallocated shard, without a lock or allocation.
    ///
    /// Thread IDs are striped; collisions remain correct through atomic adds.
    pub(crate) fn recorder(&self) -> IoStatsRecorder<'_> {
        IoStatsRecorder {
            shard: &self.shards[THREAD_SHARD.with(|index| *index)],
            queue: &self.queue,
            local: None,
        }
    }

    /// Claim a single-writer recorder for the lifetime of the io_uring worker.
    ///
    /// A repeated claim falls back to a shared shard, never changing I/O results.
    /// The single-writer handle is movable but not Sync or Clone.
    pub(crate) fn worker_recorder(&self) -> IoStatsRecorder<'_> {
        if self.worker_claimed.swap(true, Ordering::Relaxed) {
            return self.recorder();
        }
        IoStatsRecorder {
            shard: &self.worker,
            queue: &self.queue,
            local: Some(std::array::from_fn(|_| Cell::new(0))),
        }
    }

    /// Aggregate cumulative attempts/bytes and maxima at snapshot time.
    ///
    /// Shards survive producer exit until device destruction. Atomic loads make
    /// concurrent polling safe; cross-field snapshots remain best-effort.
    /// Outstanding is the caller's lifecycle count, including queued requests.
    pub(crate) fn snapshot(&self, outstanding: u64) -> HashMap<&'static str, u64> {
        let mut totals = [0u64; 10];
        let mut peak_outstanding = 0;
        let mut peak_queued = 0;
        let mut queue_full = 0u64;
        for shard in self.shards.iter().chain(std::iter::once(&self.worker)) {
            for (total, counter) in totals.iter_mut().zip(&shard.counters) {
                *total = total.wrapping_add(counter.load(Ordering::Relaxed));
            }
            peak_outstanding = peak_outstanding.max(shard.peak_outstanding.load(Ordering::Relaxed));
            peak_queued = peak_queued.max(shard.peak_queued.load(Ordering::Relaxed));
            queue_full = queue_full.wrapping_add(shard.queue_full.load(Ordering::Relaxed));
        }
        let mut result: HashMap<_, _> = COUNTER_NAMES.into_iter().zip(totals).collect();
        result.extend([
            ("outstanding_requests", outstanding),
            ("peak_outstanding_requests", peak_outstanding),
            ("queued_requests", self.queue.0.load(Ordering::Relaxed)),
            ("peak_queued_requests", peak_queued),
            ("queue_full_events", queue_full),
        ]);
        result
    }
}

/// Writer-local state with atomic publication for asynchronous snapshots.
pub(crate) struct IoStatsRecorder<'stats> {
    shard: &'stats CounterShard,
    queue: &'stats QueueGauge,
    local: Option<[Cell<u64>; 10]>,
}

impl IoStatsRecorder<'_> {
    /// Record one accepted syscall/SQE attempt and its requested length.
    /// Retries are separate attempts; buffer flags refer to this attempt only.
    pub(crate) fn submit(&self, write: bool, bytes: usize, bounce: bool, fixed: bool) {
        self.add(if write { WRITE_ATTEMPTS } else { READ_ATTEMPTS }, 1);
        self.add(if write { WRITE_BYTES } else { READ_BYTES }, bytes as u64);
        if bounce {
            self.add(BOUNCE_ATTEMPTS, 1);
            self.add(BOUNCE_BYTES, bytes as u64);
        }
        if fixed {
            self.add(FIXED_ATTEMPTS, 1);
            self.add(FIXED_BYTES, bytes as u64);
        }
    }

    /// Terminate exactly one submitted attempt, including short completions.
    pub(crate) fn complete(&self, success: bool) {
        self.add(if success { COMPLETED } else { FAILED }, 1);
    }

    /// Observe a lifecycle count without altering synchronization state.
    pub(crate) fn observe_outstanding(&self, count: u64) {
        observe_max(&self.shard.peak_outstanding, count);
    }

    /// Publish exact queue length while holding the caller's existing queue lock.
    pub(crate) fn set_queued(&self, count: usize) {
        self.queue.0.store(count as u64, Ordering::Relaxed);
        observe_max(&self.shard.peak_queued, count as u64);
    }

    /// Record a capacity-limited submission event, not a signal interruption.
    pub(crate) fn queue_full(&self) {
        self.shard.queue_full.fetch_add(1, Ordering::Relaxed);
    }

    fn add(&self, index: usize, value: u64) {
        if let Some(local) = &self.local {
            let total = local[index].get().wrapping_add(value);
            local[index].set(total);
            self.shard.counters[index].store(total, Ordering::Relaxed);
        } else {
            self.shard.counters[index].fetch_add(value, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RawBlockIoStats, SHARD_COUNT};
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn concurrent_cumulative_counters_survive_thread_exit_and_shard_collisions() {
        let stats = Arc::new(RawBlockIoStats::default());
        let barrier = Arc::new(Barrier::new(SHARD_COUNT + 9));
        let workers: Vec<_> = (0..SHARD_COUNT + 8)
            .map(|_| {
                let stats = Arc::clone(&stats);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let recorder = stats.recorder();
                    barrier.wait();
                    for _ in 0..1000 {
                        recorder.submit(true, 4096, true, false);
                        recorder.complete(true);
                        recorder.submit(false, 512, false, true);
                        recorder.complete(false);
                    }
                })
            })
            .collect();
        barrier.wait();
        let mut previous = 0;
        for _ in 0..100 {
            let current = stats.snapshot(0)["write_attempts"];
            assert!(current >= previous);
            previous = current;
        }
        for worker in workers {
            worker.join().unwrap();
        }
        let snapshot = stats.snapshot(0);
        let expected = (SHARD_COUNT as u64 + 8) * 1000;
        assert_eq!(snapshot["read_attempts"], expected);
        assert_eq!(snapshot["write_attempts"], expected);
        assert_eq!(snapshot["completed_attempts"], expected);
        assert_eq!(snapshot["failed_attempts"], expected);
        assert_eq!(snapshot["bounce_submitted_bytes"], expected * 4096);
        assert_eq!(snapshot["fixed_buffer_submitted_bytes"], expected * 512);
    }

    #[test]
    fn gauges_preserve_lifetime_peaks() {
        let stats = RawBlockIoStats::default();
        let recorder = stats.recorder();
        recorder.set_queued(10);
        recorder.set_queued(3);
        recorder.set_queued(0);
        recorder.observe_outstanding(12);
        recorder.observe_outstanding(2);
        recorder.queue_full();
        let snapshot = stats.snapshot(0);
        assert_eq!(snapshot["queued_requests"], 0);
        assert_eq!(snapshot["peak_queued_requests"], 10);
        assert_eq!(snapshot["peak_outstanding_requests"], 12);
        assert_eq!(snapshot["queue_full_events"], 1);
        assert_eq!(snapshot, stats.snapshot(0));
    }

    #[test]
    fn worker_publication_and_repeated_claim_are_lossless() {
        let stats = Arc::new(RawBlockIoStats::default());
        let reader = Arc::clone(&stats);
        let polling = thread::spawn(move || {
            let mut previous = 0;
            while previous < 10000 {
                let current = reader.snapshot(0)["completed_attempts"];
                assert!(current >= previous);
                previous = current;
                thread::yield_now();
            }
        });
        let recorder = stats.worker_recorder();
        for _ in 0..10000 {
            recorder.submit(false, 4096, true, false);
            recorder.complete(true);
        }
        polling.join().unwrap();
        drop(recorder);
        let next = stats.worker_recorder();
        next.submit(false, 512, false, false);
        next.complete(false);
        let snapshot = stats.snapshot(0);
        assert_eq!(snapshot["read_attempts"], 10001);
        assert_eq!(snapshot["read_submitted_bytes"], 10000 * 4096 + 512);
        assert_eq!(snapshot["completed_attempts"], 10000);
        assert_eq!(snapshot["failed_attempts"], 1);
        assert_eq!(RawBlockIoStats::default().snapshot(0)["read_attempts"], 0);
    }

    #[test]
    fn snapshots_combine_worker_and_producer_counters_without_mixing_devices() {
        let stats = Arc::new(RawBlockIoStats::default());
        let producer_stats = Arc::clone(&stats);
        let producer = thread::spawn(move || {
            let recorder = producer_stats.recorder();
            recorder.observe_outstanding(16);
            recorder.set_queued(12);
            for _ in 0..1000 {
                recorder.submit(true, 512, false, true);
                recorder.complete(true);
            }
        });
        let recorder = stats.worker_recorder();
        recorder.observe_outstanding(8);
        for _ in 0..1000 {
            recorder.submit(false, 4096, true, false);
            recorder.complete(true);
        }
        producer.join().unwrap();
        recorder.set_queued(0);
        let snapshot = stats.snapshot(0);
        assert_eq!(snapshot["completed_attempts"], 2000);
        assert_eq!(snapshot["peak_outstanding_requests"], 16);
        assert_eq!(snapshot["peak_queued_requests"], 12);
        assert_eq!(snapshot["queued_requests"], 0);
        assert_eq!(snapshot["bounce_attempts"], 1000);
        assert_eq!(snapshot["fixed_buffer_attempts"], 1000);
        let other = RawBlockIoStats::default();
        other.recorder().submit(true, 64, false, false);
        assert_eq!(other.snapshot(0)["write_attempts"], 1);
        assert_eq!(stats.snapshot(0)["write_attempts"], 1000);
    }
}
