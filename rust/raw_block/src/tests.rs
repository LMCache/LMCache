// SPDX-License-Identifier: Apache-2.0

use super::{
    build_fixed_buffer_regions, check_nvme_ioctl_result, fail_submissions, placement_id_to_u16,
    prepare_iouring_write_buffer, record_submission_result, resolve_fixed_buffer_idx,
    FixedBufferRegion, RawBlockDevice, SubmissionRetry, UringNotify, MAX_FIXED_BUFFER_REGION_SIZE,
    SUBMISSION_RETRY_INITIAL_DELAY, SUBMISSION_RETRY_MAX_DELAY, SUBMISSION_STALL_TIMEOUT,
};
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

enum ShutdownMethod {
    Close,
    Drop,
}

enum ShutdownWait {
    Drain,
    Join,
}

fn assert_shutdown_releases_gil(method: ShutdownMethod, wait: ShutdownWait) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut device = RawBlockDevice::new_internal(
            "/dev/null".to_string(),
            true,
            false,
            4096,
            false,
            false,
            Some("posix".to_string()),
            1,
        )
        .unwrap();
        device.use_iouring = true;
        device.queue = Some(Arc::default());
        device.shutdown = Some(Arc::new(AtomicBool::new(false)));
        device.in_flight_count.store(
            match wait {
                ShutdownWait::Drain => 1,
                ShutdownWait::Join => 0,
            },
            Ordering::Relaxed,
        );

        let in_flight_count = Arc::clone(&device.in_flight_count);
        let in_flight_cvar = Arc::clone(&device.in_flight_cvar);
        let python_progress = Arc::new(AtomicBool::new(false));
        let worker_progress = Arc::clone(&python_progress);
        let (progress_sender, progress_receiver) = mpsc::channel();
        device.worker = Some(thread::spawn(move || {
            let progressed = progress_receiver
                .recv_timeout(Duration::from_secs(5))
                .is_ok();
            worker_progress.store(progressed, Ordering::Relaxed);
            in_flight_count.store(0, Ordering::Relaxed);
            in_flight_cvar.notify_all();
        }));

        let device = Bound::new(py, device).unwrap();
        let python_thread = thread::spawn(move || {
            Python::with_gil(|_| {
                let _ = progress_sender.send(());
            });
        });

        let progressed_before_shutdown_returned = match method {
            ShutdownMethod::Close => {
                device.call_method0("close").unwrap();
                let progressed = python_progress.load(Ordering::Relaxed);
                device.call_method0("close").unwrap();
                drop(device);
                progressed
            }
            ShutdownMethod::Drop => {
                drop(device);
                python_progress.load(Ordering::Relaxed)
            }
        };
        py.allow_threads(|| python_thread.join()).unwrap();
        assert!(
            progressed_before_shutdown_returned,
            "another Python thread could not acquire the GIL during shutdown"
        );
    });
}

#[test]
fn check_nvme_ioctl_result_accepts_success() {
    assert!(check_nvme_ioctl_result(0, "NVMe ioctl failed").is_ok());
}

#[test]
fn check_nvme_ioctl_result_rejects_nvme_status() {
    assert!(check_nvme_ioctl_result(1, "NVMe ioctl failed").is_err());
}

#[test]
fn placement_id_to_u16_accepts_valid_bounds() {
    assert_eq!(placement_id_to_u16(1).unwrap(), 1);
    assert_eq!(placement_id_to_u16(65535).unwrap(), 65535);
}

#[test]
fn placement_id_to_u16_rejects_reserved_and_out_of_range_values() {
    assert!(placement_id_to_u16(0).is_err());
    assert!(placement_id_to_u16(-1).is_err());
    assert!(placement_id_to_u16(65536).is_err());
}

#[test]
fn prepare_iouring_write_buffer_keeps_fixed_buffer_for_zero_tail() {
    let mut buf = vec![0u8; 4096];
    buf[..4].copy_from_slice(b"data");
    let ptr = buf.as_ptr() as usize;

    let prepared = prepare_iouring_write_buffer(ptr, 4096, 4, 4096, false, 4096, Some(3)).unwrap();

    assert_eq!(prepared.ptr_addr, ptr);
    assert!(prepared.bounce.is_none());
    assert_eq!(prepared.fixed_buffer_idx, Some(3));
}

#[test]
fn close_releases_gil_while_draining() {
    assert_shutdown_releases_gil(ShutdownMethod::Close, ShutdownWait::Drain);
}

#[test]
fn close_releases_gil_while_joining_worker() {
    assert_shutdown_releases_gil(ShutdownMethod::Close, ShutdownWait::Join);
}

#[test]
fn drop_releases_gil_while_draining() {
    assert_shutdown_releases_gil(ShutdownMethod::Drop, ShutdownWait::Drain);
}

#[test]
fn drop_releases_gil_while_joining_worker() {
    assert_shutdown_releases_gil(ShutdownMethod::Drop, ShutdownWait::Join);
}

#[test]
fn recoverable_submission_errors_preserve_pending_until_retry_succeeds() {
    for error_code in [libc::EAGAIN, libc::EINTR, libc::EBUSY] {
        let mut pending = VecDeque::from([10, 11, 12]);
        let mut retry = SubmissionRetry::default();
        let now = Instant::now();
        let result =
            record_submission_result(&mut pending, Err(io::Error::from_raw_os_error(error_code)));
        retry.record_result(result, now).unwrap();
        assert_eq!(pending, VecDeque::from([10, 11, 12]));
        assert_eq!(
            retry.remaining_delay(now),
            Some(SUBMISSION_RETRY_INITIAL_DELAY)
        );

        let result = record_submission_result(&mut pending, Ok(1));
        retry
            .record_result(result, now + SUBMISSION_RETRY_INITIAL_DELAY)
            .unwrap();
        assert_eq!(pending, VecDeque::from([11, 12]));
        assert!(retry.remaining_delay(now).is_none());

        let result = record_submission_result(&mut pending, Ok(2));
        retry
            .record_result(result, now + SUBMISSION_RETRY_INITIAL_DELAY)
            .unwrap();
        assert!(pending.is_empty());
    }
}

#[test]
fn submission_retry_uses_capped_exponential_backoff() {
    let mut retry = SubmissionRetry::default();
    let mut now = Instant::now();
    for delay_ms in [1, 2, 4, 8, 16, 32, 64, 100, 100] {
        retry
            .record_result(Err(io::Error::from_raw_os_error(libc::EBUSY)), now)
            .unwrap();
        let delay = Duration::from_millis(delay_ms);
        assert_eq!(retry.remaining_delay(now), Some(delay));
        assert!(delay <= SUBMISSION_RETRY_MAX_DELAY);
        assert_eq!(retry.remaining_delay(now + delay / 2), Some(delay / 2));
        now += delay;
        assert!(retry.remaining_delay(now).is_none());
    }
}

#[test]
fn sustained_recoverable_submission_errors_become_terminal() {
    for error_code in [libc::EAGAIN, libc::EINTR, libc::EBUSY] {
        let mut retry = SubmissionRetry::default();
        let now = Instant::now();
        retry
            .record_result(Err(io::Error::from_raw_os_error(error_code)), now)
            .unwrap();
        let error = retry
            .record_result(
                Err(io::Error::from_raw_os_error(error_code)),
                now + SUBMISSION_STALL_TIMEOUT,
            )
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(error
            .to_string()
            .contains(&format!("os error {error_code}")));
    }
}

#[test]
fn zero_progress_submission_retries_are_bounded() {
    let mut retry = SubmissionRetry::default();
    let now = Instant::now();
    retry.record_result(Ok(0), now).unwrap();
    assert_eq!(
        retry.remaining_delay(now),
        Some(SUBMISSION_RETRY_INITIAL_DELAY)
    );
    let error = retry
        .record_result(Ok(0), now + SUBMISSION_STALL_TIMEOUT)
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert!(error.to_string().contains("accepted no requests"));
}

#[test]
fn completion_progress_resets_the_retry_delay_and_failure_budget() {
    let mut retry = SubmissionRetry::default();
    let now = Instant::now();
    retry
        .record_result(Err(io::Error::from_raw_os_error(libc::EBUSY)), now)
        .unwrap();
    retry.reset();
    assert!(retry.remaining_delay(now).is_none());
    let resumed = now + SUBMISSION_STALL_TIMEOUT;
    retry
        .record_result(Err(io::Error::from_raw_os_error(libc::EBUSY)), resumed)
        .unwrap();
    assert_eq!(
        retry.remaining_delay(resumed),
        Some(SUBMISSION_RETRY_INITIAL_DELAY)
    );
}

#[test]
fn permanent_submission_errors_are_not_retried() {
    for error_code in [libc::EBADF, libc::EINVAL, libc::EIO] {
        let mut retry = SubmissionRetry::default();
        let now = Instant::now();
        let error = retry
            .record_result(Err(io::Error::from_raw_os_error(error_code)), now)
            .unwrap_err();
        assert_eq!(error.raw_os_error(), Some(error_code));
        assert!(retry.remaining_delay(now).is_none());
    }
}

#[test]
fn retry_wait_times_out_without_an_event() {
    let notify = Arc::new(UringNotify::new().unwrap());
    let waiting_notify = Arc::clone(&notify);
    let (finished_sender, finished_receiver) = mpsc::channel();
    let worker = thread::spawn(move || {
        waiting_notify.wait(Some(Duration::from_millis(1)));
        finished_sender.send(()).unwrap();
    });
    let returned = finished_receiver
        .recv_timeout(Duration::from_secs(2))
        .is_ok();
    if !returned {
        notify.signal_producer();
    }
    worker.join().unwrap();
    assert!(returned);
}

#[test]
fn retry_wait_can_be_interrupted_by_shutdown_notification() {
    let notify = Arc::new(UringNotify::new().unwrap());
    let waiting_notify = Arc::clone(&notify);
    let (finished_sender, finished_receiver) = mpsc::channel();
    let worker = thread::spawn(move || {
        waiting_notify.wait(Some(Duration::from_secs(3)));
        finished_sender.send(()).unwrap();
    });
    notify.signal_producer();
    let returned = finished_receiver
        .recv_timeout(Duration::from_secs(2))
        .is_ok();
    worker.join().unwrap();
    assert!(returned);
}

#[test]
fn terminal_worker_failure_is_visible_to_python_before_cleanup() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut device = RawBlockDevice::new_internal(
            "/dev/null".to_string(),
            true,
            false,
            4096,
            false,
            false,
            Some("posix".to_string()),
            1,
        )
        .unwrap();
        assert!(device.worker_error().is_none());
        let queue = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        fail_submissions(
            &queue,
            &shutdown,
            &device.worker_error,
            io::Error::from_raw_os_error(libc::EIO),
        );
        device.use_iouring = true;
        device.queue = Some(queue);
        device.shutdown = Some(shutdown);
        let device = Bound::new(py, device).unwrap();
        let error = device
            .call_method0("worker_error")
            .unwrap()
            .extract::<String>()
            .unwrap();
        assert!(error.contains("io_uring worker submission failed"));
        assert!(error.contains(&format!("os error {}", libc::EIO)));
        for method in ["batched_write", "batched_read"] {
            let error = device
                .call_method1(
                    method,
                    (Vec::<u64>::new(), Vec::<u64>::new(), Vec::<usize>::new()),
                )
                .unwrap_err();
            assert!(error.to_string().contains("io_uring worker stopped"));
        }
        device.call_method0("close").unwrap();
        assert_eq!(
            device
                .call_method0("worker_error")
                .unwrap()
                .extract::<String>()
                .unwrap(),
            error,
        );
    });
}

#[test]
fn normal_shutdown_does_not_report_a_terminal_worker_error() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let device = RawBlockDevice::new_internal(
            "/dev/null".to_string(),
            true,
            false,
            4096,
            false,
            false,
            Some("posix".to_string()),
            1,
        )
        .unwrap();
        let device = Bound::new(py, device).unwrap();
        assert!(device.call_method0("worker_error").unwrap().is_none());
        device.call_method0("close").unwrap();
        assert!(device.call_method0("worker_error").unwrap().is_none());
    });
}

#[test]
fn resolve_fixed_buffer_idx_accepts_complete_subranges() {
    let regions = [FixedBufferRegion {
        start: 100,
        end: 200,
        index: 7,
    }];

    assert_eq!(resolve_fixed_buffer_idx(&regions, 100, 100), Some(7));
    assert_eq!(resolve_fixed_buffer_idx(&regions, 125, 50), Some(7));
    assert_eq!(resolve_fixed_buffer_idx(&regions, 199, 1), Some(7));
}

#[test]
fn resolve_fixed_buffer_idx_rejects_uncovered_and_invalid_ranges() {
    let regions = [FixedBufferRegion {
        start: 100,
        end: 200,
        index: 7,
    }];

    assert_eq!(resolve_fixed_buffer_idx(&regions, 99, 1), None);
    assert_eq!(resolve_fixed_buffer_idx(&regions, 200, 1), None);
    assert_eq!(resolve_fixed_buffer_idx(&regions, 150, 51), None);
    assert_eq!(resolve_fixed_buffer_idx(&regions, 150, 0), None);
    assert_eq!(resolve_fixed_buffer_idx(&regions, usize::MAX, 2), None);
}

#[test]
fn resolve_fixed_buffer_idx_rejects_cross_region_requests() {
    let regions = [
        FixedBufferRegion {
            start: 100,
            end: 200,
            index: 3,
        },
        FixedBufferRegion {
            start: 200,
            end: 300,
            index: 9,
        },
    ];

    assert_eq!(resolve_fixed_buffer_idx(&regions, 199, 2), None);
    assert_eq!(resolve_fixed_buffer_idx(&regions, 200, 1), Some(9));
}

#[test]
fn resolve_fixed_buffer_idx_preserves_original_index_after_sort() {
    let mut regions = vec![
        FixedBufferRegion {
            start: 200,
            end: 300,
            index: 0,
        },
        FixedBufferRegion {
            start: 100,
            end: 200,
            index: 1,
        },
    ];
    regions.sort_unstable_by_key(|region| region.start);

    assert_eq!(resolve_fixed_buffer_idx(&regions, 100, 1), Some(1));
    assert_eq!(resolve_fixed_buffer_idx(&regions, 200, 1), Some(0));
}

#[test]
fn build_fixed_buffer_regions_accepts_adjacent_unsorted_regions() {
    let regions = build_fixed_buffer_regions(&[200, 100], &[100, 100]).unwrap();

    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].start, 100);
    assert_eq!(regions[0].end, 200);
    assert_eq!(regions[0].index, 1);
    assert_eq!(regions[1].start, 200);
    assert_eq!(regions[1].end, 300);
    assert_eq!(regions[1].index, 0);
}

#[test]
fn build_fixed_buffer_regions_rejects_invalid_inputs() {
    assert!(build_fixed_buffer_regions(&[100], &[]).is_err());
    assert!(build_fixed_buffer_regions(&[], &[]).is_err());
    assert!(build_fixed_buffer_regions(&[0], &[1]).is_err());
    assert!(build_fixed_buffer_regions(&[100], &[0]).is_err());
    assert!(build_fixed_buffer_regions(&[100], &[MAX_FIXED_BUFFER_REGION_SIZE + 1]).is_err());
    assert!(build_fixed_buffer_regions(&[usize::MAX], &[2]).is_err());
    assert!(build_fixed_buffer_regions(&[100, 150], &[100, 100]).is_err());

    let too_many = usize::from(u16::MAX) + 2;
    assert!(build_fixed_buffer_regions(&vec![1; too_many], &vec![1; too_many]).is_err());
}
