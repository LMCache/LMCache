// SPDX-License-Identifier: Apache-2.0

use super::{
    check_nvme_ioctl_result, fail_submissions, placement_id_to_u16, prepare_iouring_write_buffer,
    record_submission_result, RawBlockDevice, SubmissionRetry, UringNotify,
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

    let prepared =
        prepare_iouring_write_buffer(ptr, 4096, 4, 4096, false, 4096, Some(3), false).unwrap();

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

// Padded-transfer shape shared by the tests below: a two-block direct prefix
// followed by a partial final block.
const ALIGN: usize = 4096;
const PAYLOAD: usize = 2 * ALIGN + 1808;
const PREFIX: usize = PAYLOAD / ALIGN * ALIGN;
const TOTAL: usize = PAYLOAD.div_ceil(ALIGN) * ALIGN;
const TAIL_PAYLOAD: usize = PAYLOAD - PREFIX;

#[test]
fn padded_write_bounces_only_tail_without_modifying_source() {
    // Inspect the preparation helper to assert allocation size, which a
    // device round trip cannot distinguish from a full-buffer copy.
    let source = super::AlignedBuf::new(TOTAL, ALIGN).unwrap();
    unsafe { std::ptr::write_bytes(source.as_mut_ptr(), 0xa5, TOTAL) };
    let ptr = source.as_ptr() as usize;
    let prepared =
        prepare_iouring_write_buffer(ptr, PAYLOAD, PAYLOAD, TOTAL, true, ALIGN, None, false)
            .unwrap();
    assert_eq!(prepared.bounce.as_ref().unwrap().len, ALIGN);
    let vectors = prepared.iovecs.as_ref().unwrap();
    assert_eq!(vectors[0].base_addr, ptr);
    assert_eq!(vectors[0].len, PREFIX);
    assert_eq!(vectors[1].len, ALIGN);
    let tail =
        unsafe { std::slice::from_raw_parts(prepared.bounce.as_ref().unwrap().as_ptr(), ALIGN) };
    assert!(tail[..TAIL_PAYLOAD].iter().all(|v| *v == 0xa5));
    assert!(tail[TAIL_PAYLOAD..].iter().all(|v| *v == 0));
    assert!(
        unsafe { std::slice::from_raw_parts(source.as_ptr(), TOTAL) }
            .iter()
            .all(|v| *v == 0xa5)
    );
}

#[test]
fn padded_read_prepares_tail_copyback() {
    let target = super::AlignedBuf::new(TOTAL, ALIGN).unwrap();
    let ptr = target.as_mut_ptr() as usize;
    let prepared =
        super::prepare_iouring_read_buffer(ptr, PAYLOAD, PAYLOAD, TOTAL, true, false, ALIGN, None)
            .unwrap();
    assert_eq!(prepared.bounce.as_ref().unwrap().len, ALIGN);
    assert_eq!(prepared.original_ptr, Some(ptr + PREFIX));
    assert_eq!(prepared.payload_len, Some(TAIL_PAYLOAD));
    assert_eq!(prepared.iovecs.as_ref().unwrap()[0].len, PREFIX);
}

#[test]
fn tail_preparation_preserves_fallbacks_and_aligned_fast_path() {
    let source_len = TOTAL + ALIGN;
    let source = super::AlignedBuf::new(source_len, ALIGN).unwrap();
    unsafe { std::ptr::write_bytes(source.as_mut_ptr(), 7, source_len) };
    let ptr = source.as_ptr() as usize;
    for size in [17, ALIGN, ALIGN + 1, 2 * ALIGN - 1, 2 * ALIGN, PAYLOAD] {
        let total = super::round_up(size, ALIGN);
        let prepared =
            prepare_iouring_write_buffer(ptr, size, size, total, true, ALIGN, Some(3), false)
                .unwrap();
        if size == total {
            assert!(prepared.bounce.is_none());
            assert!(prepared.iovecs.is_none());
            assert_eq!(prepared.fixed_buffer_idx, Some(3));
        } else {
            assert_eq!(prepared.bounce.as_ref().unwrap().len, ALIGN);
            assert_eq!(prepared.iovecs.is_some(), size > ALIGN);
            assert_eq!(prepared.fixed_buffer_idx, None);
        }
    }
    for (address, total, cmd) in [
        // Unaligned address.
        (ptr + 1, TOTAL, false),
        // More than one block of padding.
        (ptr, TOTAL + ALIGN, false),
        // io_uring_cmd takes one buffer per command.
        (ptr, TOTAL, true),
    ] {
        let prepared =
            prepare_iouring_write_buffer(address, PAYLOAD, PAYLOAD, total, true, ALIGN, None, cmd)
                .unwrap();
        assert!(prepared.iovecs.is_none());
        assert_eq!(prepared.bounce.as_ref().unwrap().len, total);
    }
}

#[test]
fn vectored_completion_copies_tail_only_on_success() {
    for is_write in [false, true] {
        for result in [-libc::EIO, 0, PREFIX as i32, TOTAL as i32] {
            let target = super::AlignedBuf::new(TOTAL, ALIGN).unwrap();
            unsafe { std::ptr::write_bytes(target.as_mut_ptr(), 0xcc, TOTAL) };
            let ptr = target.as_ptr() as usize;
            let prepared = super::prepare_iouring_read_buffer(
                ptr, PAYLOAD, PAYLOAD, TOTAL, true, false, ALIGN, None,
            )
            .unwrap();
            unsafe {
                std::ptr::write_bytes(prepared.bounce.as_ref().unwrap().as_mut_ptr(), 7, ALIGN)
            };
            let weak = std::sync::Arc::downgrade(prepared.bounce.as_ref().unwrap());
            let mut submission = super::IoSubmission {
                len: TOTAL,
                is_write,
                iovecs: prepared.iovecs,
                bounce: prepared.bounce,
                original_ptr: prepared.original_ptr,
                payload_len: prepared.payload_len,
                ..Default::default()
            };
            let outcome = super::handle_completion_result(&mut submission, result, false);
            assert_eq!(outcome.is_ok(), result == TOTAL as i32);
            let bytes = unsafe { std::slice::from_raw_parts(target.as_ptr(), TOTAL) };
            let expected = if !is_write && result == TOTAL as i32 {
                7
            } else {
                0xcc
            };
            assert!(bytes[PREFIX..PAYLOAD]
                .iter()
                .all(|value| *value == expected));
            assert!(bytes[PAYLOAD..].iter().all(|value| *value == 0xcc));
            assert!(weak.upgrade().is_none());
        }
    }
}

#[test]
fn short_vectored_retries_advance_across_prefix_and_tail() {
    // Inject completion lengths at the worker transition: device round trips
    // cannot deterministically produce each positive-short boundary.
    let align = ALIGN as i32;
    let prefix = PREFIX as i32;
    let partial = 512;
    let target_len = TOTAL + ALIGN;
    let base_offset = (4 * ALIGN) as u64;
    for is_write in [false, true] {
        for completed in [
            &[align, align, partial][..],
            &[prefix][..],
            &[prefix + partial][..],
        ] {
            let target = super::AlignedBuf::new(target_len, ALIGN).unwrap();
            unsafe { std::ptr::write_bytes(target.as_mut_ptr(), 0xcc, target_len) };
            let ptr = target.as_ptr() as usize;
            let prepared = super::prepare_iouring_read_buffer(
                ptr, PAYLOAD, PAYLOAD, TOTAL, true, false, ALIGN, None,
            )
            .unwrap();
            let tail_ptr = prepared.bounce.as_ref().unwrap().as_ptr() as usize;
            unsafe { std::ptr::write_bytes(tail_ptr as *mut u8, 7, ALIGN) };
            let weak = Arc::downgrade(prepared.bounce.as_ref().unwrap());
            let mut sub = super::IoSubmission {
                offset: base_offset,
                len: TOTAL,
                is_write,
                iovecs: prepared.iovecs,
                bounce: prepared.bounce,
                original_ptr: prepared.original_ptr,
                payload_len: prepared.payload_len,
                ..Default::default()
            };
            let mut total = 0;
            let snapshot = Arc::clone(sub.iovecs.as_ref().unwrap());
            for &count in completed {
                assert!(super::prepare_short_iouring_retry(&mut sub, count));
                total += count as usize;
                assert_eq!(sub.offset, base_offset + total as u64);
                assert_eq!(sub.len, TOTAL - total);
                let vectors = sub.iovecs.as_ref().unwrap();
                assert_eq!(vectors.iter().map(|v| v.len).sum::<usize>(), sub.len);
                assert!(vectors.iter().all(|v| v.len > 0));
                let expected_ptr = if total < PREFIX {
                    ptr + total
                } else {
                    tail_ptr + total - PREFIX
                };
                assert_eq!(vectors[0].base_addr, expected_ptr);
                assert_eq!(vectors.len(), if total < PREFIX { 2 } else { 1 });
                assert_eq!(sub.original_ptr, Some(ptr + PREFIX));
                assert_eq!(sub.payload_len, Some(TAIL_PAYLOAD));
                assert!(weak.upgrade().is_some());
                let bytes = unsafe { std::slice::from_raw_parts(target.as_ptr(), target_len) };
                assert!(bytes.iter().all(|v| *v == 0xcc));
                assert_eq!(snapshot[0].base_addr, ptr);
                assert_eq!(snapshot[0].len, PREFIX);
                assert_eq!(snapshot[1].base_addr, tail_ptr);
                assert_eq!(snapshot[1].len, ALIGN);
            }
            let remaining = sub.len as i32;
            assert!(!super::prepare_short_iouring_retry(&mut sub, remaining));
            super::handle_completion_result(&mut sub, remaining, false).unwrap();
            let bytes = unsafe { std::slice::from_raw_parts(target.as_ptr(), target_len) };
            let expected = if is_write { 0xcc } else { 7 };
            assert!(bytes[PREFIX..PAYLOAD].iter().all(|v| *v == expected));
            assert!(bytes[PAYLOAD..].iter().all(|v| *v == 0xcc));
            assert!(weak.upgrade().is_none());
        }
    }
}

#[test]
fn short_vectored_retry_then_failure_does_not_copy_tail() {
    for (result, shutdown) in [(0, false), (-libc::EIO, false), (512, true)] {
        let target = super::AlignedBuf::new(TOTAL, ALIGN).unwrap();
        unsafe { std::ptr::write_bytes(target.as_mut_ptr(), 0xcc, TOTAL) };
        let ptr = target.as_ptr() as usize;
        let prepared = super::prepare_iouring_read_buffer(
            ptr, PAYLOAD, PAYLOAD, TOTAL, true, false, ALIGN, None,
        )
        .unwrap();
        let weak = Arc::downgrade(prepared.bounce.as_ref().unwrap());
        let mut sub = super::IoSubmission {
            len: TOTAL,
            iovecs: prepared.iovecs,
            bounce: prepared.bounce,
            original_ptr: prepared.original_ptr,
            payload_len: prepared.payload_len,
            ..Default::default()
        };
        assert!(super::prepare_short_iouring_retry(&mut sub, PREFIX as i32));
        if !shutdown {
            assert!(!super::prepare_short_iouring_retry(&mut sub, result));
        }
        assert_eq!(sub.len, TOTAL - PREFIX);
        assert_eq!(sub.offset, PREFIX as u64);
        assert!(super::handle_completion_result(&mut sub, result, shutdown).is_err());
        let bytes = unsafe { std::slice::from_raw_parts(target.as_ptr(), TOTAL) };
        assert!(bytes.iter().all(|v| *v == 0xcc));
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn short_retry_preserves_scalar_and_nvme_completion_rules() {
    let (ptr_addr, offset, len) = (4 * ALIGN, (8 * ALIGN) as u64, 2 * ALIGN);
    let initial = (ptr_addr, offset, len);
    let advanced = (ptr_addr + ALIGN, offset + ALIGN as u64, len - ALIGN);
    for is_write in [false, true] {
        let mut sub = super::IoSubmission {
            ptr_addr,
            offset,
            len,
            is_write,
            fixed_buffer_idx: Some(3),
            ..Default::default()
        };
        for result in [-libc::EIO, 0, len as i32] {
            assert!(!super::prepare_short_iouring_retry(&mut sub, result));
            assert_eq!((sub.ptr_addr, sub.offset, sub.len), initial);
        }
        assert!(super::prepare_short_iouring_retry(&mut sub, ALIGN as i32));
        assert_eq!((sub.ptr_addr, sub.offset, sub.len), advanced);
        assert_eq!(sub.fixed_buffer_idx, Some(3));
        sub.nvme_cmd_data = Some(super::NvmeCmdData {
            nsid: 1,
            lba_shift: 9,
            dtype: 0,
            dspec: 0,
        });
        for result in [0, 1, -libc::EIO] {
            assert!(!super::prepare_short_iouring_retry(&mut sub, result));
            assert_eq!((sub.ptr_addr, sub.offset, sub.len), advanced);
        }
        super::handle_completion_result(&mut sub, 0, false).unwrap();
    }
}
