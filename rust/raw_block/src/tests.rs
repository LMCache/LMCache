// SPDX-License-Identifier: Apache-2.0

use super::{check_nvme_ioctl_result, placement_id_to_u16, RawBlockDevice};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

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
