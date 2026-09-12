// SPDX-License-Identifier: Apache-2.0

//! libblkio helper functions for [`RawBlockDevice`](super::RawBlockDevice).
//!
//! When `io_engine = "libblkio"` is passed to [`RawBlockDevice::new()`], the
//! constructor delegates to [`blkio_open_device()`] and the I/O paths dispatch
//! through [`do_aligned_io()`].
//!
//! Design notes:
//! - Each instance owns a single `blkio` handle + queue (single-threaded I/O).
//! - Per-I/O buffer registration (`map_mem_region` / `unmap_mem_region`) is
//!   used.  This matches the NIXL `registerBlkioBuf` pattern and keeps the
//!   implementation simple at the cost of some overhead per I/O.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;

use crate::blkio_ffi;

/// Convert a negative libblkio error code to a Python exception.
pub(crate) fn blkio_err(msg: &str, code: i32) -> PyErr {
    let errno_str = unsafe {
        let p = libc::strerror(-code);
        if p.is_null() {
            "unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    PyRuntimeError::new_err(format!("{msg}: {errno_str}"))
}

// ------------------------------------------------------------------ helpers

/// Submit one I/O via libblkio and synchronously wait for its completion.
///
/// `handle` and `queue` are the blkio/blkioq pointers stored as `usize`.
/// `buf` must already satisfy alignment requirements.
pub(crate) fn do_aligned_io(
    handle: usize,
    queue: usize,
    is_read: bool,
    offset: u64,
    buf: *mut u8,
    len: usize,
) -> Result<(), PyErr> {
    let handle = handle as *mut blkio_ffi::blkio;
    let queue = queue as *mut blkio_ffi::blkioq;

    // Register buffer with blkio.
    let region = blkio_ffi::blkio_mem_region {
        addr: buf as *mut c_void,
        len,
        iova: 0,
        fd_offset: 0,
        fd: -1,
        flags: 0,
    };

    let ret = unsafe { blkio_ffi::blkio_map_mem_region(handle, &region) };
    if ret < 0 {
        return Err(blkio_err("blkio_map_mem_region failed", ret));
    }

    // Submit I/O.
    let mut comp = blkio_ffi::blkio_completion {
        user_data: ptr::null_mut(),
        error_msg: ptr::null(),
        ret: 0,
        reserved_: [0u8; 12],
    };

    if is_read {
        unsafe {
            blkio_ffi::blkioq_read(
                queue,
                offset,
                buf as *mut c_void,
                len,
                ptr::null_mut(),
                0,
            );
        }
    } else {
        unsafe {
            blkio_ffi::blkioq_write(
                queue,
                offset,
                buf as *const c_void,
                len,
                ptr::null_mut(),
                0,
            );
        }
    }

    // Wait for the single completion.
    let ret = unsafe { blkio_ffi::blkioq_do_io(queue, &mut comp, 1, 1, ptr::null_mut()) };
    if ret < 0 {
        unsafe { blkio_ffi::blkio_unmap_mem_region(handle, &region) };
        return Err(blkio_err("blkioq_do_io failed", ret));
    }
    if comp.ret < 0 {
        unsafe { blkio_ffi::blkio_unmap_mem_region(handle, &region) };
        return Err(blkio_err("blkio I/O completion error", comp.ret));
    }

    unsafe { blkio_ffi::blkio_unmap_mem_region(handle, &region) };
    Ok(())
}

// --------------------------------------------------------- device lifecycle

/// Open a block device via libblkio and return `(handle, queue, size)`.
///
/// The caller takes ownership of the handle and must call
/// [`blkio_close_device()`] to destroy it.
///
/// Args:
///     path: block device or file path.
///     writable: open for writing as well.
///     use_odirect: enable O_DIRECT.
///     blkio_driver: libblkio driver name (e.g. "io_uring",
///         "virtio-blk-vhost-user").
///
/// Returns:
///     `(handle_usize, queue_usize, capacity_bytes)` on success.
pub(crate) fn blkio_open_device(
    path: &str,
    writable: bool,
    use_odirect: bool,
    blkio_driver: &str,
) -> Result<(usize, usize, u64), PyErr> {
    if path.is_empty() {
        return Err(PyValueError::new_err("path must not be empty"));
    }

    let c_driver = CString::new(blkio_driver)
        .map_err(|_| PyValueError::new_err("blkio_driver contains null byte"))?;
    let c_path =
        CString::new(path).map_err(|_| PyValueError::new_err("path contains null byte"))?;

    let mut handle: *mut blkio_ffi::blkio = ptr::null_mut();

    // Create
    let ret = unsafe { blkio_ffi::blkio_create(c_driver.as_ptr(), &mut handle) };
    if ret < 0 {
        return Err(blkio_err("blkio_create failed", ret));
    }

    // Set path
    let c_name_path = CString::new("path").unwrap();
    let ret = unsafe { blkio_ffi::blkio_set_str(handle, c_name_path.as_ptr(), c_path.as_ptr()) };
    if ret < 0 {
        unsafe { blkio_ffi::blkio_destroy(&mut handle) };
        return Err(blkio_err("blkio set path failed", ret));
    }

    // O_DIRECT
    if use_odirect {
        let c_name = CString::new("direct").unwrap();
        let ret = unsafe { blkio_ffi::blkio_set_bool(handle, c_name.as_ptr(), true) };
        if ret < 0 {
            unsafe { blkio_ffi::blkio_destroy(&mut handle) };
            return Err(blkio_err("blkio enable O_DIRECT failed", ret));
        }
    }

    // Read-only
    if !writable {
        let c_name = CString::new("read-only").unwrap();
        let ret = unsafe { blkio_ffi::blkio_set_bool(handle, c_name.as_ptr(), true) };
        if ret < 0 {
            unsafe { blkio_ffi::blkio_destroy(&mut handle) };
            return Err(blkio_err("blkio set read-only failed", ret));
        }
    }

    // Connect + start
    let ret = unsafe { blkio_ffi::blkio_connect(handle) };
    if ret < 0 {
        unsafe { blkio_ffi::blkio_destroy(&mut handle) };
        return Err(blkio_err("blkio_connect failed", ret));
    }
    let ret = unsafe { blkio_ffi::blkio_start(handle) };
    if ret < 0 {
        unsafe { blkio_ffi::blkio_destroy(&mut handle) };
        return Err(blkio_err("blkio_start failed", ret));
    }

    // Queue
    let queue = unsafe { blkio_ffi::blkio_get_queue(handle, 0) };
    if queue.is_null() {
        unsafe { blkio_ffi::blkio_destroy(&mut handle) };
        return Err(PyRuntimeError::new_err(
            "blkio_get_queue(0) returned null",
        ));
    }

    // Query capacity
    let mut capacity: u64 = 0;
    let c_cap = CString::new("capacity").unwrap();
    let ret = unsafe { blkio_ffi::blkio_get_uint64(handle, c_cap.as_ptr(), &mut capacity) };
    if ret < 0 {
        unsafe { blkio_ffi::blkio_destroy(&mut handle) };
        return Err(blkio_err("blkio query capacity failed", ret));
    }

    Ok((handle as usize, queue as usize, capacity))
}

/// Destroy a blkio instance previously opened by [`blkio_open_device()`].
///
/// `handle` is the blkio pointer stored as `usize`.  After this call the
/// handle must not be used again.
pub(crate) fn blkio_close_device(handle: usize) {
    if handle != 0 {
        let mut p = handle as *mut blkio_ffi::blkio;
        unsafe { blkio_ffi::blkio_destroy(&mut p) };
    }
}
