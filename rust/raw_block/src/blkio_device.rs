// SPDX-License-Identifier: Apache-2.0

//! [`BlkioBlockDevice`] — a PyO3-exposed block device I/O class backed by
//! libblkio's `io_uring` driver.
//!
//! This is a drop-in replacement for [`RawBlockDevice`](super::RawBlockDevice)
//! for the synchronous I/O paths (`pwrite_from_buffer`, `pread_into`,
//! `size_bytes`, `close`).  The Python `RustRawBlockBackend` plugin selects
//! this class when `rust_raw_block.io_backend = "libblkio"`.
//!
//! Design notes:
//! - Each instance owns a single `blkio` handle + queue (single-threaded I/O).
//! - Per-I/O buffer registration (`map_mem_region` / `unmap_mem_region`) is
//!   used.  This matches the existing C++ `BlkioConnector` pattern and keeps
//!   the implementation simple at the cost of some overhead per I/O.
//! - When O_DIRECT is enabled the caller is expected to supply aligned
//!   `offset` and `total_len`.  If the source/destination pointer is not
//!   aligned a bounce buffer is allocated via `posix_memalign`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::blkio_ffi;
use crate::{get_pybuffer, release_pybuffer};

// Re-use the aligned-buffer helper already defined in lib.rs.
use crate::AlignedBuf;

/// Block device I/O via libblkio (io_uring driver).
///
/// Exposes the same Python-facing methods as `RawBlockDevice`:
///   - `pwrite_from_buffer(offset, data, payload_len, total_len)`
///   - `pread_into(offset, out, payload_len, total_len)`
///   - `size_bytes()`
///   - `close()`
#[pyclass]
pub struct BlkioBlockDevice {
    /// Stored as usize to satisfy Send/Sync (raw pointers do not impl Send).
    /// SAFETY: The pointee lifetime is managed by blkio_destroy in close().
    handle: usize,
    queue: usize,
    size: u64,
    use_odirect: bool,
    alignment: usize,
    closed: AtomicBool,
}

// SAFETY: The blkio handle is only accessed through &self methods, never
// concurrently mutated.  Python's GIL serialises calls; the GIL-release
// sections only touch local stack data or the owned handle (single queue).
unsafe impl Send for BlkioBlockDevice {}
unsafe impl Sync for BlkioBlockDevice {}

/// Convert a negative libblkio error code to a Python exception.
fn blkio_err(msg: &str, code: i32) -> PyErr {
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
fn do_aligned_io(
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

// --------------------------------------------------------- impl BlkioBlockDevice

#[pymethods]
impl BlkioBlockDevice {
    /// Open a block device via libblkio.
    ///
    /// Args:
    ///     path: block device or file path.
    ///     writable: open for writing as well.
    ///     use_odirect: enable O_DIRECT.
    ///     alignment: required alignment (default 4096).
    #[new]
    #[pyo3(signature = (path, writable, use_odirect=false, alignment=4096))]
    fn new(path: &str, writable: bool, use_odirect: bool, alignment: usize) -> PyResult<Self> {
        if path.is_empty() {
            return Err(PyValueError::new_err("path must not be empty"));
        }
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(PyValueError::new_err("alignment must be a power of two"));
        }

        let c_driver = CString::new("io_uring").unwrap();
        let c_path = CString::new(path)
            .map_err(|_| PyValueError::new_err("path contains null byte"))?;

        let mut handle: *mut blkio_ffi::blkio = ptr::null_mut();

        // Create
        let ret = unsafe { blkio_ffi::blkio_create(c_driver.as_ptr(), &mut handle) };
        if ret < 0 {
            return Err(blkio_err("blkio_create failed", ret));
        }

        // Set path
        let c_name_path = CString::new("path").unwrap();
        let ret =
            unsafe { blkio_ffi::blkio_set_str(handle, c_name_path.as_ptr(), c_path.as_ptr()) };
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
            return Err(PyRuntimeError::new_err("blkio_get_queue(0) returned null"));
        }

        // Query capacity
        let mut capacity: u64 = 0;
        let c_cap = CString::new("capacity").unwrap();
        let ret = unsafe { blkio_ffi::blkio_get_uint64(handle, c_cap.as_ptr(), &mut capacity) };
        if ret < 0 {
            unsafe { blkio_ffi::blkio_destroy(&mut handle) };
            return Err(blkio_err("blkio query capacity failed", ret));
        }

        Ok(Self {
            handle: handle as usize,
            queue: queue as usize,
            size: capacity,
            use_odirect,
            alignment,
            closed: AtomicBool::new(false),
        })
    }

    // ---------------------------------------------------------------- I/O

    /// Write data to the block device.
    ///
    /// `payload_len` bytes of real data are written; the region
    /// `[payload_len, total_len)` is zero-padded.  For O_DIRECT,
    /// `offset` and `total_len` must be aligned.
    #[pyo3(signature = (offset, data, payload_len=None, total_len=None))]
    fn pwrite_from_buffer(
        &self,
        py: Python<'_>,
        offset: u64,
        data: &Bound<'_, PyAny>,
        payload_len: Option<usize>,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("device is closed"));
        }

        let view = get_pybuffer(py, data, false)?;
        let ptr = view.buf as *const u8;
        let buf_len = view.len as usize;
        if ptr.is_null() {
            release_pybuffer(view);
            return Err(PyValueError::new_err("null buffer pointer"));
        }

        let payload_len = payload_len.unwrap_or(buf_len);
        if payload_len > buf_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err("payload_len exceeds buffer length"));
        }
        let total_len = total_len.unwrap_or(payload_len);
        if total_len < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        let align = self.alignment;
        if self.use_odirect {
            if (offset as usize) % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            if total_len % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        let handle = self.handle;
        let queue = self.queue;
        let ptr_usize = ptr as usize;

        let res = py.allow_threads(move || {
            let src = ptr_usize as *const u8;
            let src_aligned = (src as usize) % align == 0;

            // Fast path: no padding, source aligned.
            if total_len == payload_len && src_aligned {
                return do_aligned_io(handle, queue, false, offset, src as *mut u8, total_len);
            }

            // O_DIRECT hybrid: source aligned, needs padding.
            if src_aligned && total_len > payload_len {
                let aligned_prefix = payload_len / align * align;
                if aligned_prefix > 0 {
                    do_aligned_io(handle, queue, false, offset, src as *mut u8, aligned_prefix)?;
                }
                let tail_payload = payload_len - aligned_prefix;
                let tail_total = total_len - aligned_prefix;
                if tail_total > 0 {
                    let tail_offset = offset + aligned_prefix as u64;
                    let bounce = AlignedBuf::new(tail_total, align)?;
                    unsafe {
                        if tail_payload > 0 {
                            libc::memcpy(
                                bounce.as_mut_ptr() as *mut libc::c_void,
                                src.add(aligned_prefix) as *const libc::c_void,
                                tail_payload,
                            );
                        }
                        if tail_total > tail_payload {
                            libc::memset(
                                bounce.as_mut_ptr().add(tail_payload) as *mut libc::c_void,
                                0,
                                tail_total - tail_payload,
                            );
                        }
                    }
                    do_aligned_io(handle, queue, false, tail_offset, bounce.as_mut_ptr(), tail_total)?;
                }
                return Ok(());
            }

            // Full bounce path.
            let bounce = AlignedBuf::new(total_len, align)?;
            unsafe {
                libc::memcpy(
                    bounce.as_mut_ptr() as *mut libc::c_void,
                    src as *const libc::c_void,
                    payload_len,
                );
                if total_len > payload_len {
                    libc::memset(
                        bounce.as_mut_ptr().add(payload_len) as *mut libc::c_void,
                        0,
                        total_len - payload_len,
                    );
                }
            }
            do_aligned_io(handle, queue, false, offset, bounce.as_mut_ptr(), total_len)
        });

        release_pybuffer(view);
        res
    }

    /// Read data from the block device.
    ///
    /// Reads `total_len` bytes (aligned for O_DIRECT) and copies
    /// `payload_len` bytes into the output buffer.
    #[pyo3(signature = (offset, out, payload_len, total_len=None))]
    fn pread_into(
        &self,
        py: Python<'_>,
        offset: u64,
        out: &Bound<'_, PyAny>,
        payload_len: usize,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("device is closed"));
        }

        let view = get_pybuffer(py, out, true)?;
        if view.readonly != 0 {
            release_pybuffer(view);
            return Err(PyValueError::new_err("output buffer is readonly"));
        }
        let cap = view.len as usize;
        if cap < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err(format!(
                "output buffer too small: cap={cap} need={payload_len}"
            )));
        }
        let ptr = view.buf as *mut u8;
        if ptr.is_null() {
            release_pybuffer(view);
            return Err(PyValueError::new_err("null buffer pointer"));
        }

        let total_len = total_len.unwrap_or(payload_len);
        if total_len < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        let align = self.alignment;
        if self.use_odirect {
            if (offset as usize) % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            if total_len % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        let handle = self.handle;
        let queue = self.queue;
        let dst_usize = ptr as usize;

        let res = py.allow_threads(move || {
            let dst = dst_usize as *mut u8;
            let dst_aligned = (dst as usize) % align == 0;

            // Fast path: no over-read, destination aligned.
            if total_len == payload_len && dst_aligned {
                return do_aligned_io(handle, queue, true, offset, dst, payload_len);
            }

            // O_DIRECT: destination aligned and large enough.
            if dst_aligned && cap >= total_len {
                return do_aligned_io(handle, queue, true, offset, dst, total_len);
            }

            // O_DIRECT hybrid: read aligned prefix directly, bounce tail.
            if dst_aligned {
                let aligned_prefix = payload_len / align * align;
                if aligned_prefix > 0 {
                    do_aligned_io(handle, queue, true, offset, dst, aligned_prefix)?;
                }
                let tail_payload = payload_len - aligned_prefix;
                let tail_total = total_len - aligned_prefix;
                if tail_total > 0 {
                    let tail_offset = offset + aligned_prefix as u64;
                    let bounce = AlignedBuf::new(tail_total, align)?;
                    do_aligned_io(handle, queue, true, tail_offset, bounce.as_mut_ptr(), tail_total)?;
                    if tail_payload > 0 {
                        unsafe {
                            libc::memcpy(
                                dst.add(aligned_prefix) as *mut libc::c_void,
                                bounce.as_ptr() as *const libc::c_void,
                                tail_payload,
                            );
                        }
                    }
                }
                return Ok(());
            }

            // Full bounce read path.
            let round_up = |x: usize, a: usize| (x + a - 1) / a * a;
            let bounce = AlignedBuf::new(round_up(total_len, align), align)?;
            do_aligned_io(handle, queue, true, offset, bounce.as_mut_ptr(), total_len)?;
            unsafe {
                libc::memcpy(
                    dst as *mut libc::c_void,
                    bounce.as_ptr() as *const libc::c_void,
                    payload_len,
                );
            }
            Ok(())
        });

        release_pybuffer(view);
        res
    }

    // ---------------------------------------------------------- queries

    /// Return the device size in bytes.
    fn size_bytes(&self) -> PyResult<u64> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("device is closed"));
        }
        Ok(self.size)
    }

    /// Shut down the libblkio instance.
    fn close(&mut self) -> PyResult<()> {
        if !self.closed.load(Ordering::Relaxed) {
            self.closed.store(true, Ordering::Relaxed);
            let h = self.handle;
            if h != 0 {
                let mut p = h as *mut blkio_ffi::blkio;
                unsafe { blkio_ffi::blkio_destroy(&mut p) };
                self.handle = 0;
                self.queue = 0;
            }
        }
        Ok(())
    }
}

impl Drop for BlkioBlockDevice {
    fn drop(&mut self) {
        if !self.closed.load(Ordering::Relaxed) {
            let _ = self.close();
        }
    }
}
