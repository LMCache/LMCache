// SPDX-License-Identifier: Apache-2.0
#![allow(unknown_lints)]

//! Raw block device I/O extension for LMCache.
//! Provides direct block device access with optional O_DIRECT support.
//!
//! Design notes (for reviewers unfamiliar with Rust / Linux I/O):
//! - This module exposes a very small surface to Python via PyO3.
//! - We wrap Linux `pread` / `pwrite` on a file descriptor opened from a
//!   block device (e.g., /dev/nvmeXnY) or a regular file (for tests).
//! - When O_DIRECT is enabled, Linux requires aligned offsets and I/O sizes.
//!   If Python buffers are aligned, we use them directly; otherwise we fallback
//!   to a bounce buffer (aligned via `posix_memalign`) for safety.

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::ffi::CString;
use std::os::unix::io::RawFd;
use std::slice;

// Linux ioctl for block device size in bytes.
// Defined in <linux/fs.h>: BLKGETSIZE64 _IOR(0x12,114,size_t)
const BLKGETSIZE64: libc::c_ulong = 0x8008_1272; // ioctl op to query block size

// Buffer protocol flags (from CPython C-API).
const PYBUF_WRITABLE: i32 = 0x0001; // buffer must be writable
const PYBUF_ND: i32 = 0x0008; // request N-dimensional buffer
const PYBUF_STRIDES: i32 = 0x0010 | PYBUF_ND; // request strides info
const PYBUF_ANY_CONTIGUOUS: i32 = 0x0080 | PYBUF_STRIDES; // accept any contiguous layout

// O_DIRECT is Linux-only; define a no-op fallback for other platforms.
#[cfg(target_os = "linux")]
const O_DIRECT: i32 = libc::O_DIRECT;
#[cfg(not(target_os = "linux"))]
const O_DIRECT: i32 = 0;

/// Round up to nearest multiple of alignment (required for O_DIRECT).
#[allow(clippy::manual_div_ceil)]
// Small helper used to align sizes for O_DIRECT I/O.
fn round_up(x: usize, align: usize) -> usize {
    (x + align - 1) / align * align
}

// Fetch errno for the last libc call on this thread.
fn errno() -> i32 {
    // SAFETY: libc call.
    #[cfg(target_os = "linux")]
    unsafe {
        *libc::__errno_location()
    }
    #[cfg(target_os = "macos")]
    unsafe {
        *libc::__error()
    }
}

// Convert errno to a Python OSError with a message.
fn os_err(msg: &str) -> PyErr {
    PyOSError::new_err((errno(), msg.to_string()))
}

// Low-level write loop that retries until all bytes are written.
// This isolates the raw syscalls from Python-facing logic.
fn pwrite_from_ptr(
    fd: RawFd,
    mut offset: u64,
    mut ptr: *const u8,
    mut len: usize,
) -> Result<(), PyErr> {
    while len > 0 {
        // SAFETY: caller guarantees ptr is valid for len bytes.
        let chunk = unsafe { slice::from_raw_parts(ptr, len) };
        let n = unsafe {
            libc::pwrite(
                fd,
                chunk.as_ptr() as *const libc::c_void,
                chunk.len(),
                offset as libc::off_t,
            )
        };
        if n < 0 {
            return Err(os_err("pwrite failed"));
        }
        let n = n as usize;
        offset += n as u64;
        // SAFETY: advance ptr by n bytes.
        unsafe {
            ptr = ptr.add(n);
        }
        len -= n;
    }
    Ok(())
}

// Low-level read loop that retries until all bytes are read.
// We treat EOF as an error because the caller expects a full read.
fn pread_into(fd: RawFd, offset: u64, mut dst: *mut u8, mut size: usize) -> Result<(), PyErr> {
    let mut off = offset;
    while size > 0 {
        // SAFETY: pread writes into dst for size bytes.
        let n = unsafe { libc::pread(fd, dst as *mut libc::c_void, size, off as libc::off_t) };
        if n < 0 {
            return Err(os_err("pread failed"));
        }
        if n == 0 {
            return Err(PyRuntimeError::new_err("unexpected EOF"));
        }
        let n = n as usize;
        // SAFETY: advance dst by n bytes.
        unsafe {
            dst = dst.add(n);
        }
        off += n as u64;
        size -= n;
    }
    Ok(())
}

// Determine file/device size in bytes (ioctl for block device, fstat fallback).
fn fd_size_bytes(fd: RawFd) -> Result<u64, PyErr> {
    // Try ioctl first (block device / loop device).
    let mut size: u64 = 0;
    // SAFETY: ioctl expects pointer to u64 for BLKGETSIZE64.
    let rc = unsafe { libc::ioctl(fd, BLKGETSIZE64, &mut size as *mut u64) };
    if rc == 0 {
        return Ok(size);
    }

    // Fallback to fstat for regular files.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let rc2 = unsafe { libc::fstat(fd, &mut st as *mut libc::stat) };
    if rc2 != 0 {
        return Err(os_err("fstat failed"));
    }
    Ok(st.st_size as u64)
}

/// Aligned buffer for O_DIRECT I/O.
/// Allocated with posix_memalign so the pointer satisfies alignment requirements.
/// Automatically freed on drop.
struct AlignedBuf {
    ptr: *mut u8,
    #[allow(dead_code)]
    len: usize,
    #[allow(dead_code)]
    align: usize,
}

impl AlignedBuf {
    // Allocate an aligned buffer suitable for O_DIRECT.
    fn new(len: usize, align: usize) -> Result<Self, PyErr> {
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        // SAFETY: posix_memalign writes to p.
        let rc = unsafe { libc::posix_memalign(&mut p as *mut *mut libc::c_void, align, len) };
        if rc != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "posix_memalign failed rc={rc}"
            )));
        }
        if p.is_null() {
            return Err(PyRuntimeError::new_err("posix_memalign returned null"));
        }
        Ok(Self {
            ptr: p as *mut u8,
            len,
            align,
        })
    }

    // Mutable pointer for read/write syscalls.
    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    // Const pointer for write syscalls.
    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::free(self.ptr as *mut libc::c_void);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

// Acquire a Python buffer view with the requested mutability.
fn get_pybuffer<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    writable: bool,
) -> Result<pyo3::ffi::Py_buffer, PyErr> {
    // SAFETY: PyObject_GetBuffer follows CPython buffer protocol.
    unsafe {
        let mut view: pyo3::ffi::Py_buffer = std::mem::zeroed();
        // Request a contiguous byte-view. This lets Rust issue a single syscall
        // against a flat pointer instead of handling Python strides/shapes.
        let flags = if writable {
            PYBUF_WRITABLE | PYBUF_ANY_CONTIGUOUS
        } else {
            PYBUF_ANY_CONTIGUOUS
        };
        let rc = pyo3::ffi::PyObject_GetBuffer(obj.as_ptr(), &mut view, flags);
        if rc != 0 {
            return Err(PyErr::fetch(py));
        }
        Ok(view)
    }
}

// Release a buffer view previously acquired by get_pybuffer.
fn release_pybuffer(mut view: pyo3::ffi::Py_buffer) {
    // SAFETY: view was created by PyObject_GetBuffer.
    unsafe {
        pyo3::ffi::PyBuffer_Release(&mut view);
    }
}

/// Single-FD synchronous raw block device interface.
/// This is intentionally minimal: open -> pread/pwrite -> close.
/// Higher-level policies (slotting, manifests, etc.) live in Python.
#[pyclass]
struct RawBlockDevice {
    fd: RawFd,         // raw file descriptor
    size: u64,         // cached device size in bytes
    closed: bool,      // avoid double-close
    use_odirect: bool, // enforce alignment + bypass page cache
    alignment: usize,  // required alignment in bytes
}

#[pymethods]
impl RawBlockDevice {
    #[new]
    #[pyo3(signature=(path, writable, use_odirect=false, alignment=4096))]
    fn new(path: String, writable: bool, use_odirect: bool, alignment: usize) -> PyResult<Self> {
        let cpath = CString::new(path).map_err(|_| PyValueError::new_err("path contains NUL"))?;
        let mut flags = if writable {
            libc::O_RDWR
        } else {
            libc::O_RDONLY
        };
        if use_odirect {
            flags |= O_DIRECT;
        }
        // SAFETY: open returns fd or -1.
        let fd = unsafe { libc::open(cpath.as_ptr(), flags) };
        if fd < 0 {
            return Err(os_err("open failed"));
        }
        let size = fd_size_bytes(fd)?;
        Ok(Self {
            fd,
            size,
            closed: false,
            use_odirect,
            alignment,
        })
    }

    // Expose cached size to Python.
    fn size_bytes(&self) -> PyResult<u64> {
        Ok(self.size)
    }

    /// Write bytes from any Python buffer object into the device.
    /// For O_DIRECT, we use direct pointer I/O when aligned and fallback to
    /// bounce buffering only for the unaligned/padded tail.
    #[pyo3(signature=(offset, data, payload_len=None, total_len=None))]
    fn pwrite_from_buffer(
        &self,
        py: Python<'_>,
        offset: u64,
        data: &Bound<'_, PyAny>,
        payload_len: Option<usize>,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        if self.closed {
            return Err(PyRuntimeError::new_err("device is closed"));
        }
        let fd = self.fd;

        let view = get_pybuffer(py, data, false)?;
        let ptr = view.buf as *const u8;
        let buf_len = view.len as usize;
        if ptr.is_null() {
            release_pybuffer(view);
            return Err(PyValueError::new_err("null buffer pointer"));
        }

        // `payload_len`: user bytes to write.
        // `total_len`: actual I/O length. For O_DIRECT this is often aligned up.
        // Example: payload=4100, align=4096 -> total_len=8192.
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
            #[allow(clippy::manual_is_multiple_of)]
            if (offset as usize) % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            #[allow(clippy::manual_is_multiple_of)]
            if total_len % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        // Store pointer as integer before releasing the GIL. The closure passed
        // to `allow_threads` must own plain data and cannot borrow `view`.
        // We still keep `view` alive until I/O finishes, then release it below.
        let ptr_usize = ptr as usize;
        let res = py.allow_threads(move || {
            let src = ptr_usize as *const u8;
            let src_aligned = (src as usize).is_multiple_of(align);
            if total_len == payload_len && !self.use_odirect {
                // direct write without padding
                return pwrite_from_ptr(fd, offset, src, payload_len);
            }

            if self.use_odirect && src_aligned {
                if total_len == payload_len {
                    // Fully aligned fast path: no copies.
                    return pwrite_from_ptr(fd, offset, src, total_len);
                }

                // Hybrid path for O_DIRECT with padding:
                // - If the Python pointer is aligned, we avoid copying the large
                //   aligned prefix and write it directly.
                // - Only the tail is copied into an aligned bounce buffer, then
                //   zero-padded to satisfy O_DIRECT full-block writes.
                //
                // This keeps copy cost proportional to tail size, not payload size.
                let aligned_prefix = payload_len / align * align;
                if aligned_prefix > 0 {
                    pwrite_from_ptr(fd, offset, src, aligned_prefix)?;
                }
                let tail_payload = payload_len - aligned_prefix;
                let tail_total = total_len - aligned_prefix;
                if tail_total > 0 {
                    let tail_offset = offset
                        .checked_add(aligned_prefix as u64)
                        .ok_or_else(|| PyValueError::new_err("offset overflow"))?;
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
                    pwrite_from_ptr(fd, tail_offset, bounce.as_ptr(), tail_total)?;
                }
                return Ok(());
            }

            // Full bounce path:
            // - required when source pointer is not alignment-safe for O_DIRECT.
            // - also used when non-O_DIRECT call asks for padding behavior.
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
            pwrite_from_ptr(fd, offset, bounce.as_ptr(), total_len)
        });
        // Always release the CPython buffer view once the blocking I/O closure
        // completes. This decrements exporter-side view count correctly.
        release_pybuffer(view);
        res?;
        Ok(())
    }

    /// Read exactly `payload_len` bytes into a writable Python buffer.
    /// For O_DIRECT, use direct reads when destination is aligned and fallback
    /// to a hybrid/read-bounce path when needed.
    #[pyo3(signature=(offset, out, payload_len, total_len=None))]
    fn pread_into(
        &self,
        py: Python<'_>,
        offset: u64,
        out: &Bound<'_, PyAny>,
        payload_len: usize,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        if self.closed {
            return Err(PyRuntimeError::new_err("device is closed"));
        }
        let fd = self.fd;
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

        // `payload_len`: bytes caller wants copied into `out`.
        // `total_len`: bytes to read from device. For O_DIRECT this is usually
        // aligned up and can be larger than payload_len.
        let total_len = total_len.unwrap_or(payload_len);
        if total_len < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        let align = self.alignment;
        if self.use_odirect {
            #[allow(clippy::manual_is_multiple_of)]
            if (offset as usize) % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            #[allow(clippy::manual_is_multiple_of)]
            if total_len % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        // Same pattern as write path: move raw address into closure-safe value
        // while retaining `view` lifetime until closure completion.
        let dst_usize = ptr as usize;
        let res = py.allow_threads(move || {
            let dst = dst_usize as *mut u8;
            let dst_aligned = (dst as usize).is_multiple_of(align);
            if total_len == payload_len && !self.use_odirect {
                return pread_into(fd, offset, dst, payload_len);
            }

            if self.use_odirect && dst_aligned {
                if cap >= total_len {
                    // Fully aligned fast path: no copies.
                    return pread_into(fd, offset, dst, total_len);
                }

                // Hybrid path for O_DIRECT with smaller destination capacity:
                // - read aligned prefix directly into destination.
                // - read aligned tail into bounce buffer.
                // - copy only payload tail bytes back into destination.
                //
                // This avoids writing beyond Python buffer capacity while still
                // honoring O_DIRECT aligned read requirements.
                let aligned_prefix = payload_len / align * align;
                if aligned_prefix > 0 {
                    pread_into(fd, offset, dst, aligned_prefix)?;
                }
                let tail_payload = payload_len - aligned_prefix;
                let tail_total = total_len - aligned_prefix;
                if tail_total > 0 {
                    let tail_offset = offset
                        .checked_add(aligned_prefix as u64)
                        .ok_or_else(|| PyValueError::new_err("offset overflow"))?;
                    let bounce = AlignedBuf::new(tail_total, align)?;
                    pread_into(fd, tail_offset, bounce.as_mut_ptr(), tail_total)?;
                    unsafe {
                        if tail_payload > 0 {
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

            // Full bounce read path:
            // read aligned size into temporary aligned memory, then copy the
            // requested payload portion to Python output buffer.
            let bounce = AlignedBuf::new(round_up(total_len, align), align)?;
            pread_into(fd, offset, bounce.as_mut_ptr(), total_len)?;
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
        res?;
        Ok(())
    }

    // Explicit close from Python. Drop also closes if needed.
    fn close(&mut self) -> PyResult<()> {
        if !self.closed {
            // SAFETY: close fd once.
            let rc = unsafe { libc::close(self.fd) };
            if rc != 0 {
                return Err(os_err("close failed"));
            }
            self.closed = true;
        }
        Ok(())
    }
}

impl Drop for RawBlockDevice {
    fn drop(&mut self) {
        if !self.closed {
            unsafe {
                libc::close(self.fd);
            }
            self.closed = true;
        }
    }
}

#[pymodule]
fn lmcache_rust_raw_block_io(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RawBlockDevice>()?;
    Ok(())
}
