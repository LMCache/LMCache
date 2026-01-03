// SPDX-License-Identifier: Apache-2.0

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyAny;
use std::ffi::CString;
use std::os::unix::io::RawFd;
use std::slice;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

// Linux ioctl for block device size in bytes.
// Defined in <linux/fs.h>: BLKGETSIZE64 _IOR(0x12,114,size_t)
const BLKGETSIZE64: libc::c_ulong = 0x8008_1272;

// Buffer protocol flags (from CPython C-API).
const PYBUF_WRITABLE: i32 = 0x0001;
const PYBUF_ND: i32 = 0x0008;
const PYBUF_STRIDES: i32 = 0x0010 | PYBUF_ND;
const PYBUF_ANY_CONTIGUOUS: i32 = 0x0080 | PYBUF_STRIDES;

fn round_up(x: usize, align: usize) -> usize {
    (x + align - 1) / align * align
}

fn cpu_count() -> usize {
    // SAFETY: sysconf is a libc call; returns -1 on error.
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if n <= 0 {
        1
    } else {
        n as usize
    }
}

fn errno() -> i32 {
    // SAFETY: libc call.
    unsafe { *libc::__errno_location() }
}

fn os_err(msg: &str) -> PyErr {
    PyOSError::new_err((errno(), msg.to_string()))
}

#[derive(Clone, Debug)]
enum ErrorInfo {
    Os { errno: i32, msg: String },
    Value { msg: String },
    Runtime { msg: String },
}

fn errorinfo_to_pyerr(e: ErrorInfo) -> PyErr {
    match e {
        ErrorInfo::Os { errno, msg } => PyOSError::new_err((errno, msg)),
        ErrorInfo::Value { msg } => PyValueError::new_err(msg),
        ErrorInfo::Runtime { msg } => PyRuntimeError::new_err(msg),
    }
}

fn pwrite_from_ptr_e(
    fd: RawFd,
    mut offset: u64,
    mut ptr: *const u8,
    mut len: usize,
) -> Result<(), ErrorInfo> {
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
            return Err(ErrorInfo::Os {
                errno: errno(),
                msg: "pwrite failed".to_string(),
            });
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

fn pread_into_e(fd: RawFd, offset: u64, mut dst: *mut u8, mut size: usize) -> Result<(), ErrorInfo> {
    let mut off = offset;
    while size > 0 {
        let n = unsafe {
            libc::pread(
                fd,
                dst as *mut libc::c_void,
                size,
                off as libc::off_t,
            )
        };
        if n < 0 {
            return Err(ErrorInfo::Os {
                errno: errno(),
                msg: "pread failed".to_string(),
            });
        }
        if n == 0 {
            return Err(ErrorInfo::Runtime {
                msg: "unexpected EOF".to_string(),
            });
        }
        let n = n as usize;
        unsafe {
            dst = dst.add(n);
        }
        off += n as u64;
        size -= n;
    }
    Ok(())
}

fn aligned_buf_new_e(len: usize, align: usize) -> Result<AlignedBuf, ErrorInfo> {
    let mut p: *mut libc::c_void = std::ptr::null_mut();
    let rc = unsafe { libc::posix_memalign(&mut p as *mut *mut libc::c_void, align, len) };
    if rc != 0 {
        return Err(ErrorInfo::Runtime {
            msg: format!("posix_memalign failed rc={rc}"),
        });
    }
    if p.is_null() {
        return Err(ErrorInfo::Runtime {
            msg: "posix_memalign returned null".to_string(),
        });
    }
    Ok(AlignedBuf { ptr: p as *mut u8, len, align })
}

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

fn pwrite_all(fd: RawFd, mut offset: u64, mut buf: &[u8]) -> Result<(), PyErr> {
    while !buf.is_empty() {
        // SAFETY: pwrite reads from buf pointer for buf.len bytes.
        let n = unsafe {
            libc::pwrite(
                fd,
                buf.as_ptr() as *const libc::c_void,
                buf.len(),
                offset as libc::off_t,
            )
        };
        if n < 0 {
            return Err(os_err("pwrite failed"));
        }
        let n = n as usize;
        offset += n as u64;
        buf = &buf[n..];
    }
    Ok(())
}

fn pwrite_from_ptr(fd: RawFd, mut offset: u64, mut ptr: *const u8, mut len: usize) -> Result<(), PyErr> {
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

fn pread_exact(fd: RawFd, offset: u64, size: usize) -> Result<Vec<u8>, PyErr> {
    let mut out = vec![0u8; size];
    let mut read = 0usize;
    while read < size {
        // SAFETY: pread writes into out[read..] for remaining bytes.
        let n = unsafe {
            libc::pread(
                fd,
                out[read..].as_mut_ptr() as *mut libc::c_void,
                size - read,
                (offset as libc::off_t) + (read as libc::off_t),
            )
        };
        if n < 0 {
            return Err(os_err("pread failed"));
        }
        if n == 0 {
            return Err(PyRuntimeError::new_err("unexpected EOF"));
        }
        read += n as usize;
    }
    Ok(out)
}

fn pread_into(fd: RawFd, offset: u64, mut dst: *mut u8, mut size: usize) -> Result<(), PyErr> {
    let mut off = offset;
    while size > 0 {
        // SAFETY: pread writes into dst for size bytes.
        let n = unsafe {
            libc::pread(
                fd,
                dst as *mut libc::c_void,
                size,
                off as libc::off_t,
            )
        };
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

struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    align: usize,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Result<Self, PyErr> {
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        // SAFETY: posix_memalign writes to p.
        let rc = unsafe { libc::posix_memalign(&mut p as *mut *mut libc::c_void, align, len) };
        if rc != 0 {
            return Err(PyRuntimeError::new_err(format!("posix_memalign failed rc={rc}")));
        }
        if p.is_null() {
            return Err(PyRuntimeError::new_err("posix_memalign returned null"));
        }
        Ok(Self { ptr: p as *mut u8, len, align })
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

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

fn get_pybuffer<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    writable: bool,
) -> Result<pyo3::ffi::Py_buffer, PyErr> {
    // SAFETY: PyObject_GetBuffer follows CPython buffer protocol.
    unsafe {
        let mut view: pyo3::ffi::Py_buffer = std::mem::zeroed();
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

fn release_pybuffer(mut view: pyo3::ffi::Py_buffer) {
    // SAFETY: view was created by PyObject_GetBuffer.
    unsafe {
        pyo3::ffi::PyBuffer_Release(&mut view);
    }
}

fn pick_fd(fds: &[RawFd]) -> RawFd {
    if fds.len() == 1 {
        return fds[0];
    }
    // SAFETY: sched_getcpu is a libc call; returns -1 on error.
    let cpu = unsafe { libc::sched_getcpu() };
    if cpu < 0 {
        return fds[0];
    }
    let idx = (cpu as usize) % fds.len();
    fds[idx]
}

fn pin_thread_to_cpu(cpu: usize) {
    // Best-effort pinning (Linux).
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let tid = libc::pthread_self();
        let _rc = libc::pthread_setaffinity_np(tid, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[derive(Clone, Copy)]
enum Priority {
    High,
    Medium,
    Low,
}

enum TaskKind {
    Pwrite {
        offset: u64,
        payload_len: usize,
        total_len: usize,
    },
    Pread {
        offset: u64,
        payload_len: usize,
        total_len: usize,
    },
}

struct HeldPyBuffer(pyo3::ffi::Py_buffer);

// We treat Py_buffer as an owned handle whose underlying PyObject is kept alive
// by the buffer protocol; we only touch/release it under the GIL.
unsafe impl Send for HeldPyBuffer {}
unsafe impl Sync for HeldPyBuffer {}

impl HeldPyBuffer {
    fn ptr_ro(&self) -> *const u8 {
        self.0.buf as *const u8
    }
    fn ptr_rw(&self) -> *mut u8 {
        self.0.buf as *mut u8
    }
    fn len(&self) -> usize {
        self.0.len as usize
    }
    fn readonly(&self) -> i32 {
        self.0.readonly
    }
    fn release(mut self) {
        unsafe { pyo3::ffi::PyBuffer_Release(&mut self.0) };
    }
}

struct Task {
    prio: Priority,
    kind: TaskKind,
    buf: HeldPyBuffer,
    future: Option<Py<PyAny>>, // concurrent.futures.Future
    batch: Option<Arc<Mutex<BatchState>>>,
}

struct BatchState {
    remaining: usize,
    done: bool,
    future: Py<PyAny>, // concurrent.futures.Future
}

struct WorkerQueue {
    high: VecDeque<Task>,
    mid: VecDeque<Task>,
    low: VecDeque<Task>,
    stop: bool,
}

struct WorkerState {
    q: Mutex<WorkerQueue>,
    cv: Condvar,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            q: Mutex::new(WorkerQueue {
                high: VecDeque::new(),
                mid: VecDeque::new(),
                low: VecDeque::new(),
                stop: false,
            }),
            cv: Condvar::new(),
        }
    }

    fn push(&self, t: Task) {
        let mut g = self.q.lock().unwrap();
        match t.prio {
            Priority::High => g.high.push_back(t),
            Priority::Medium => g.mid.push_back(t),
            Priority::Low => g.low.push_back(t),
        }
        self.cv.notify_one();
    }

    fn pop(&self) -> Option<Task> {
        let mut g = self.q.lock().unwrap();
        loop {
            if g.stop {
                return None;
            }
            if let Some(t) = g.high.pop_front() {
                return Some(t);
            }
            if let Some(t) = g.mid.pop_front() {
                return Some(t);
            }
            if let Some(t) = g.low.pop_front() {
                return Some(t);
            }
            g = self.cv.wait(g).unwrap();
        }
    }

    fn shutdown(&self) {
        let mut g = self.q.lock().unwrap();
        g.stop = true;
        self.cv.notify_all();
    }
}

fn make_concurrent_future(py: Python<'_>) -> PyResult<Py<PyAny>> {
    // Lazily import concurrent.futures.Future
    let cf = py.import("concurrent.futures")?;
    let cls = cf.getattr("Future")?;
    let fut = cls.call0()?;
    Ok(fut.into())
}

fn future_set_result(py: Python<'_>, fut: &Py<PyAny>) {
    let _ = fut.bind(py).call_method1("set_result", (py.None(),));
}

fn future_set_exception(py: Python<'_>, fut: &Py<PyAny>, err: PyErr) {
    let _ = fut.bind(py).call_method1("set_exception", (err,));
}

struct Completion {
    buf: HeldPyBuffer,
    future: Option<Py<PyAny>>,
    batch: Option<Arc<Mutex<BatchState>>>,
    err: Option<ErrorInfo>,
}

struct CompletionQueue {
    items: VecDeque<Completion>,
    stop: bool,
}

struct CompletionState {
    q: Mutex<CompletionQueue>,
    cv: Condvar,
}

impl CompletionState {
    fn new() -> Self {
        Self {
            q: Mutex::new(CompletionQueue {
                items: VecDeque::new(),
                stop: false,
            }),
            cv: Condvar::new(),
        }
    }

    fn push(&self, c: Completion) {
        let mut g = self.q.lock().unwrap();
        g.items.push_back(c);
        self.cv.notify_one();
    }

    fn pop_batch(&self) -> Option<Vec<Completion>> {
        let mut g = self.q.lock().unwrap();
        loop {
            if g.stop && g.items.is_empty() {
                return None;
            }
            if !g.items.is_empty() {
                let mut out = Vec::with_capacity(g.items.len());
                while let Some(c) = g.items.pop_front() {
                    out.push(c);
                }
                return Some(out);
            }
            g = self.cv.wait(g).unwrap();
        }
    }

    fn shutdown(&self) {
        let mut g = self.q.lock().unwrap();
        g.stop = true;
        self.cv.notify_all();
    }
}

#[pyclass]
struct RawBlockDevice {
    fd: RawFd,
    size: u64,
    closed: bool,
    use_odirect: bool,
    alignment: usize,
}

#[pymethods]
impl RawBlockDevice {
    #[new]
    #[pyo3(signature=(path, writable, use_odirect=false, alignment=4096))]
    fn new(path: String, writable: bool, use_odirect: bool, alignment: usize) -> PyResult<Self> {
        let cpath = CString::new(path.clone()).map_err(|_| PyValueError::new_err("path contains NUL"))?;
        let mut flags = if writable { libc::O_RDWR } else { libc::O_RDONLY };
        if use_odirect {
            flags |= libc::O_DIRECT;
        }
        // SAFETY: open returns fd or -1.
        let fd = unsafe { libc::open(cpath.as_ptr(), flags) };
        if fd < 0 {
            return Err(os_err("open failed"));
        }
        let size = fd_size_bytes(fd)?;
        Ok(Self { fd, size, closed: false, use_odirect, alignment })
    }

    fn size_bytes(&self) -> PyResult<u64> {
        Ok(self.size)
    }

    fn pread<'py>(&self, py: Python<'py>, offset: u64, size: usize) -> PyResult<Bound<'py, PyBytes>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("device is closed"));
        }
        let fd = self.fd;
        if !self.use_odirect {
            let data = py.allow_threads(move || pread_exact(fd, offset, size))?;
            return Ok(PyBytes::new(py, &data));
        }

        let align = self.alignment;
        if align == 0 {
            return Err(PyValueError::new_err("alignment must be > 0"));
        }
        if (offset as usize) % align != 0 {
            return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
        }
        let total = round_up(size, align);
        let data = py.allow_threads(move || {
            let bounce = AlignedBuf::new(total, align)?;
            pread_into(fd, offset, bounce.as_mut_ptr(), total)?;
            // SAFETY: bounce contains total bytes; copy only requested size.
            let slice = unsafe { slice::from_raw_parts(bounce.as_ptr(), size) };
            Ok::<Vec<u8>, PyErr>(slice.to_vec())
        })?;
        Ok(PyBytes::new(py, &data))
    }

    fn pwrite(&self, py: Python<'_>, offset: u64, data: &[u8]) -> PyResult<()> {
        if self.closed {
            return Err(PyRuntimeError::new_err("device is closed"));
        }
        let fd = self.fd;
        if !self.use_odirect {
            let buf = data.to_vec();
            py.allow_threads(move || pwrite_all(fd, offset, &buf))?;
            return Ok(());
        }

        let align = self.alignment;
        if (offset as usize) % align != 0 {
            return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
        }
        let payload_len = data.len();
        let total_len = round_up(payload_len, align);
        let buf = data.to_vec();
        py.allow_threads(move || {
            let bounce = AlignedBuf::new(total_len, align)?;
            // SAFETY: bounce is total_len, buf is payload_len.
            unsafe {
                libc::memcpy(
                    bounce.as_mut_ptr() as *mut libc::c_void,
                    buf.as_ptr() as *const libc::c_void,
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
        })?;
        Ok(())
    }

    /// Zero-copy write: write the bytes from any Python buffer object.
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

        let payload_len = payload_len.unwrap_or(buf_len);
        if payload_len > buf_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err("payload_len exceeds buffer length"));
        }
        let mut total_len = total_len.unwrap_or(payload_len);
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
        if total_len % align != 0 && self.use_odirect {
            release_pybuffer(view);
            return Err(PyValueError::new_err("invalid alignment"));
        }

        // If padding is requested (total_len > payload_len), always use bounce.
        // For O_DIRECT we also always use bounce (Python buffer alignment is not guaranteed).
        let ptr_usize = ptr as usize;
        let res = py.allow_threads(move || {
            let src = ptr_usize as *const u8;
            if total_len == payload_len && !self.use_odirect {
                // direct write without padding
                return pwrite_from_ptr(fd, offset, src, payload_len);
            }
            // bounce + optional pad zeros
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
        release_pybuffer(view);
        res?;
        Ok(())
    }

    /// Zero-copy read: read exactly `size` bytes into a writable Python buffer.
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

        let mut total_len = total_len.unwrap_or(payload_len);
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

        let dst_usize = ptr as usize;
        let res = py.allow_threads(move || {
            let dst = dst_usize as *mut u8;
            if total_len == payload_len && !self.use_odirect {
                return pread_into(fd, offset, dst, payload_len);
            }
            // bounce read then copy payload_len into dst
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

#[pyclass]
struct RawBlockDevicePool {
    fds: Vec<RawFd>,
    size: u64,
    closed: bool,
    use_odirect: bool,
    alignment: usize,
}

#[pymethods]
impl RawBlockDevicePool {
    #[new]
    #[pyo3(signature=(path, writable, num_fds=None, use_odirect=false, alignment=4096))]
    fn new(
        path: String,
        writable: bool,
        num_fds: Option<usize>,
        use_odirect: bool,
        alignment: usize,
    ) -> PyResult<Self> {
        let cpath =
            CString::new(path).map_err(|_| PyValueError::new_err("path contains NUL"))?;
        let mut flags = if writable { libc::O_RDWR } else { libc::O_RDONLY };
        if use_odirect {
            flags |= libc::O_DIRECT;
        }
        let n = num_fds.unwrap_or(0);
        let n = if n == 0 { cpu_count() } else { n };
        let n = std::cmp::max(1, n);

        let mut fds: Vec<RawFd> = Vec::with_capacity(n);
        for _ in 0..n {
            // SAFETY: open returns fd or -1.
            let fd = unsafe { libc::open(cpath.as_ptr(), flags) };
            if fd < 0 {
                // Cleanup already opened fds.
                for ofd in fds.drain(..) {
                    unsafe {
                        libc::close(ofd);
                    }
                }
                return Err(os_err("open failed"));
            }
            fds.push(fd);
        }
        let size = fd_size_bytes(fds[0])?;
        Ok(Self {
            fds,
            size,
            closed: false,
            use_odirect,
            alignment,
        })
    }

    fn size_bytes(&self) -> PyResult<u64> {
        Ok(self.size)
    }

    fn pread<'py>(
        &self,
        py: Python<'py>,
        offset: u64,
        size: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("device is closed"));
        }
        let fd = pick_fd(&self.fds);
        if !self.use_odirect {
            let data = py.allow_threads(move || pread_exact(fd, offset, size))?;
            return Ok(PyBytes::new(py, &data));
        }
        let align = self.alignment;
        if align == 0 {
            return Err(PyValueError::new_err("alignment must be > 0"));
        }
        if (offset as usize) % align != 0 {
            return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
        }
        let total = round_up(size, align);
        let data = py.allow_threads(move || {
            let bounce = AlignedBuf::new(total, align)?;
            pread_into(fd, offset, bounce.as_mut_ptr(), total)?;
            let slice = unsafe { slice::from_raw_parts(bounce.as_ptr(), size) };
            Ok::<Vec<u8>, PyErr>(slice.to_vec())
        })?;
        Ok(PyBytes::new(py, &data))
    }

    fn pwrite(&self, py: Python<'_>, offset: u64, data: &[u8]) -> PyResult<()> {
        if self.closed {
            return Err(PyRuntimeError::new_err("device is closed"));
        }
        let fd = pick_fd(&self.fds);
        if !self.use_odirect {
            let buf = data.to_vec();
            py.allow_threads(move || pwrite_all(fd, offset, &buf))?;
            return Ok(());
        }
        let align = self.alignment;
        if (offset as usize) % align != 0 {
            return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
        }
        let payload_len = data.len();
        let total_len = round_up(payload_len, align);
        let buf = data.to_vec();
        py.allow_threads(move || {
            let bounce = AlignedBuf::new(total_len, align)?;
            unsafe {
                libc::memcpy(
                    bounce.as_mut_ptr() as *mut libc::c_void,
                    buf.as_ptr() as *const libc::c_void,
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
        })?;
        Ok(())
    }

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
        let fd = pick_fd(&self.fds);

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

        let ptr_usize = ptr as usize;
        let res = py.allow_threads(move || {
            let src = ptr_usize as *const u8;
            if total_len == payload_len && !self.use_odirect {
                return pwrite_from_ptr(fd, offset, src, payload_len);
            }
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
        release_pybuffer(view);
        res?;
        Ok(())
    }

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
        let fd = pick_fd(&self.fds);

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

        let dst_usize = ptr as usize;
        let res = py.allow_threads(move || {
            let dst = dst_usize as *mut u8;
            if total_len == payload_len && !self.use_odirect {
                return pread_into(fd, offset, dst, payload_len);
            }
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

    fn close(&mut self) -> PyResult<()> {
        if !self.closed {
            for fd in self.fds.drain(..) {
                unsafe {
                    libc::close(fd);
                }
            }
            self.closed = true;
        }
        Ok(())
    }
}

impl Drop for RawBlockDevicePool {
    fn drop(&mut self) {
        if !self.closed {
            for fd in self.fds.drain(..) {
                unsafe {
                    libc::close(fd);
                }
            }
            self.closed = true;
        }
    }
}

#[pyclass]
struct RawBlockScheduler {
    fds: Vec<RawFd>,
    size: u64,
    closed: bool,
    use_odirect: bool,
    alignment: usize,
    workers: Vec<Arc<WorkerState>>,
    threads: Vec<thread::JoinHandle<()>>,
    completion: Arc<CompletionState>,
    completion_thread: Option<thread::JoinHandle<()>>,
}

#[pymethods]
impl RawBlockScheduler {
    #[new]
    #[pyo3(signature=(path, writable, num_workers=None, num_fds=None, use_odirect=false, alignment=4096))]
    fn new(
        path: String,
        writable: bool,
        num_workers: Option<usize>,
        num_fds: Option<usize>,
        use_odirect: bool,
        alignment: usize,
    ) -> PyResult<Self> {
        let cpath =
            CString::new(path).map_err(|_| PyValueError::new_err("path contains NUL"))?;
        let mut flags = if writable { libc::O_RDWR } else { libc::O_RDONLY };
        if use_odirect {
            flags |= libc::O_DIRECT;
        }

        let n_workers = num_workers.unwrap_or(0);
        let n_workers = if n_workers == 0 { cpu_count() } else { n_workers };
        let n_workers = std::cmp::max(1, n_workers);

        let n_fds = num_fds.unwrap_or(0);
        let n_fds = if n_fds == 0 { n_workers } else { n_fds };
        let n_fds = std::cmp::max(1, n_fds);

        let mut fds: Vec<RawFd> = Vec::with_capacity(n_fds);
        for _ in 0..n_fds {
            let fd = unsafe { libc::open(cpath.as_ptr(), flags) };
            if fd < 0 {
                for ofd in fds.drain(..) {
                    unsafe { libc::close(ofd) };
                }
                return Err(os_err("open failed"));
            }
            fds.push(fd);
        }
        let size = fd_size_bytes(fds[0])?;

        let mut workers: Vec<Arc<WorkerState>> = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            workers.push(Arc::new(WorkerState::new()));
        }

        let completion = Arc::new(CompletionState::new());
        let completion_clone = completion.clone();
        let completion_thread = thread::spawn(move || {
            // Drain completion queue in batches; process each batch under one GIL acquisition.
            while let Some(batch) = completion_clone.pop_batch() {
                Python::with_gil(|py| {
                    for c in batch {
                        c.buf.release();
                        if let Some(errinfo) = c.err.clone() {
                            let pyerr = errorinfo_to_pyerr(errinfo);
                            if let Some(fut) = &c.future {
                                future_set_exception(py, fut, pyerr.clone_ref(py));
                            }
                            if let Some(bs) = &c.batch {
                                let mut st = bs.lock().unwrap();
                                if !st.done {
                                    st.done = true;
                                    future_set_exception(py, &st.future, pyerr);
                                }
                            }
                        } else {
                            if let Some(fut) = &c.future {
                                future_set_result(py, fut);
                            }
                            if let Some(bs) = &c.batch {
                                let mut st = bs.lock().unwrap();
                                if !st.done {
                                    if st.remaining > 0 {
                                        st.remaining -= 1;
                                    }
                                    if st.remaining == 0 {
                                        st.done = true;
                                        future_set_result(py, &st.future);
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });

        let mut threads: Vec<thread::JoinHandle<()>> = Vec::with_capacity(n_workers);
        for i in 0..n_workers {
            let w = workers[i].clone();
            let fds_clone = fds.clone();
            let use_odirect = use_odirect;
            let alignment = alignment;
            let completion = completion.clone();
            threads.push(thread::spawn(move || {
                // best-effort cpu pin
                pin_thread_to_cpu(i % cpu_count());
                loop {
                    let task = w.pop();
                    if task.is_none() {
                        break;
                    }
                    let task = task.unwrap();
                    let fd = pick_fd(&fds_clone);
                    let res: Result<(), ErrorInfo> = (|| match task.kind {
                        TaskKind::Pwrite { offset, payload_len, total_len } => {
                            // Use GIL only for buffer release & future completion; I/O without GIL.
                            // pwrite_from_ptr needs pointer.
                            let ptr = task.buf.ptr_ro();
                            if ptr.is_null() {
                                Err(ErrorInfo::Value { msg: "null buffer pointer".to_string() })
                            } else if use_odirect {
                                if (offset as usize) % alignment != 0 || total_len % alignment != 0 {
                                    Err(ErrorInfo::Value { msg: "O_DIRECT alignment error".to_string() })
                                } else {
                                    // Always bounce for O_DIRECT to ensure alignment.
                                    let bounce = aligned_buf_new_e(total_len, alignment)?;
                                    unsafe {
                                        libc::memcpy(
                                            bounce.as_mut_ptr() as *mut libc::c_void,
                                            ptr as *const libc::c_void,
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
                                    pwrite_from_ptr_e(fd, offset, bounce.as_ptr(), total_len)
                                }
                            } else {
                                pwrite_from_ptr_e(fd, offset, ptr, payload_len)
                            }
                        }
                        TaskKind::Pread { offset, payload_len, total_len } => {
                            let ptr = task.buf.ptr_rw();
                            if ptr.is_null() {
                                Err(ErrorInfo::Value { msg: "null buffer pointer".to_string() })
                            } else if use_odirect {
                                if (offset as usize) % alignment != 0 || total_len % alignment != 0 {
                                    Err(ErrorInfo::Value { msg: "O_DIRECT alignment error".to_string() })
                                } else {
                                    let bounce = aligned_buf_new_e(total_len, alignment)?;
                                    pread_into_e(fd, offset, bounce.as_mut_ptr(), total_len)?;
                                    unsafe {
                                        libc::memcpy(
                                            ptr as *mut libc::c_void,
                                            bounce.as_ptr() as *const libc::c_void,
                                            payload_len,
                                        );
                                    }
                                    Ok(())
                                }
                            } else {
                                pread_into_e(fd, offset, ptr, payload_len)
                            }
                        }
                    })();

                    completion.push(Completion {
                        buf: task.buf,
                        future: task.future,
                        batch: task.batch,
                        err: res.err(),
                    });
                }
            }));
        }

        Ok(Self {
            fds,
            size,
            closed: false,
            use_odirect,
            alignment,
            workers,
            threads,
            completion,
            completion_thread: Some(completion_thread),
        })
    }

    fn size_bytes(&self) -> PyResult<u64> {
        Ok(self.size)
    }

    #[pyo3(signature=(offset, data, payload_len, total_len, priority=2))]
    fn submit_pwrite_from_buffer(
        &self,
        py: Python<'_>,
        offset: u64,
        data: &Bound<'_, PyAny>,
        payload_len: usize,
        total_len: usize,
        priority: i32,
    ) -> PyResult<Py<PyAny>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("scheduler is closed"));
        }
        let view = get_pybuffer(py, data, false)?;
        if payload_len > view.len as usize {
            release_pybuffer(view);
            return Err(PyValueError::new_err("payload_len exceeds buffer length"));
        }
        let fut = make_concurrent_future(py)?;
        let prio = match priority {
            0 => Priority::High,
            1 => Priority::Medium,
            _ => Priority::Low,
        };
        // Choose worker based on current CPU of submitter (per-cpu submission).
        let cpu = unsafe { libc::sched_getcpu() };
        let idx = if cpu < 0 {
            0
        } else {
            (cpu as usize) % self.workers.len()
        };
        self.workers[idx].push(Task {
            prio,
            kind: TaskKind::Pwrite {
                offset,
                payload_len,
                total_len,
            },
            buf: HeldPyBuffer(view),
            future: Some(fut.clone_ref(py)),
            batch: None,
        });
        Ok(fut)
    }

    #[pyo3(signature=(offset, out, payload_len, total_len, priority=0))]
    fn submit_pread_into(
        &self,
        py: Python<'_>,
        offset: u64,
        out: &Bound<'_, PyAny>,
        payload_len: usize,
        total_len: usize,
        priority: i32,
    ) -> PyResult<Py<PyAny>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("scheduler is closed"));
        }
        let view = get_pybuffer(py, out, true)?;
        if view.readonly != 0 {
            release_pybuffer(view);
            return Err(PyValueError::new_err("output buffer is readonly"));
        }
        if payload_len > view.len as usize {
            release_pybuffer(view);
            return Err(PyValueError::new_err("output buffer too small"));
        }
        let fut = make_concurrent_future(py)?;
        let prio = match priority {
            0 => Priority::High,
            1 => Priority::Medium,
            _ => Priority::Low,
        };
        let cpu = unsafe { libc::sched_getcpu() };
        let idx = if cpu < 0 {
            0
        } else {
            (cpu as usize) % self.workers.len()
        };
        self.workers[idx].push(Task {
            prio,
            kind: TaskKind::Pread {
                offset,
                payload_len,
                total_len,
            },
            buf: HeldPyBuffer(view),
            future: Some(fut.clone_ref(py)),
            batch: None,
        });
        Ok(fut)
    }

    #[pyo3(signature=(requests, priority=2))]
    fn submit_pwritev_from_buffers(
        &self,
        py: Python<'_>,
        requests: &Bound<'_, PyAny>,
        priority: i32,
    ) -> PyResult<Py<PyAny>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("scheduler is closed"));
        }

        let fut = make_concurrent_future(py)?;
        let prio = match priority {
            0 => Priority::High,
            1 => Priority::Medium,
            _ => Priority::Low,
        };

        let mut parsed: Vec<(u64, HeldPyBuffer, usize, usize)> = Vec::new();
        let iter = requests.iter()?;
        for item in iter {
            let obj = item?;
            let tup = obj.downcast::<pyo3::types::PyTuple>()?;
            if tup.len() != 4 {
                for (_, b, _, _) in parsed.drain(..) {
                    b.release();
                }
                return Err(PyValueError::new_err(
                    "each request must be a 4-tuple (offset, data, payload_len, total_len)",
                ));
            }
            let offset: u64 = tup.get_item(0)?.extract()?;
            let data = tup.get_item(1)?;
            let payload_len: usize = tup.get_item(2)?.extract()?;
            let total_len: usize = tup.get_item(3)?.extract()?;
            let view = get_pybuffer(py, &data, false)?;
            let held = HeldPyBuffer(view);
            if payload_len > held.len() {
                for (_, b, _, _) in parsed.drain(..) {
                    b.release();
                }
                held.release();
                return Err(PyValueError::new_err("payload_len exceeds buffer length"));
            }
            if total_len < payload_len {
                for (_, b, _, _) in parsed.drain(..) {
                    b.release();
                }
                held.release();
                return Err(PyValueError::new_err("total_len must be >= payload_len"));
            }
            parsed.push((offset, held, payload_len, total_len));
        }

        if parsed.is_empty() {
            future_set_result(py, &fut);
            return Ok(fut);
        }

        let batch = Arc::new(Mutex::new(BatchState {
            remaining: parsed.len(),
            done: false,
            future: fut.clone_ref(py),
        }));

        // Choose worker based on current CPU of submitter (per-cpu submission).
        let cpu = unsafe { libc::sched_getcpu() };
        let idx = if cpu < 0 {
            0
        } else {
            (cpu as usize) % self.workers.len()
        };
        for (offset, held, payload_len, total_len) in parsed {
            self.workers[idx].push(Task {
                prio,
                kind: TaskKind::Pwrite {
                    offset,
                    payload_len,
                    total_len,
                },
                buf: held,
                future: None,
                batch: Some(batch.clone()),
            });
        }
        Ok(fut)
    }

    #[pyo3(signature=(requests, priority=0))]
    fn submit_preadv_into(
        &self,
        py: Python<'_>,
        requests: &Bound<'_, PyAny>,
        priority: i32,
    ) -> PyResult<Py<PyAny>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("scheduler is closed"));
        }

        let fut = make_concurrent_future(py)?;
        let prio = match priority {
            0 => Priority::High,
            1 => Priority::Medium,
            _ => Priority::Low,
        };

        let mut parsed: Vec<(u64, HeldPyBuffer, usize, usize)> = Vec::new();
        let iter = requests.iter()?;
        for item in iter {
            let obj = item?;
            let tup = obj.downcast::<pyo3::types::PyTuple>()?;
            if tup.len() != 4 {
                for (_, b, _, _) in parsed.drain(..) {
                    b.release();
                }
                return Err(PyValueError::new_err(
                    "each request must be a 4-tuple (offset, out, payload_len, total_len)",
                ));
            }
            let offset: u64 = tup.get_item(0)?.extract()?;
            let out = tup.get_item(1)?;
            let payload_len: usize = tup.get_item(2)?.extract()?;
            let total_len: usize = tup.get_item(3)?.extract()?;
            let view = get_pybuffer(py, &out, true)?;
            let held = HeldPyBuffer(view);
            if held.readonly() != 0 {
                for (_, b, _, _) in parsed.drain(..) {
                    b.release();
                }
                held.release();
                return Err(PyValueError::new_err("output buffer is readonly"));
            }
            if payload_len > held.len() {
                for (_, b, _, _) in parsed.drain(..) {
                    b.release();
                }
                held.release();
                return Err(PyValueError::new_err("output buffer too small"));
            }
            if total_len < payload_len {
                for (_, b, _, _) in parsed.drain(..) {
                    b.release();
                }
                held.release();
                return Err(PyValueError::new_err("total_len must be >= payload_len"));
            }
            parsed.push((offset, held, payload_len, total_len));
        }

        if parsed.is_empty() {
            future_set_result(py, &fut);
            return Ok(fut);
        }

        let batch = Arc::new(Mutex::new(BatchState {
            remaining: parsed.len(),
            done: false,
            future: fut.clone_ref(py),
        }));

        let cpu = unsafe { libc::sched_getcpu() };
        let idx = if cpu < 0 {
            0
        } else {
            (cpu as usize) % self.workers.len()
        };
        for (offset, held, payload_len, total_len) in parsed {
            self.workers[idx].push(Task {
                prio,
                kind: TaskKind::Pread {
                    offset,
                    payload_len,
                    total_len,
                },
                buf: held,
                future: None,
                batch: Some(batch.clone()),
            });
        }
        Ok(fut)
    }

    fn close(&mut self) -> PyResult<()> {
        if !self.closed {
            for w in &self.workers {
                w.shutdown();
            }
            for t in self.threads.drain(..) {
                let _ = t.join();
            }
            // Drain completions and stop completion thread.
            self.completion.shutdown();
            if let Some(t) = self.completion_thread.take() {
                let _ = t.join();
            }
            for fd in self.fds.drain(..) {
                unsafe { libc::close(fd) };
            }
            self.closed = true;
        }
        Ok(())
    }
}

impl Drop for RawBlockScheduler {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[pymodule]
fn lmcache_rust_raw_block_io(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RawBlockDevice>()?;
    m.add_class::<RawBlockDevicePool>()?;
    m.add_class::<RawBlockScheduler>()?;
    Ok(())
}

