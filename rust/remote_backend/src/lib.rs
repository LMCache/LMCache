// SPDX-License-Identifier: Apache-2.0

//! Rust remote backend for LMCache.
//!
//! Exposes `RustRemoteBackend` PyO3 class that dynamically
//! loads a connector shared library at runtime via
//! dlopen/libloading.  The connector must export the C ABI
//! defined in `include/connector_api.h`.
//!
//! Performance-critical logic (put/get I/O, metadata index,
//! put-task dedup) lives entirely in Rust.  Python only
//! handles memory allocation (PyTorch tensor pool) and
//! async scheduling.

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::sync::Mutex;

// Buffer protocol flags (CPython C-API).
const PYBUF_WRITABLE: i32 = 0x0001;
const PYBUF_ND: i32 = 0x0008;
const PYBUF_STRIDES: i32 = 0x0010 | PYBUF_ND;
const PYBUF_ANY_CONTIGUOUS: i32 = 0x0080 | PYBUF_STRIDES;

fn get_pybuffer<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    writable: bool,
) -> Result<pyo3::ffi::Py_buffer, PyErr> {
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
    unsafe {
        pyo3::ffi::PyBuffer_Release(&mut view);
    }
}

// -------------------------------------------------------
// C ABI function signatures (connector_api.h)
// -------------------------------------------------------

type FnCreate = unsafe extern "C" fn(*const libc::c_char, libc::size_t) -> *mut libc::c_void;
type FnDestroy = unsafe extern "C" fn(*mut libc::c_void);
type FnExists = unsafe extern "C" fn(*mut libc::c_void, *const libc::c_char) -> i32;
type FnPut =
    unsafe extern "C" fn(*mut libc::c_void, *const libc::c_char, *const u8, libc::size_t) -> i32;
type FnGet = unsafe extern "C" fn(
    *mut libc::c_void,
    *const libc::c_char,
    *mut u8,
    libc::size_t,
    *mut libc::size_t,
) -> i32;
type FnRemove = unsafe extern "C" fn(*mut libc::c_void, *const libc::c_char) -> i32;
type FnFileSize = unsafe extern "C" fn(*mut libc::c_void, *const libc::c_char, *mut u64) -> i32;
type FnListKeys = unsafe extern "C" fn(
    *mut libc::c_void,
    *mut libc::c_char,
    libc::size_t,
    *mut libc::size_t,
) -> i32;

// -------------------------------------------------------
// ConnectorHandle
// -------------------------------------------------------

struct ConnectorHandle {
    _lib: libloading::Library,
    handle: *mut libc::c_void,
    fn_destroy: FnDestroy,
    fn_exists: FnExists,
    fn_put: FnPut,
    fn_get: FnGet,
    fn_remove: FnRemove,
    fn_file_size: FnFileSize,
    fn_list_keys: FnListKeys,
}

unsafe impl Send for ConnectorHandle {}
unsafe impl Sync for ConnectorHandle {}

impl ConnectorHandle {
    fn load(lib_path: &str, config_json: &str) -> Result<Self, PyErr> {
        let lib = unsafe { libloading::Library::new(lib_path) }.map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to load connector '{}': {}", lib_path, e))
        })?;

        unsafe {
            let fn_create: libloading::Symbol<FnCreate> = lib
                .get(b"connector_create")
                .map_err(|e| PyRuntimeError::new_err(format!("symbol connector_create: {e}")))?;
            let fn_destroy: FnDestroy = *lib
                .get::<FnDestroy>(b"connector_destroy")
                .map_err(|e| PyRuntimeError::new_err(format!("symbol connector_destroy: {e}")))?;
            let fn_exists: FnExists = *lib
                .get::<FnExists>(b"connector_exists")
                .map_err(|e| PyRuntimeError::new_err(format!("symbol connector_exists: {e}")))?;
            let fn_put: FnPut = *lib
                .get::<FnPut>(b"connector_put")
                .map_err(|e| PyRuntimeError::new_err(format!("symbol connector_put: {e}")))?;
            let fn_get: FnGet = *lib
                .get::<FnGet>(b"connector_get")
                .map_err(|e| PyRuntimeError::new_err(format!("symbol connector_get: {e}")))?;
            let fn_remove: FnRemove = *lib
                .get::<FnRemove>(b"connector_remove")
                .map_err(|e| PyRuntimeError::new_err(format!("symbol connector_remove: {e}")))?;
            let fn_file_size: FnFileSize = *lib
                .get::<FnFileSize>(b"connector_file_size")
                .map_err(|e| PyRuntimeError::new_err(format!("symbol connector_file_size: {e}")))?;
            let fn_list_keys: FnListKeys = *lib
                .get::<FnListKeys>(b"connector_list_keys")
                .map_err(|e| PyRuntimeError::new_err(format!("symbol connector_list_keys: {e}")))?;

            let cfg_bytes = config_json.as_bytes();
            let handle = fn_create(cfg_bytes.as_ptr() as *const libc::c_char, cfg_bytes.len());
            if handle.is_null() {
                return Err(PyRuntimeError::new_err("connector_create returned NULL"));
            }

            Ok(Self {
                _lib: lib,
                handle,
                fn_destroy,
                fn_exists,
                fn_put,
                fn_get,
                fn_remove,
                fn_file_size,
                fn_list_keys,
            })
        }
    }

    fn exists(&self, key: &CString) -> bool {
        unsafe { (self.fn_exists)(self.handle, key.as_ptr()) == 1 }
    }

    fn put(&self, key: &CString, data: *const u8, len: usize) -> i32 {
        unsafe { (self.fn_put)(self.handle, key.as_ptr(), data, len) }
    }

    fn get(&self, key: &CString, out: *mut u8, cap: usize, out_len: &mut usize) -> i32 {
        unsafe { (self.fn_get)(self.handle, key.as_ptr(), out, cap, out_len) }
    }

    fn remove(&self, key: &CString) -> i32 {
        unsafe { (self.fn_remove)(self.handle, key.as_ptr()) }
    }

    fn file_size(&self, key: &CString) -> Result<Option<u64>, PyErr> {
        let mut size: u64 = 0;
        let rc = unsafe { (self.fn_file_size)(self.handle, key.as_ptr(), &mut size) };
        match rc {
            0 => Ok(Some(size)),
            1 => Ok(None),
            _ => Err(PyIOError::new_err("connector_file_size failed")),
        }
    }

    fn list_keys(&self) -> Result<Vec<String>, PyErr> {
        let cap: usize = 1024 * 1024;
        let mut buf = vec![0u8; cap];
        let mut out_len: usize = 0;
        let rc = unsafe {
            (self.fn_list_keys)(
                self.handle,
                buf.as_mut_ptr() as *mut libc::c_char,
                cap,
                &mut out_len,
            )
        };
        if rc != 0 {
            return Err(PyIOError::new_err("connector_list_keys failed"));
        }
        let s = std::str::from_utf8(&buf[..out_len])
            .map_err(|_| PyIOError::new_err("connector_list_keys: invalid UTF-8"))?;
        Ok(s.split('\n')
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect())
    }
}

impl Drop for ConnectorHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.fn_destroy)(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

fn make_ckey(key: &str) -> Result<CString, PyErr> {
    CString::new(key).map_err(|_| PyValueError::new_err("key contains NUL byte"))
}

// -------------------------------------------------------
// Rust-side metadata index and put-task dedup
// -------------------------------------------------------

#[derive(Clone)]
struct ChunkMeta {
    data_len: usize,
}

struct BackendState {
    /// Metadata index: key -> chunk metadata.
    meta_index: HashMap<String, ChunkMeta>,
    /// In-flight put tasks for dedup.
    put_tasks: HashSet<String>,
}

// -------------------------------------------------------
// PyO3 class: RustRemoteBackend
// -------------------------------------------------------

/// Remote backend that delegates to a dynamically loaded
/// connector shared library.
///
/// All I/O, metadata indexing, and put-task deduplication
/// happen in Rust.  Python only handles memory allocation
/// (PyTorch tensor pool) and async scheduling.
#[pyclass]
struct RustRemoteBackend {
    conn: ConnectorHandle,
    state: Mutex<BackendState>,
}

#[pymethods]
impl RustRemoteBackend {
    #[new]
    fn new(connector_lib: &str, config_json: &str) -> PyResult<Self> {
        let conn = ConnectorHandle::load(connector_lib, config_json)?;
        Ok(Self {
            conn,
            state: Mutex::new(BackendState {
                meta_index: HashMap::new(),
                put_tasks: HashSet::new(),
            }),
        })
    }

    /// Check if a key exists in the connector.
    fn exists(&self, key: &str) -> PyResult<bool> {
        let ckey = make_ckey(key)?;
        Ok(self.conn.exists(&ckey))
    }

    /// Try to add key to the put-task set (dedup).
    ///
    /// Returns True if the key was added (not a dup).
    /// Returns False if already in-flight.
    fn try_add_put_task(&self, key: &str) -> PyResult<bool> {
        let mut st = self.state.lock().unwrap();
        Ok(st.put_tasks.insert(key.to_string()))
    }

    /// Remove key from put-task set.
    fn remove_put_task(&self, key: &str) -> PyResult<()> {
        let mut st = self.state.lock().unwrap();
        st.put_tasks.remove(key);
        Ok(())
    }

    /// Check if key is in the put-task set.
    fn in_put_tasks(&self, key: &str) -> PyResult<bool> {
        let st = self.state.lock().unwrap();
        Ok(st.put_tasks.contains(key))
    }

    /// Record chunk metadata after a successful put.
    fn record_meta(&self, key: &str, data_len: usize) -> PyResult<()> {
        let mut st = self.state.lock().unwrap();
        st.meta_index
            .insert(key.to_string(), ChunkMeta { data_len });
        Ok(())
    }

    /// Get the stored data length for a key, or None.
    fn get_data_len(&self, key: &str) -> PyResult<Option<usize>> {
        let st = self.state.lock().unwrap();
        Ok(st.meta_index.get(key).map(|m| m.data_len))
    }

    /// Write data for a key (GIL-released).
    ///
    /// Reads directly from the Python buffer's raw pointer
    /// (zero-copy).  The connector write happens entirely
    /// outside the GIL.
    fn put_blocking(&self, py: Python<'_>, key: &str, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let ckey = make_ckey(key)?;
        let view = get_pybuffer(py, data, false)?;
        let ptr = view.buf as *const u8;
        let buf_len = view.len as usize;
        if ptr.is_null() {
            release_pybuffer(view);
            return Err(PyValueError::new_err("null buffer pointer"));
        }
        let ptr_val = ptr as usize;
        let conn = &self.conn;
        let res = py.allow_threads(move || {
            let src = ptr_val as *const u8;
            let rc = conn.put(&ckey, src, buf_len);
            if rc != 0 {
                Err(PyIOError::new_err("connector_put failed"))
            } else {
                Ok(())
            }
        });
        release_pybuffer(view);
        res
    }

    /// Read data for a key into a writable buffer
    /// (GIL-released, zero-copy).
    ///
    /// Returns bytes read, or None if key not found.
    fn get_into(
        &self,
        py: Python<'_>,
        key: &str,
        out: &Bound<'_, PyAny>,
    ) -> PyResult<Option<usize>> {
        let ckey = make_ckey(key)?;
        let view = get_pybuffer(py, out, true)?;
        if view.readonly != 0 {
            release_pybuffer(view);
            return Err(PyValueError::new_err("output buffer is readonly"));
        }
        let cap = view.len as usize;
        let ptr = view.buf as *mut u8;
        if ptr.is_null() {
            release_pybuffer(view);
            return Err(PyValueError::new_err("null buffer pointer"));
        }
        let dst_val = ptr as usize;
        let conn = &self.conn;
        let res = py.allow_threads(move || {
            let dst = dst_val as *mut u8;
            let mut out_len: usize = 0;
            let rc = conn.get(&ckey, dst, cap, &mut out_len);
            match rc {
                0 => Ok(Some(out_len)),
                1 => Ok(None),
                _ => Err(PyIOError::new_err("connector_get failed")),
            }
        });
        release_pybuffer(view);
        res
    }

    /// Remove data for a key.
    fn remove(&self, key: &str) -> PyResult<bool> {
        let ckey = make_ckey(key)?;
        let rc = self.conn.remove(&ckey);
        match rc {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(PyIOError::new_err("connector_remove failed")),
        }
    }

    /// Get data size for a key.
    fn file_size(&self, key: &str) -> PyResult<Option<u64>> {
        let ckey = make_ckey(key)?;
        self.conn.file_size(&ckey)
    }

    /// List all keys.
    fn list_keys(&self) -> PyResult<Vec<String>> {
        self.conn.list_keys()
    }

    /// Close the backend.
    fn close(&mut self) -> PyResult<()> {
        Ok(())
    }
}

// -------------------------------------------------------
// Module
// -------------------------------------------------------

#[pymodule]
fn lmcache_rust_remote_backend_io(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustRemoteBackend>()?;
    Ok(())
}
