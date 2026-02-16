// SPDX-License-Identifier: Apache-2.0

//! Built-in filesystem connector for LMCache remote backend.
//!
//! Compiled as a standalone cdylib (`lmcache_connector_fs.so`)
//! that is loaded at runtime by `RustRemoteBackend` via dlopen.
//!
//! Implements the C ABI defined in `connector_api.h`.

use std::ffi::{CStr, CString};
use std::os::unix::io::RawFd;
use std::path::Path;

// -----------------------------------------------------------
// Internal FS connector state
// -----------------------------------------------------------

struct FsConnector {
    base_path: String,
    tmp_dir: Option<String>,
    use_odirect: bool,
    alignment: usize,
}

// -----------------------------------------------------------
// Helpers
// -----------------------------------------------------------

/// Validate and sanitize path to prevent path traversal attacks
fn validate_path(path: &str) -> Result<String, &'static str> {
    // Check for path traversal sequences
    if path.contains("..") {
        return Err("Path contains path traversal sequences (..)");
    }

    // Check for absolute paths (should be relative for safety)
    if path.starts_with('/') {
        return Err("Path must be relative, not absolute");
    }

    // Normalize the path
    let normalized = path.replace("//", "/");

    Ok(normalized)
}

#[allow(clippy::manual_div_ceil)]
fn round_up(x: usize, align: usize) -> usize {
    (x + align - 1) / align * align
}

fn errno_val() -> i32 {
    #[cfg(target_os = "linux")]
    unsafe {
        *libc::__errno_location()
    }
    #[cfg(target_os = "macos")]
    unsafe {
        *libc::__error()
    }
}

struct AlignedBuf {
    ptr: *mut u8,
    #[allow(dead_code)]
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Option<Self> {
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut p as *mut *mut libc::c_void, align, len) };
        if rc != 0 || p.is_null() {
            return None;
        }
        Some(Self {
            ptr: p as *mut u8,
            len,
        })
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::free(self.ptr as *mut libc::c_void);
            }
        }
    }
}

fn write_all(fd: RawFd, mut ptr: *const u8, mut len: usize) -> bool {
    while len > 0 {
        let n = unsafe { libc::write(fd, ptr as *const libc::c_void, len) };
        if n < 0 {
            return false;
        }
        let n = n as usize;
        unsafe { ptr = ptr.add(n) };
        len -= n;
    }
    true
}

fn read_all(fd: RawFd, mut dst: *mut u8, mut size: usize) -> bool {
    while size > 0 {
        let n = unsafe { libc::read(fd, dst as *mut libc::c_void, size) };
        if n <= 0 {
            return false;
        }
        let n = n as usize;
        unsafe { dst = dst.add(n) };
        size -= n;
    }
    true
}

// -----------------------------------------------------------
// FsConnector implementation
// -----------------------------------------------------------

impl FsConnector {
    fn new(config_json: &str) -> Option<Self> {
        // Minimal JSON parsing without serde dependency.
        // Expected keys: base_path, use_odirect, alignment,
        // tmp_subdir.
        let base_path = Self::json_str_value(config_json, "base_path")?;
        if base_path.is_empty() {
            return None;
        }

        // Validate base_path: must be absolute and not contain ..
        if base_path.contains("..") {
            eprintln!("base_path contains path traversal sequences: {}", base_path);
            return None;
        }

        let use_odirect = Self::json_bool_value(config_json, "use_odirect").unwrap_or(false);
        let alignment = Self::json_int_value(config_json, "alignment").unwrap_or(4096) as usize;
        let tmp_subdir = Self::json_str_value(config_json, "tmp_subdir");

        if std::fs::create_dir_all(&base_path).is_err() {
            return None;
        }

        let tmp_dir = tmp_subdir.filter(|s| !s.is_empty()).and_then(|sub| {
            // Validate tmp_subdir to prevent path traversal
            match validate_path(&sub) {
                Ok(safe_sub) => {
                    let full = format!("{}/{}", base_path, safe_sub);
                    let _ = std::fs::create_dir_all(&full);
                    Some(full)
                }
                Err(e) => {
                    eprintln!("Invalid tmp_subdir: {}", e);
                    None
                }
            }
        });

        Some(Self {
            base_path,
            tmp_dir,
            use_odirect,
            alignment,
        })
    }

    fn key_to_filename(key: &str) -> String {
        let safe = key.replace('/', "-SEP-");
        format!("{safe}.data")
    }

    fn file_path(&self, key: &str) -> String {
        format!("{}/{}", self.base_path, Self::key_to_filename(key))
    }

    fn tmp_path(&self, key: &str) -> String {
        let name = Self::key_to_filename(key);
        match &self.tmp_dir {
            Some(d) => format!("{d}/{name}"),
            None => {
                format!("{}/{name}.tmp", self.base_path)
            }
        }
    }

    fn exists(&self, key: &str) -> bool {
        Path::new(&self.file_path(key)).exists()
    }

    fn put(&self, key: &str, data: *const u8, data_len: usize) -> i32 {
        let fpath = self.file_path(key);
        let tpath = self.tmp_path(key);

        let c_tmp = match CString::new(tpath.as_str()) {
            Ok(c) => c,
            Err(_) => return -1,
        };
        let c_final = match CString::new(fpath.as_str()) {
            Ok(c) => c,
            Err(_) => return -1,
        };

        #[allow(unused_mut)]
        let mut flags = libc::O_CREAT | libc::O_WRONLY | libc::O_TRUNC;
        #[cfg(target_os = "linux")]
        if self.use_odirect {
            flags |= libc::O_DIRECT;
        }
        let fd = unsafe { libc::open(c_tmp.as_ptr(), flags, 0o644) };
        if fd < 0 {
            return -1;
        }

        let ok = if self.use_odirect {
            let aligned_len = round_up(data_len, self.alignment);
            match AlignedBuf::new(aligned_len, self.alignment) {
                Some(bounce) => {
                    unsafe {
                        libc::memcpy(
                            bounce.ptr as *mut libc::c_void,
                            data as *const libc::c_void,
                            data_len,
                        );
                        if aligned_len > data_len {
                            libc::memset(
                                bounce.ptr.add(data_len) as *mut libc::c_void,
                                0,
                                aligned_len - data_len,
                            );
                        }
                    }
                    let w = write_all(fd, bounce.ptr as *const u8, aligned_len);
                    if w {
                        // Truncate to exact size
                        unsafe {
                            libc::ftruncate(fd, data_len as libc::off_t);
                        }
                    }
                    w
                }
                None => {
                    unsafe { libc::close(fd) };
                    return -1;
                }
            }
        } else {
            write_all(fd, data, data_len)
        };

        unsafe { libc::close(fd) };
        if !ok {
            return -1;
        }

        // Atomic rename
        let rc = unsafe { libc::rename(c_tmp.as_ptr(), c_final.as_ptr()) };
        if rc != 0 {
            return -1;
        }
        0
    }

    fn get(&self, key: &str, out_buf: *mut u8, out_cap: usize, out_len: &mut usize) -> i32 {
        let fpath = self.file_path(key);
        let c_path = match CString::new(fpath.as_str()) {
            Ok(c) => c,
            Err(_) => return -1,
        };

        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::stat(c_path.as_ptr(), &mut st) };
        if rc != 0 {
            return if errno_val() == libc::ENOENT { 1 } else { -1 };
        }
        let file_size = st.st_size as usize;
        let read_size = std::cmp::min(file_size, out_cap);

        #[allow(unused_mut)]
        let mut flags = libc::O_RDONLY;
        #[cfg(target_os = "linux")]
        if self.use_odirect {
            flags |= libc::O_DIRECT;
        }
        let fd = unsafe { libc::open(c_path.as_ptr(), flags) };
        if fd < 0 {
            return -1;
        }

        let ok = if self.use_odirect {
            let aligned_len = round_up(read_size, self.alignment);
            match AlignedBuf::new(aligned_len, self.alignment) {
                Some(bounce) => {
                    let r = read_all(fd, bounce.ptr, aligned_len);
                    if r {
                        unsafe {
                            libc::memcpy(
                                out_buf as *mut libc::c_void,
                                bounce.ptr as *const libc::c_void,
                                read_size,
                            );
                        }
                    }
                    r
                }
                None => {
                    unsafe { libc::close(fd) };
                    return -1;
                }
            }
        } else {
            read_all(fd, out_buf, read_size)
        };

        unsafe { libc::close(fd) };
        if !ok {
            return -1;
        }
        *out_len = read_size;
        0
    }

    fn remove(&self, key: &str) -> i32 {
        let fpath = self.file_path(key);
        match std::fs::remove_file(&fpath) {
            Ok(()) => 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
            Err(_) => -1,
        }
    }

    fn file_size(&self, key: &str, out_size: &mut u64) -> i32 {
        let fpath = self.file_path(key);
        match std::fs::metadata(&fpath) {
            Ok(m) => {
                *out_size = m.len();
                0
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => 1,
            Err(_) => -1,
        }
    }

    fn list_keys(&self, out_buf: *mut u8, out_cap: usize, out_len: &mut usize) -> i32 {
        let entries = match std::fs::read_dir(&self.base_path) {
            Ok(e) => e,
            Err(_) => return -1,
        };
        let mut pos: usize = 0;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".data") {
                continue;
            }
            let bytes = name.as_bytes();
            let need = bytes.len() + 1; // +1 for '\n'
            if pos + need > out_cap {
                break;
            }
            unsafe {
                libc::memcpy(
                    out_buf.add(pos) as *mut libc::c_void,
                    bytes.as_ptr() as *const libc::c_void,
                    bytes.len(),
                );
                *out_buf.add(pos + bytes.len()) = b'\n';
            }
            pos += need;
        }
        *out_len = pos;
        0
    }

    // ---- Minimal JSON helpers (no serde) ----

    fn json_str_value(json: &str, key: &str) -> Option<String> {
        let pattern = format!("\"{}\"", key);
        let idx = json.find(&pattern)?;
        let rest = &json[idx + pattern.len()..];
        // skip whitespace and colon
        let rest = rest.trim_start();
        let rest = rest.strip_prefix(':')?;
        let rest = rest.trim_start();
        let rest = rest.strip_prefix('"')?;
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    }

    fn json_bool_value(json: &str, key: &str) -> Option<bool> {
        let pattern = format!("\"{}\"", key);
        let idx = json.find(&pattern)?;
        let rest = &json[idx + pattern.len()..];
        let rest = rest.trim_start();
        let rest = rest.strip_prefix(':')?;
        let rest = rest.trim_start();
        if rest.starts_with("true") {
            Some(true)
        } else if rest.starts_with("false") {
            Some(false)
        } else {
            None
        }
    }

    fn json_int_value(json: &str, key: &str) -> Option<i64> {
        let pattern = format!("\"{}\"", key);
        let idx = json.find(&pattern)?;
        let rest = &json[idx + pattern.len()..];
        let rest = rest.trim_start();
        let rest = rest.strip_prefix(':')?;
        let rest = rest.trim_start();
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '-')
            .unwrap_or(rest.len());
        rest[..end].parse().ok()
    }
}

// -----------------------------------------------------------
// C ABI exports
// -----------------------------------------------------------

/// # Safety
///
/// `config_json` must point to a valid UTF-8 string of
/// length `config_json_len`.
#[no_mangle]
pub unsafe extern "C" fn connector_create(
    config_json: *const libc::c_char,
    config_json_len: libc::size_t,
) -> *mut libc::c_void {
    if config_json.is_null() || config_json_len == 0 {
        return std::ptr::null_mut();
    }
    let slice = unsafe { std::slice::from_raw_parts(config_json as *const u8, config_json_len) };
    let json_str = match std::str::from_utf8(slice) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    match FsConnector::new(json_str) {
        Some(c) => Box::into_raw(Box::new(c)) as *mut libc::c_void,
        None => std::ptr::null_mut(),
    }
}

/// # Safety
///
/// `handle` must be a valid pointer returned by
/// `connector_create`.
#[no_mangle]
pub unsafe extern "C" fn connector_destroy(handle: *mut libc::c_void) {
    if !handle.is_null() {
        let _ = unsafe { Box::from_raw(handle as *mut FsConnector) };
    }
}

/// # Safety
///
/// `handle` and `key` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn connector_exists(
    handle: *mut libc::c_void,
    key: *const libc::c_char,
) -> i32 {
    let conn = unsafe { &*(handle as *const FsConnector) };
    let key_str = unsafe { CStr::from_ptr(key) }.to_str().unwrap_or("");
    if conn.exists(key_str) {
        1
    } else {
        0
    }
}

/// # Safety
///
/// `handle`, `key`, and `data` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn connector_put(
    handle: *mut libc::c_void,
    key: *const libc::c_char,
    data: *const u8,
    data_len: libc::size_t,
) -> i32 {
    let conn = unsafe { &*(handle as *const FsConnector) };
    let key_str = unsafe { CStr::from_ptr(key) }.to_str().unwrap_or("");
    conn.put(key_str, data, data_len)
}

/// # Safety
///
/// `handle`, `key`, `out_buf`, and `out_len` must be valid.
#[no_mangle]
pub unsafe extern "C" fn connector_get(
    handle: *mut libc::c_void,
    key: *const libc::c_char,
    out_buf: *mut u8,
    out_cap: libc::size_t,
    out_len: *mut libc::size_t,
) -> i32 {
    let conn = unsafe { &*(handle as *const FsConnector) };
    let key_str = unsafe { CStr::from_ptr(key) }.to_str().unwrap_or("");
    let len_ref = unsafe { &mut *out_len };
    conn.get(key_str, out_buf, out_cap, len_ref)
}

/// # Safety
///
/// `handle` and `key` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn connector_remove(
    handle: *mut libc::c_void,
    key: *const libc::c_char,
) -> i32 {
    let conn = unsafe { &*(handle as *const FsConnector) };
    let key_str = unsafe { CStr::from_ptr(key) }.to_str().unwrap_or("");
    conn.remove(key_str)
}

/// # Safety
///
/// `handle`, `key`, and `out_size` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn connector_file_size(
    handle: *mut libc::c_void,
    key: *const libc::c_char,
    out_size: *mut u64,
) -> i32 {
    let conn = unsafe { &*(handle as *const FsConnector) };
    let key_str = unsafe { CStr::from_ptr(key) }.to_str().unwrap_or("");
    let size_ref = unsafe { &mut *out_size };
    conn.file_size(key_str, size_ref)
}

/// # Safety
///
/// `handle`, `out_buf`, and `out_len` must be valid.
#[no_mangle]
pub unsafe extern "C" fn connector_list_keys(
    handle: *mut libc::c_void,
    out_buf: *mut libc::c_char,
    out_cap: libc::size_t,
    out_len: *mut libc::size_t,
) -> i32 {
    let conn = unsafe { &*(handle as *const FsConnector) };
    let len_ref = unsafe { &mut *out_len };
    conn.list_keys(out_buf as *mut u8, out_cap, len_ref)
}
