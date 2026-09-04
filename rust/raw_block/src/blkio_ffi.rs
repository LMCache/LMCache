// SPDX-License-Identifier: Apache-2.0

//! Raw FFI bindings for the libblkio C library.
//!
//! Only the subset of the API used by [`RawBlockDevice`](super::RawBlockDevice)
//! (when `io_engine = "libblkio"`) is declared here.  Types match
//! `<blkio.h>` from libblkio >= 1.3.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

// Opaque handles ----------------------------------------------------------

/// Opaque blkio instance.
#[repr(C)]
pub struct blkio {
    _opaque: [u8; 0],
}

/// Opaque blkio queue.
#[repr(C)]
pub struct blkioq {
    _opaque: [u8; 0],
}

// Structs -----------------------------------------------------------------

/// Memory region descriptor for buffer registration.
#[repr(C)]
pub struct blkio_mem_region {
    pub addr: *mut c_void,
    pub len: usize,
    pub iova: u64,
    pub fd_offset: i64,
    pub fd: c_int,
    pub flags: u32,
}

/// I/O completion result.
#[repr(C)]
pub struct blkio_completion {
    pub user_data: *mut c_void,
    pub error_msg: *const c_char,
    pub ret: c_int,
    pub reserved_: [u8; 12],
}

// Functions ---------------------------------------------------------------

extern "C" {
    // Lifecycle
    pub fn blkio_create(driver: *const c_char, bp: *mut *mut blkio) -> c_int;
    pub fn blkio_connect(b: *mut blkio) -> c_int;
    pub fn blkio_start(b: *mut blkio) -> c_int;
    pub fn blkio_destroy(bp: *mut *mut blkio);

    // Properties
    pub fn blkio_set_str(b: *mut blkio, name: *const c_char, value: *const c_char) -> c_int;
    pub fn blkio_set_bool(b: *mut blkio, name: *const c_char, value: bool) -> c_int;
    pub fn blkio_get_uint64(b: *mut blkio, name: *const c_char, value: *mut u64) -> c_int;

    // Memory regions
    pub fn blkio_map_mem_region(b: *mut blkio, region: *const blkio_mem_region) -> c_int;
    pub fn blkio_unmap_mem_region(b: *mut blkio, region: *const blkio_mem_region);

    // Queue
    pub fn blkio_get_queue(b: *mut blkio, index: c_int) -> *mut blkioq;

    // I/O submission
    pub fn blkioq_read(
        q: *mut blkioq,
        start: u64,
        buf: *mut c_void,
        len: usize,
        user_data: *mut c_void,
        flags: u32,
    );
    pub fn blkioq_write(
        q: *mut blkioq,
        start: u64,
        buf: *const c_void,
        len: usize,
        user_data: *mut c_void,
        flags: u32,
    );

    // I/O completion
    pub fn blkioq_do_io(
        q: *mut blkioq,
        completions: *mut blkio_completion,
        min_completions: c_int,
        max_completions: c_int,
        timeout: *mut libc::timespec,
    ) -> c_int;
}
