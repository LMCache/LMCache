// SPDX-License-Identifier: Apache-2.0

use std::ffi::CString;
use std::os::raw::c_char;
use std::os::unix::ffi::OsStrExt;

fn main() {
    let prefix = [
        env!("LMCACHE_PYTHON_EXECUTABLE"),
        "-m",
        "lmcache.cli.main",
        "server",
    ];
    let mut arguments: Vec<_> = prefix
        .iter()
        .map(|argument| argument.as_bytes().to_vec())
        .chain(
            std::env::args_os()
                .skip(1)
                .map(|arg| arg.as_bytes().to_vec()),
        )
        .map(|argument| {
            CString::new(argument)
                .expect("OS arguments cannot contain NUL")
                .into_bytes_with_nul()
        })
        .collect();
    let argc = arguments.len().try_into().expect("too many arguments");
    let mut argv: Vec<_> = arguments
        .iter_mut()
        .map(|argument| argument.as_mut_ptr().cast::<c_char>())
        .collect();
    argv.push(std::ptr::null_mut());

    // SAFETY: CPython is initialized exactly once on the main thread. argv has
    // argc writable, NUL-terminated strings followed by a null pointer; every
    // buffer remains alive until Py_BytesMain finishes interpreter finalization.
    let status = unsafe { pyo3_ffi::Py_BytesMain(argc, argv.as_mut_ptr()) };
    std::process::exit(status);
}
