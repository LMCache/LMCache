// SPDX-License-Identifier: Apache-2.0

fn main() {
    let config = pyo3_build_config::get();
    let executable = config
        .executable
        .as_deref()
        .expect("building lmcache-server requires a local CPython; set PYO3_PYTHON");
    println!("cargo:rustc-env=LMCACHE_PYTHON_EXECUTABLE={executable}");
    pyo3_build_config::add_python_framework_link_args();
}
