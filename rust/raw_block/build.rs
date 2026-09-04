// SPDX-License-Identifier: Apache-2.0

//! Build script for lmcache_rust_raw_block_io.
//!
//! When the `blkio` cargo feature is active, this locates libblkio via
//! pkg-config and emits the correct linker flags.

fn main() {
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    if cfg!(feature = "blkio") {
        // Try pkg-config first for best cross-distro support.
        let lib = pkg_config::Config::new()
            .atleast_version("1.3")
            .probe("blkio");

        match lib {
            Ok(_) => {} // pkg-config emitted the right -l / -L flags
            Err(_) => {
                // Fallback: assume libblkio is installed in system paths.
                println!("cargo:rustc-link-lib=blkio");
            }
        }
    }
}
