# Rust server entry point

`lmcache-server` embeds CPython and runs the existing `lmcache server` command
in the same process. This is an experimental bootstrap for native server work;
request handling, storage and shutdown remain in Python.

## Build and run

Use Linux or macOS, Cargo, and CPython 3.12 or 3.13 with development libraries
and the full LMCache server installed. On Linux the interpreter must provide
a linkable libpython (for a custom CPython build, use `--enable-shared`).

From the repository root, with that Python environment activated:

```bash
export PYO3_PYTHON="$(command -v python)"
cargo build --release --locked --manifest-path rust/server/Cargo.toml
rust/server/target/release/lmcache-server --help
rust/server/target/release/lmcache-server --host 127.0.0.1 --port 5555
```

All arguments are the same as `lmcache server`, including HTTP and transport
options. The binary uses the interpreter selected at build time, even when
launched outside its virtualenv. Keep that interpreter and environment in
place; rebuild to select a different environment. The binary is not a
standalone or relocatable server distribution. `PYTHONHOME`, `PYTHONPATH` and
other Python environment variables retain their usual CPython meanings.

## Tests

The executable contract tests need CPython and Cargo, without LMCache or a GPU:

```bash
export PYO3_PYTHON="$(command -v python)"
cargo test --locked --manifest-path rust/server/Cargo.toml
cargo fmt --manifest-path rust/server/Cargo.toml --check
cargo clippy --locked --manifest-path rust/server/Cargo.toml --all-targets -- -D warnings
```

See [the entry-point design](../../docs/design/v1/multiprocess/rust-entrypoint.md)
for the ownership boundary and validation scope.
