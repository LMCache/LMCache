// SPDX-License-Identifier: Apache-2.0

use pyo3::exceptions::{PyRuntimeError, PySystemExit};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn initialize_python() {
    let executable = CString::new(env!("LMCACHE_PYTHON_EXECUTABLE")).unwrap();
    let arguments: Vec<_> = std::env::args_os()
        .map(|arg| CString::new(arg.as_bytes()).unwrap().into_bytes_with_nul())
        .collect();
    let mut argv: Vec<_> = arguments.iter().map(|arg| arg.as_ptr().cast()).collect();

    // SAFETY: This process initializes CPython once, on its main thread. The
    // config APIs copy the argument buffers. SaveThread releases the initial
    // GIL before Rust starts waiting or joining its HTTP worker.
    unsafe {
        let mut config = std::mem::MaybeUninit::<pyo3::ffi::PyConfig>::uninit();
        pyo3::ffi::PyConfig_InitPythonConfig(config.as_mut_ptr());
        let mut config = config.assume_init();
        config.install_signal_handlers = 0;
        config.parse_argv = 0;
        let status = pyo3::ffi::PyConfig_SetBytesString(
            &mut config,
            std::ptr::addr_of_mut!(config.program_name),
            executable.as_ptr(),
        );
        if pyo3::ffi::PyStatus_Exception(status) != 0 {
            pyo3::ffi::Py_ExitStatusException(status);
        }
        let status = pyo3::ffi::PyConfig_SetBytesArgv(
            &mut config,
            argv.len().try_into().unwrap(),
            argv.as_mut_ptr(),
        );
        if pyo3::ffi::PyStatus_Exception(status) != 0 {
            pyo3::ffi::Py_ExitStatusException(status);
        }
        let status = pyo3::ffi::Py_InitializeFromConfig(&config);
        pyo3::ffi::PyConfig_Clear(&mut config);
        if pyo3::ffi::PyStatus_Exception(status) != 0 {
            pyo3::ffi::Py_ExitStatusException(status);
        }
        pyo3::ffi::PyEval_SaveThread();
    }
}

struct ServerRuntime {
    bridge: Py<PyAny>,
    http_server: Py<PyAny>,
    request_server: Py<PyAny>,
    engine: Py<PyAny>,
    event_bus: Py<PyAny>,
}

impl ServerRuntime {
    fn start() -> PyResult<Self> {
        Python::with_gil(|py| {
            let bridge = py.import("lmcache.v1.multiprocess.rust_server")?;
            let args = PyList::new(py, std::env::args_os().skip(1))?;
            let bootstrap = bridge.call_method1("prepare_server", (args,))?;
            let http_server = bootstrap.getattr("http_server")?.unbind();
            let result = bootstrap.call_method0("start_engine");
            let resources = match result {
                Ok(resources) => resources,
                Err(error) => {
                    if let Err(cleanup) = bridge.call_method0("shutdown_observability") {
                        eprintln!("Observability shutdown failed: {cleanup}");
                    }
                    return Err(error);
                }
            };
            let (request_server, engine, event_bus) = resources.extract()?;
            Ok(Self {
                bridge: bridge.into_any().unbind(),
                http_server,
                request_server,
                engine,
                event_bus,
            })
        })
    }

    fn close(self) -> PyResult<()> {
        Python::with_gil(|py| {
            let mut first_error = None;
            for (resource, method) in [
                (&self.event_bus, "stop"),
                (&self.request_server, "close"),
                (&self.engine, "close"),
                (&self.bridge, "shutdown_observability"),
            ] {
                if let Err(error) = resource.call_method0(py, method) {
                    eprintln!("Shutdown {method} failed: {error}");
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
            match first_error {
                Some(error) => Err(error),
                None => Ok(()),
            }
        })
    }

    fn run(self, shutdown: &AtomicUsize) -> PyResult<()> {
        let http_server = Python::with_gil(|py| self.http_server.clone_ref(py));
        let worker = thread::Builder::new()
            .name("lmcache-http".into())
            .spawn(move || Python::with_gil(|py| http_server.call_method0(py, "run").map(|_| ())));
        let worker = match worker {
            Ok(worker) => worker,
            Err(error) => {
                let cleanup = self.close();
                if let Err(error) = cleanup {
                    eprintln!("Server cleanup failed: {error}");
                }
                return Err(PyRuntimeError::new_err(error.to_string()));
            }
        };
        eprintln!("lmcache-server: Rust lifecycle running");
        while !worker.is_finished() {
            if shutdown.load(Ordering::Relaxed) != 0 {
                let started = Python::with_gil(|py| {
                    self.http_server.getattr(py, "started")?.extract::<bool>(py)
                })?;
                // Uvicorn skips its shutdown phase if should_exit is set before
                // startup completes. Let the HTTP lifespan finish starting first.
                if started {
                    eprintln!("lmcache-server: draining HTTP before closing MP resources");
                    Python::with_gil(|py| self.http_server.setattr(py, "should_exit", true))?;
                    break;
                }
            }
            thread::sleep(Duration::from_millis(100));
        }
        let result = worker
            .join()
            .unwrap_or_else(|_| Err(PyRuntimeError::new_err("HTTP worker panicked")));
        eprintln!("lmcache-server: closing event bus, request transport and engine");
        let cleanup = self.close();
        result.and(cleanup)
    }
}

fn main() {
    let shutdown = Arc::new(AtomicUsize::new(0));
    let mut signals = Vec::new();
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        match signal_hook::flag::register_usize(signal, Arc::clone(&shutdown), signal as usize) {
            Ok(id) => signals.push(id),
            Err(error) => {
                eprintln!("Cannot install signal handler: {error}");
                std::process::exit(1);
            }
        }
    }
    initialize_python();
    let result = ServerRuntime::start().and_then(|runtime| runtime.run(&shutdown));
    let status = Python::with_gil(|py| match result {
        Ok(()) => 0,
        Err(error) if error.is_instance_of::<PySystemExit>(py) => {
            let code = error.value(py).getattr("code").unwrap();
            if code.is_none() {
                0
            } else if let Ok(code) = code.extract::<i32>() {
                code
            } else {
                eprintln!("{code}");
                1
            }
        }
        Err(error) => {
            error.print(py);
            1
        }
    });
    // SAFETY: All Rust-owned Python handles and HTTP workers are gone. Acquire
    // the GIL on the initializing thread for the one interpreter finalization.
    let finalized = unsafe {
        pyo3::ffi::PyGILState_Ensure();
        pyo3::ffi::Py_FinalizeEx()
    };
    for id in signals {
        signal_hook::low_level::unregister(id);
    }
    std::process::exit(if finalized < 0 { 120 } else { status });
}
