# Rust server entry point

## Scope and CLI contract

`lmcache-server [SERVER_ARGS...]` embeds CPython in the Rust executable and
runs `lmcache.cli.main` with `server` prepended to the supplied arguments.
Its flags, help, exit codes, HTTP API, request transport and shutdown behavior
come from the existing `lmcache server` command. It starts no Python child
process. Existing Python entry points remain available.

This first step establishes an executable for future native server work.
Request dispatch, storage state and lifecycle management still run in Python;
there is no performance claim or new Rust async runtime in this step.

```text
lmcache-server [args] (Rust process)
  -> CPython Py_BytesMain(-m lmcache.cli.main server [args])
     -> ServerCommand.execute
        -> run_http_server -> existing MP engine and storage manager
```

## Interpreter and ownership

The build selects CPython through `PYO3_PYTHON`. The executable links its
libpython and records that interpreter's executable path as CPython's program
name, so CPython discovers the same environment, including its virtualenv.
The selected environment must contain the full LMCache server installation.
The binary requires that interpreter and environment to remain at their build
locations; standalone distribution and relocatable packaging are deferred.
The initial implementation supports Unix hosts (Linux and macOS).

Rust passes OS argument bytes without shell interpolation. CPython initializes
on the main thread, executes the existing CLI, handles Python exit semantics,
and finalizes through `Py_BytesMain`. Rust returns its status. Python continues
to own signal handlers and server resource cleanup. Calling Python here is a
bootstrap boundary, not a proposed callback interface for the native storage
core.

## Validation

Cargo subprocess tests use an isolated Python module fixture to verify
same-process execution, argument fidelity, interpreter environment selection,
`SystemExit` and uncaught exception handling through the executable. These
tests need CPython development libraries but no LMCache or GPU installation.
Real server validation additionally requires the full LMCache environment:
check help, HTTP health, and graceful termination using the Rust executable.
