# Mooncake process transport (experimental)

Related design discussion: [#6165](https://github.com/LMCache/LMCache/issues/6165).

`mooncake_process_client.py` implements the native completion protocol in a
dedicated Linux child. It is **not registered by the L2 factory**. Existing
Mooncake configurations continue to use the existing in-process implementation.
This is a component for reviewing the failure boundary, not a deployment recipe.

## Why a process

A Python timeout cannot cancel a C++ SDK call. A failed/partial native GET or SET
also does not establish that every native access to the supplied buffer has
ended. Reporting a synthetic miss at that point can release an allocation too
early. The parent therefore kills and reaps the worker before publishing failed
data-operation completions. If exit cannot be confirmed within `kill_timeout`,
it terminates the parent with exit code 78, optionally writing a diagnostic file.

```text
Caller owns arena/allocations        Parent process          Child process
        |                           ProcessNativeClient      Mooncake SDK
        | submit(keys, views)              |                     |
        |--------------------------------->|                     |
        |                                  | keys + offsets      |
        |                                  |-------------------->|
        |                                  |                     | DMA
        |                                  | successful result   |
        | poll / drain result <------------|<--------------------|
        |                                  |                     |
        | retain allocation                | timeout/fault       |
        |                                  | SIGKILL + waitpid   |
        |                                  |-------------------->|
        | failed result <------------------| confirmed exit       |
        | release allocation               |                     |
```

The worker exclusively owns its SDK client and verbs resources. Reaping is used
under the Linux/provider contract that exit cleanup destroys its QPs/MRs before
the process is reaped; the worker must not fork or export those resources.
This is not a guarantee for arbitrary driver/HCA failures. Successful results
still depend on the SDK's successful-completion contract.

## Shared mapping and ownership

The constructor requires a writable shared file mapping starting at offset zero,
covering the entire backing file. It validates `/proc/self/maps` against the
file's device, inode, size and mapping offsets. A same-sized unrelated file or a
private copy-on-write mapping is rejected before starting the worker. The parent
retains the verified FD and passes it to each child, so pathname replacement
cannot redirect a restart to another arena. Both processes translate buffer
views to checked offsets; no KV payload crosses the command socket.

The caller owns allocation lifetime. GET destinations must remain exclusively
writable and SET sources immutable until the terminal result or successful
`quiesce()`. Holding a Python memoryview is not sufficient to prevent allocator
reuse. The mapping must not be truncated, replaced in the parent, or reused for
a new service generation while a worker could still access it. After a parent
crash, allocate a fresh backing object and reclaim the old generation only after
its worker/resources are known to be gone.

**Current L1 TTL locks do not satisfy this ownership contract.** They can expire
while the worker is alive. An operation deadline smaller than the configured TTL
does not solve reservation age, scheduler pauses, or driver teardown delays.
This is why factory activation is deliberately absent.

## Admission and shutdown

- Only the manager thread performs socket I/O; the submitting thread enqueues a
  bounded metadata frame. Its CPU cost still scales with key/frame size.
- `max_pending` bounds accepted operations plus undrained terminal results.
  Saturation raises `BufferError` **before admission**. The caller must drain
  results before retrying. Frames are at most 16 MiB and batches 16,384 keys.
- Before readiness, after worker failure, and during quiesce, admitted operations
  receive immediate failures without touching native buffers.
- Deadlines include time in the parent queue. Worker replacement uses new parent
  IDs, so late completions cannot alias an operation from another generation.
- `quiesce()` stops admission to the worker, kills/reaps it, and publishes all
  outstanding terminal failures while preserving the completion notifier.
  `close()` then releases the notifier and retained FD. Both are idempotent.
- The manager/watchdog needs parent CPU scheduling. No wall-clock response bound
  is claimed when the parent is suspended or the OS cannot schedule it.

## Activation gates

Before connecting this component to `StorageManager`:

1. Establish non-expiring ownership for every active native I/O, including error
   paths, request aborts, eviction and shutdown. Do not rely on TTL comparison.
2. Stop controller admission/loops while retaining their buffers; fence adapters;
   then release controller resources. Account for multiple adapters and wrappers.
3. Provide allocator-owned mapping metadata and fresh service generations.
   Reject layouts such as serialized temporary buffers outside the shared arena.
4. Validate real native SDK initialization, RDMA success/failure/partial completion,
   disconnected startup, restart and shutdown on supported kernel/providers.
5. Measure IPC admission cost and serving throughput/TTFT. No performance claim
   or model-serving evaluation is established by the component tests.

## Validation

`tests/v1/distributed/test_mooncake_process_client.py` drives the real private
socket, subprocess, mmap and notifier paths with an SDK-shaped test double.
It covers shared offsets, mapping mismatch/private mapping rejection, file
replacement across restart, failed/partial GET and SET, crashes, SIGSTOP, bounded
admission/results, startup timeout, shutdown and notifier lifetime. A narrowly
injected unreapable-process test checks that failure is not published when exit
cannot be confirmed. These tests do not validate RDMA terminality.

With the repository's usual CPU-native test environment:

```bash
uv run --no-sync python -m pytest --noconftest \
  tests/v1/distributed/test_mooncake_process_client.py -q
```

For a source-only environment, the existing `tests/v1/distributed/__init__.py`
skips the entire package without native Bitmap. A temporary symlink lets pytest
collect this same self-contained test file without changing that package gate:

```bash
scratch=$(mktemp -d)
ln -s "$PWD/tests/v1/distributed/test_mooncake_process_client.py" "$scratch/"
uv run --no-sync python -m pytest --noconftest \
  "$scratch/test_mooncake_process_client.py" -q
```

The test loads the component source directly, not the L2 factory, and reports no
factory, GPU, SDK, RDMA or end-to-end serving coverage.
