# Mooncake admission and recovery guard (experimental)

Related discussion: [#6165](https://github.com/LMCache/LMCache/issues/6165).
This component composes with the [process transport](mooncake_process_client.md).
Neither component is registered by the L2 factory; existing deployments are
unchanged. The process transport's L1 ownership, controller shutdown and RDMA
validation gates also apply here. Do not wrap the unmodified in-process SDK and
treat the guard as a DMA cancellation mechanism.

## Responsibility boundary

```text
caller owns allocations until terminal result
    |
    v
GuardedNativeClient       SubprocessDataProbe
 admission + results ---> separate SDK client + private payload
    |                    PUT / GET / compare bytes / remove
    v
TerminalNativeClient (for example ProcessNativeClient)
 native lifetime + terminal completions
    |
    v
remote Mooncake
```

`TerminalNativeClient` submissions return unique nonnegative IDs without waiting
for remote work. `BufferError` means rejection before admission. Other exceptions
do not establish buffer safety; callers must fence before releasing allocations.
Every GET/SET completion, including failures, means native buffer access has
ended. `quiesce()` fences all operations and makes their results drainable without
closing the notifier. The guard cannot enforce these guarantees for an arbitrary
client implementation.

## Admission and recovery

```text
OPEN (bypass) -- cooldown --> HALF_OPEN (probe; still bypass)
  ^                              |
  | failure / slow GET/SET        | enough timely verified probes,
  |                              | no old operations, native ready
  +-------------------------- CLOSED (remote admission)
                                  |
                                  +-- periodic independent data probe
```

The guard starts OPEN and initializes at most one native client in a background
thread. An unavailable client therefore does not block the caller. An arbitrary
Python initializer cannot be canceled safely: the supported factory must return
promptly, as the process transport constructor does. A stuck custom factory can
still block shutdown; no universal initializer deadline is claimed.

An expired EXISTS returns a failed lookup immediately because it carries no L1
buffers. The native lookup remains tracked until its late result arrives, which
is discarded exactly once. A slow GET/SET opens admission but **does not complete
that operation**. Its buffers remain owned until the transport's terminal result.
The separate GET limit prevents one unhealthy remote path from consuming all
allowed data operations.

`max_pending` bounds both outstanding native IDs (including expired lookups) and
pending public results plus undrained completions. A caller that does not drain
results receives `BufferError` before any new admission. Bypasses use negative
IDs, so they cannot alias native completions.

Recovery requires consecutive timely data probes, native readiness and no old
operations. A failure generation invalidates probes started before newer native
faults. This prevents a stale successful probe from closing the circuit. Periodic
healthy probes detect backends whose EXISTS path reports errors as misses.

## Independent probe

`SubprocessDataProbe` starts a separate interpreter with the public Mooncake
Python SDK, no caller buffers and a 1 MiB local transfer buffer. It uses a random
probe key and payload, verifies PUT/GET bytes and removes the object. It validates
the configured data path, not just master TCP reachability. It is a small health
check, not a bandwidth test or proof that another client's registered regions
are healthy. Native readiness and real operations remain separate signals.

One child may run at a time. The deadline includes startup. On timeout, the probe
kills and reaps that child; an unreapable child retains its handle and blocks
replacement. `close()` can retry reaping and reports failure instead of hiding
it. A timeout/crash after PUT can leave the small probe object remotely until a
later probe removes it or the store evicts it.

The SDK setup fields currently used are `local_hostname`, `metadata_server`,
`master_server_addr`, and optional `protocol`, `rdma_devices`, `tenant_id`.
Deployments needing other connection/authentication options require an explicit
probe implementation; these component tests do not certify SDK compatibility.

## Shutdown and observability

Quiesce switches to bypass, stops the monitor, joins initialization, fences the
native client, relays every pending terminal result, and closes the probe. A
native client that returns from quiesce with unresolved operations is rejected;
the guard does not invent DMA completions. The notifier remains usable until
`close()`. Submissions racing quiesce receive ordinary safe bypass results within
the result budget, rather than raising because the monitor is stopping.

`report_status()` reports circuit state/reason, remote admission, pending/expired
operations, slow data operations, failure/bypass counts, probe state and policy.
“Remote available” describes admission at that instant, not a service SLA.
The first affected request may still wait for a lookup or transport deadline;
recomputation and reduced cache hit rate can increase latency.

## Validation

The guard suite tests startup bypass, lookup timeout and late-result disposal,
non-synthetic GET/SET completion, admission/result bounds, stale probe rejection,
HALF_OPEN recovery, data integrity probes, probe timeouts and unreapable children,
quiesce with pending work, and process-transport composition. The composed cases
observe notification only after the worker PID disappears and then verify that
the shared arena remains readable/writable. Fake SDK data is used; these are not
GPU, RDMA, model-serving or throughput tests.

```bash
uv run --no-sync python -m pytest --noconftest \
  tests/v1/distributed/test_mooncake_process_client.py \
  tests/v1/distributed/test_mooncake_resilience.py -q
```

In a source-only environment, use the temporary-symlink collection approach
described in the process transport document for both test files. This bypasses
the existing distributed-test native-extension gate, not the code under test.
