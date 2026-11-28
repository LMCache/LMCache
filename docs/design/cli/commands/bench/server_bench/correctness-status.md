# Server-bench correctness status

`lmcache bench server` runs `BaselineBenchCase` instances and merges their
`BenchResult` values before emitting metrics. Recorded checks cover cold/warm
lookups, transfers, checksum availability, local comparisons and optional
server comparisons. A completed case with a failed check must not produce a
successful process status.

## Reporting contract

`run_server_bench` closes every started client, emits the existing metrics
summary (including any `--output` file), and then examines the recorded checks.
If any check failed, it names the failed checks and their failed/total counts
on stderr and raises `SystemExit(1)`. Passing work retains exit status 0. A
clean interruption also retains status 0 if completed work recorded no failures;
`BenchResult.succeeded` alone cannot implement this policy because it also
rejects interrupted runs. An empty run with no checks remains a successful
no-op.

The CLI grammar and metrics schema do not change. `Checksum FAIL` still counts
completed comparisons that mismatched, so unavailable checksums do not inflate
that count. Automation must use process status to determine overall success.

```text
main -> ServerBenchCommand.execute -> run_server_bench
  -> BaselineBenchCase.run / run_concurrently -> merged BenchResult
  -> close clients -> emit metrics -> inspect recorded failures
     -> any failure: stderr diagnostic + exit 1
     -> no failure: success (including clean interruption)
```

## Asynchronous STORE visibility

A successful STORE in `lmcache_driven` mode can return before `finish_write`
makes its objects lookup-visible. An immediate warm miss is a failed benchmark
check, but does not by itself establish eviction or data loss. This reporting
change adds no retries, changes no STORE completion semantics, and does not
exclude failed warm checks from reported measurements. A commit-visibility
barrier is a separate data-path contract change.
