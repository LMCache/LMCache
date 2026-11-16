# Register only ready vLLM workers

`register_kv_caches()` retains the latest tensors locally. After warmup and
final KV sizing, vLLM's `on_kv_cache_ready()` calls the existing adapter's
registration, waits for ACK, and starts heartbeat before returning.

```
local tensors -> warmup/final sizing -> ready -> REGISTER/ACK -> heartbeat -> requests
```

A worker that fails before ready leaves no Server GPU KV registration. A ready
but request-idle worker sends heartbeat immediately. Registration cost stays in
startup; no first-request activation is added. Server L1 initialization is unchanged.

Requires vllm-project/vllm#66574 and the LMCache-owned connector module path.
Old vLLM is rejected. Duplicate ready notifications are no-ops. Registration
errors fail startup; RPC timeout does not cancel a stuck device operation.
Sleep/wake and outage recovery reuse the existing adapter paths.

This addresses the idle-worker trigger in #4715 and startup concerns from #4732,
but does not implement #4732's missing-registration probe. Ordinary PING still
returns healthy when the Server is alive but a Worker registration was reaped.
This PR does not redesign recovery, reaper fencing or in-flight transfer cleanup.
