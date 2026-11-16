# Register only ready vLLM workers

`LMCacheMPConnector.register_kv_caches()` retains the latest worker tensor
references without exporting IPC or creating server GPU resources. After warmup,
Graph capture, final KV sizing and device synchronization, vLLM invokes
`on_kv_cache_ready()`. That callback builds the final layout metadata, calls the
existing adapter registration, waits for its ACK, initializes the existing
feature dispatcher, and starts the worker heartbeat before returning.

```
local tensors -> warmup/final sizing -> ready callback
                                      -> IPC export / server import -> ACK
                                      -> heartbeat -> startup returns -> requests
```

Thus REGISTER means the engine has reached the startup-ready boundary. A worker
that dies before that boundary leaves no GPU KV registration. The Server's L1
pool still initializes independently; chunk-size negotiation may occur earlier.
No registration or first-request activation thread is added. Registration adds
to startup time, not to an already-started engine's first request.

The connector rejects vLLM without the ready hook (vllm-project/vllm#66574);
there is no eager fallback. Repeated successful ready notifications are no-ops;
a second tensor callback before readiness replaces the retained dictionary.
The hook computes edited views and group metadata from the final configuration.
Existing sleep/wake unregister/restore and Server-restart recovery stay in the
adapter. The heartbeat begins immediately after successful startup registration,
so an idle ready worker does not wait for traffic to prove liveness.

Registration failures propagate to fail startup. RPC waits use `mq_timeout`,
which does not cancel a stuck CUDA call or make Server-side work disappear. A
crash during REGISTER/ACK can still leave resources until the existing reaper
runs. This change does not redesign in-flight transfer draining, lease tokens,
or reaper safety. Experimental dispatcher registration follows the same boundary.

Tests exercise no registration before ready, latest tensors, heartbeat without
traffic, idempotent ready, warmup shutdown, explicit old-vLLM rejection, timeout
propagation, and existing transfer/MultiConnector/sleep-wake behavior.
