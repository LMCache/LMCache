# MP Server Autostart

## Summary

The vLLM multiprocess connector can optionally start a local LMCache MP server
from the vLLM worker process. This is disabled by default and only targets
single-node deployments where the connector endpoint resolves to localhost.

The feature exists to let `vllm serve` bring up the local MP server without a
separate orchestration step. It does not replace explicit server management for
multi-node deployments, remote servers, crash recovery, or automatic restart.

## Startup ownership

vLLM creates worker KV connectors before the scheduler KV connector. Starting
the MP server from the scheduler would make workers race the server startup and
potentially time out while creating their message queue clients.

To avoid that ordering problem, the worker adapter owns startup:

```text
worker rank 0
  -> parse autostart config
  -> start local lmcache MP HTTP server if ZMQ PING is not already healthy
  -> wait for ZMQ PING
  -> create MessageQueueClient

other local workers
  -> wait for ZMQ PING
  -> create MessageQueueClient

scheduler adapter
  -> connect only
```

The owner election uses the vLLM worker rank, not `kv_worker_id`. Under MLA,
multiple tensor-parallel ranks can share the same derived `kv_worker_id`, so
`kv_worker_id == 0` would allow more than one worker to attempt startup. The
actual vLLM rank is unique within the local scheduler group.

## Configuration

Autostart is controlled through `kv_connector_extra_config`:

| Key | Meaning |
|---|---|
| `lmcache.mp.autostart` | Enables worker-0 startup when true. |
| `lmcache.mp.autostart.wait_timeout` | Seconds to wait for ZMQ PING readiness. |
| `lmcache.mp.autostart.server_args` | Extra CLI args for the server process. |

The connector derives the MP endpoint from `lmcache.mp.host` /
`lmcache.mp.port`, or from the connector server URL when those keys are absent.
Only `localhost`, `127.0.0.1`, and `::1` are accepted. Endpoint CLI flags
`--host`, `--port`, and `--http-host` are rejected in `server_args` because the
autostarted server must bind the same local endpoint the connector will use.

Required server sizing options, such as `--l1-size-gb` and `--eviction-policy`,
remain explicit in `server_args`; the connector does not infer them from the
vLLM configuration.

## Health check and failure handling

Readiness is checked through the same ZMQ path the connector uses for normal MP
communication: a temporary `MessageQueueClient` sends `RequestType.PING` and
waits for the response. This avoids coupling startup readiness to the HTTP
frontend.

If worker 0 starts a process and it exits or fails to become healthy before the
timeout, the launcher terminates that owned process and raises `ConnectionError`.
If the server is already healthy, worker 0 does not start another process.

## Lifetime

The autostarted server is treated as a shared local service. Normal vLLM adapter
shutdown does not terminate it because another vLLM instance may still be using
the same MP server. Operators should stop the server process separately when it
is no longer needed.

## Tests

Unit coverage validates config parsing, local-host restrictions, endpoint flag
rejection, ZMQ PING health checks, failure cleanup, worker-0 ownership,
non-owner worker waiting, legacy connector connect-only behavior, and scheduler
connect-only behavior.

The Buildkite `mp_autostart_tp2` smoke test starts `vllm serve` with TP=2 and
verifies that the MP server is not pre-existing, is autostarted by vLLM, and
responds to an independent ZMQ PING.
