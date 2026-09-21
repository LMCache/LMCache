# MP Server Autostart

## Summary

The vLLM multiprocess connector can optionally start a local LMCache MP server
from the vLLM worker process. This is disabled by default and only targets
single-node, single-server deployments with a local connector endpoint.
Connector initialization rejects auto-start with more than one server URL,
before creating adapters or waiting for a server.

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
  -> create transport request client

other local workers
  -> wait for ZMQ PING
  -> create transport request client

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

The launcher uses the connector's resolved server URL. When
`lmcache.mp.server_urls` is set, `lmcache.mp.host` and `lmcache.mp.port` are
ignored, including during auto-start. Otherwise the connector builds the URL
from those host/port settings.
Only `localhost` and `127.0.0.1` are accepted. IPv6 endpoints (including `::1`)
raise `ValueError` before probing or starting a process: the MP ZMQ transport
does not enable IPv6 sockets. Disabling auto-start leaves connect-only behavior
unchanged. Endpoint CLI flags
`--host`, `--port`, and `--http-host` are rejected in `server_args` because the
autostarted server must bind the same local endpoint the connector will use.

Required server sizing options, such as `--l1-size-gb` and `--eviction-policy`,
remain explicit in `server_args`; the connector does not infer them from the
vLLM configuration.

## Health check and failure handling

Readiness is checked through the same ZMQ path the connector uses for normal MP
communication: `RequestClientFactory.create` creates a temporary client and
`client.ping(None)` checks server readiness without requiring an already
registered worker instance. The transport handles PING payload serialization;
the launcher does not duplicate its wire protocol. This avoids coupling startup
readiness to the HTTP frontend. The wait timeout must be positive and finite.

If worker 0 starts a process and it exits or fails to become healthy before the
timeout, the launcher terminates that owned process and raises `ConnectionError`.
If the server is already healthy, worker 0 does not start another process.

## Lifetime

The autostarted server is a child of vLLM worker 0, not an independently managed
service. LMCache's adapter shutdown does not call the launcher's `shutdown()`
method, but vLLM can terminate the child through its own process-tree cleanup.
Real TP=2 testing with vLLM 0.28.0 confirmed that its default SIGTERM shutdown
path recursively killed the MP server. This result does not establish the
behavior of other vLLM versions or exit paths: neither survival nor automatic
cleanup is guaranteed. Operators must stop any remaining auto-started server
when it is no longer needed.

Servers that must survive vLLM restarts or be shared across vLLM instances must
be started and managed separately, outside the vLLM process tree. Connect-only
mode supports this deployment, and auto-start still skips spawning when the
configured server is already healthy. Keeping persistent service management
external avoids adding daemonization, restart, or process-reaping policy to
this startup convenience feature.

## Tests

Unit coverage validates config parsing, local-host restrictions, endpoint flag
rejection, ZMQ PING health checks, failure cleanup, worker-0 ownership,
non-owner worker waiting, legacy connector connect-only behavior, and scheduler
connect-only behavior.

The Buildkite `mp_autostart_tp2` smoke test starts `vllm serve` with TP=2 and
verifies that the MP server is not pre-existing, is autostarted by vLLM, and
responds to an independent ZMQ PING.
