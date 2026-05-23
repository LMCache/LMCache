# LMCache Frontend Example (MP mode)

This example shows how to bring up the **LMCache Frontend Dashboard**
as a runtime plugin of the LMCache multiprocess (MP) server, together
with a trivial discovery service.

Architecture:

```
+-----------------------------+
|  LMCache MP HTTP Server     |
|  (http_server.py)           |
|                             |
|  MPRuntimePluginLauncher    |                +---------------------------+
|    |                        |                |  simple_discover_service  |
|    +-> lmcache_mp_frontend  |   heartbeat    |  (lmcache.tools)          |
|        _plugin.py (subproc) | -------------> |                           |
|        -> app.main()        |   (HTTP GET)   |  /lmcache_heartbeat       |
|           - HeartbeatService|                |  /lmcache_infos           |
|           - (--no-http)     |                +---------------------------+
+-----------------------------+
```

Only the frontend plugin talks to the discovery service; the MP HTTP
server itself is never contacted by it.

The real discovery service in production is expected to be provided
by each company. `lmcache.tools.simple_discover_service` is shipped
with the package as a flask-based reference implementation so you can
try the flow end-to-end right after `pip install lmcache`.

## Files

| File | Description |
|------|-------------|
| `lmcache/tools/simple_discover_service.py` | A tiny Flask app exposing `/lmcache_heartbeat` and `/lmcache_infos` endpoints. Runnable via `python3 -m lmcache.tools.simple_discover_service`. |
| `run_mp_server_with_frontend.sh` | Launches the LMCache MP HTTP server with the frontend plugin wired in. |

## Quick Start

1. Install the Python deps used by the frontend plugin and discovery
   service (not pulled in by default to keep the base install slim):

   ```bash
   pip install flask httpx fastapi uvicorn
   ```

2. Start the example discovery service (listens on ``0.0.0.0:5000``):

   ```bash
   python3 -m lmcache.tools.simple_discover_service
   ```

   The service also exposes ``/heartbeat`` as an alias of
   ``/lmcache_heartbeat`` for compatibility with older clients.

3. Start the LMCache MP HTTP server with the frontend plugin:

   ```bash
   bash examples/lmcache_frontend/run_mp_server_with_frontend.sh
   ```

   Under the hood this runs (see the script for the full command):

   ```bash
   python3 -m lmcache.v1.multiprocess.http_server \
       --host localhost --port 5555 \
       --http-host 0.0.0.0 --http-port 8085 \
       --l1-size-gb 2 \
       --eviction-policy LRU \
       --runtime-plugin-locations \
           lmcache/lmcache_frontend/lmcache_mp_plugin/lmcache_mp_frontend_plugin.py \
       --runtime-plugin-config \
           '{"plugin.frontend.heartbeat-url": "http://localhost:5000/lmcache_heartbeat"}'
   ```

   Note: the config key is normalised internally, so both
   ``plugin.frontend.heartbeat-url`` and
   ``plugin.frontend.heartbeat_url`` are accepted.

4. Inspect the registered nodes:

   ```bash
   curl http://localhost:5000/lmcache_infos
   ```

## How it works

* `MPRuntimePluginLauncher` (in `lmcache/v1/multiprocess/`) launches
  `lmcache_mp_frontend_plugin.py` as a subprocess and injects the
  aggregated config via the `LMCACHE_RUNTIME_PLUGIN_CONFIG` env var.
* The plugin script builds argv for `lmcache.lmcache_frontend.app.main()`
  and runs it with `--no-http`, so only the heartbeat loop runs.
* `HeartbeatService` periodically `GET`s the configured `heartbeat_url`
  with `api_address`, `pid`, `version`, etc. The discovery service
  records the latest report and exposes them via `/lmcache_infos`.

## Using a different discovery service

Replace the `plugin.frontend.heartbeat-url` value with your own
endpoint. Any HTTP service that accepts
`GET <url>?api_address=...&pid=...&version=...&other_info=...`
will work.
