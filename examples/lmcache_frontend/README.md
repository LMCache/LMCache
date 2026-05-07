# LMCache Frontend Example (MP mode)

This example shows how to bring up the **LMCache Frontend Dashboard**
as a runtime plugin of the LMCache multiprocess (MP) server, together
with a trivial discovery service.

Architecture:

```
+-----------------------------+       heartbeat (HTTP GET)
|  LMCache MP HTTP Server     | ------------------------------> +---------------------------+
|  (http_server.py)           |                                 |  simple_discover_service  |
|                             | <----------------------- poll --|  (this example)           |
|  MPRuntimePluginLauncher    |                                 +---------------------------+
|    |                        |                                         ^
|    +-> lmcache_mp_frontend  |                                         |
|        _plugin.py (subproc) |                                         |
|        -> app.main()        |                                         |
|           - heartbeat       |--- heartbeat ---------------------------+
|           - (optional) UI   |
+-----------------------------+
```

The real discovery service in production is expected to be provided
by each company. `simple_discover_service.py` is only a flask-based
example so you can try the flow end-to-end.

## Files

| File | Description |
|------|-------------|
| `simple_discover_service.py` | A tiny Flask app exposing `/lmcache_heartbeat` and `/lmcache_infos` endpoints. |
| `run_mp_server_with_frontend.sh` | Launches the LMCache MP HTTP server with the frontend plugin wired in. |

## Quick Start

1. Install the Python deps used by the frontend plugin and discovery
   service (not pulled in by default to keep the base install slim):

   ```bash
   pip install flask httpx fastapi uvicorn
   ```

2. Start the example discovery service (port 5000):

   ```bash
   python examples/lmcache_frontend/simple_discover_service.py
   ```

3. Start the LMCache MP HTTP server with the frontend plugin:

   ```bash
   bash examples/lmcache_frontend/run_mp_server_with_frontend.sh
   ```

   Under the hood this passes:

   ```
   --runtime-plugin-locations \
       lmcache/lmcache_frontend/lmcache_mp_plugin/lmcache_mp_frontend_plugin.py
   --runtime-plugin-config \
       '{"plugin.frontend.heartbeat-url": "http://localhost:5000/lmcache_heartbeat"}'
   ```

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
