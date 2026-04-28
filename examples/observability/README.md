# LMCache Observability Example

Minimal example showing per-request OTel tracing and metrics for LMCache + vLLM,
visualized in Grafana.

## Stack

```
LMCache / vLLM
  └─ OTLP gRPC → OTel Collector (:4320)
                   ├─ traces  → Tempo (:3200)
                   └─ metrics → Prometheus (:9091)
                                  └─ Grafana (:3000)
```

## Step 1 — Start the observability stack

```bash
cd examples/observability
docker compose up -d
```

## Step 2 — Start LMCache + vLLM

```bash
MODEL=/your/model/path bash start-server.sh
```
 

## Step 3 — Send requests to populate traces

```bash
# Run a short long-doc-qa benchmark: first query is a miss, subsequent
# queries against the same document are cache hits.
lmcache bench engine \
  --engine-url http://localhost:8100 \
  --workload long-doc-qa \
  --kv-cache-volume 1 \
  --ldqa-query-per-document 10
```

## Step 4 — Visualize in Grafana

Open **http://localhost:3000** → **Explore** → datasource **Tempo**.

```
# All request root spans
{ name = "request" }

# Filter to a specific session
{ name = "request" && span.session_id = "<request_id>" }

# Only cache-hit requests (had a retrieve)
{ name = "request" } >> { name = "mp.retrieve" }
```

Click any trace to open the waterfall:

```
request  [══════════════════════════════════════]
  mp.lookup_prefetch  [════]
  mp.retrieve               [════════]
  mp.store                            [══════]
```

The pre-provisioned **LMCache** dashboard under **Dashboards** shows cache hit
rate, StorageManager read/write rates, and the live trace panel.

## Files

```
docker-compose.yml          — 4-service stack (collector, tempo, prometheus, grafana)
otel-collector.yml          — OTLP receiver → Tempo + Prometheus fan-out
tempo.yml                   — local trace storage
prometheus.yml              — scrapes lmcache metrics from collector
grafana/provisioning/       — auto-provisioned datasources + dashboard
start-server.sh             — launches LMCache server + vLLM with OTLP enabled
```
