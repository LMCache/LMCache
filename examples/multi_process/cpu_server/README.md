# CPU-only cache server with a CUDA vLLM worker

This example runs a CPU-only LMCache server and a one-GPU vLLM worker on the
same Kubernetes node. The worker uses `engine_driven` to copy KV tensors between
its GPU and CPU chunks; inline messages carry those chunks to the server's
2 GiB in-memory cache.

```text
CUDA vLLM worker             CPU-only LMCache server
GPU KV <-> CPU chunks <----> CPU cache
              inline messages over node loopback:6555
```

The [original example](../) uses a GPU-visible server for the CUDA IPC path.
A cache stored in CPU RAM does not make that server CPU-only: CUDA IPC still
needs GPU access. This release uses pickle serialization for inline transfer.
Here, both sides explicitly select `engine_driven`, and the
server cannot see GPUs. Inline transfer needs no shared host `/dev/shm`; the
worker has its own 1 GiB memory-backed volume for vLLM multiprocessing.

## Prerequisites

- An x86-64 Linux Kubernetes GPU node with a compatible NVIDIA driver/container runtime
  and device plugin, at least one allocatable GPU, and enough host RAM for the
  cache server (4 GiB requested, 8 GiB limit) plus vLLM and the system.
- `kubectl` access that can create a namespace, label the selected node and
  deploy workloads using host networking. Adapt the GPU-node toleration if your
  cluster uses a different taint.
- Network access from the worker to download `Qwen/Qwen3-0.6B`.
- The manifests pin the official LMCache v0.5.6 image on both sides to keep
  their wire protocol compatible. The server overrides the image entrypoint
  and runs only LMCache, with GPUs hidden. When upgrading, update both image
  references together and repeat the checks below.

Use one otherwise unused, trusted GPU node. Ports **6555, 8000 and 9090 must be
free** on that node. Do not run the original example there at the same time.
Both Pods use the node's network namespace; a Kubernetes namespace does not
isolate their ports. Transport and inference bind to `127.0.0.1`; the server's
Prometheus endpoint on port 9090 uses the default all-interface bind, so restrict
access to that port with the node's firewall/network controls.

The server requests no GPU and hides GPU visibility through the container
runtime environment. Do not attach a GPU RuntimeClass or management CDI device
to it. Verify actual GPU invisibility below; a missing GPU resource request
alone is insufficient.

## Start the server, then the worker

Run these commands from the repository root. Set `GPU_NODE` to the chosen node's
name and label **only that node** for this example:

```bash
GPU_NODE=your-gpu-node
kubectl label node "$GPU_NODE" lmcache.ai/cpu-server-example=true
kubectl apply -f examples/multi_process/cpu_server/lmcache-daemonset.yaml
kubectl -n multi-process-cpu-server rollout status ds/lmcache-cpu-server --timeout=600s
```

Server readiness checks only that the transport socket accepts connections. It
does not prove registration, storage or retrieval. Verify CPU-only detection:

```bash
kubectl -n multi-process-cpu-server exec ds/lmcache-cpu-server -- \
  /opt/venv/bin/python3 -c '
import torch
from lmcache.v1.platform import torch_dev, torch_device_type
from lmcache.v1.platform.devices.cpu.stub_cpu_device import StubCPUDevice
print(torch.cuda.is_available(), torch.cuda.device_count(), torch_device_type, torch_dev)
if torch.cuda.is_available() or torch.cuda.device_count() != 0 or torch_device_type != "cpu" or not isinstance(torch_dev, StubCPUDevice):
    raise SystemExit("Cache server is not CPU-only")
'
kubectl apply -f examples/multi_process/cpu_server/vllm-deployment.yaml
kubectl -n multi-process-cpu-server rollout status deploy/vllm-cpu-server --timeout=900s
kubectl -n multi-process-cpu-server get pods -o wide
```

Both Pods must be on `GPU_NODE`. Worker startup/readiness checks call vLLM's
`/health` on loopback from inside the container. Requests below also run inside
the worker, so no Service, external endpoint or port forwarding is needed.

## Check cache reuse

Record server logs and the initial metrics:

```bash
kubectl -n multi-process-cpu-server logs ds/lmcache-cpu-server
kubectl -n multi-process-cpu-server exec deploy/vllm-cpu-server -- \
  python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:9090/metrics").read().decode())'
```

The server log should report `StubCPUDevice` and
`Using pickle non-GPU transfer strategy`; worker logs should show a CUDA
transfer context with `mode=engine_driven` and no GPU UUID registration error.
Run this request twice. The repeated text makes the prompt span multiple cache
chunks; the response's `usage.prompt_tokens` should exceed 1,024.

```bash
kubectl -n multi-process-cpu-server exec -i deploy/vllm-cpu-server -- python3 - <<'PY'
import json
import urllib.request

prompt = ("The cache holds computed attention keys and values so repeated inputs "
          "can reuse earlier work. ") * 96 + "Summarize this in one sentence."
payload = json.dumps({"model": "Qwen/Qwen3-0.6B", "prompt": prompt,
                      "temperature": 0, "max_tokens": 32}).encode()
request = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
                                 data=payload,
                                 headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=120) as response:
    print(response.read().decode())
PY
```

Read metrics after each request using the earlier command. Poll for up to 30
seconds for asynchronous writes to finish before repeating: the cold request must increase
`lmcache_mp_l1_write_chunks_total`; the repeated request must increase
`lmcache_mp_l1_read_chunks_total`. Compare counter deltas rather than lifetime
totals. vLLM's internal prefix caching is disabled so it cannot explain reuse.
A successful HTTP response alone does not prove a cache hit.

For a restart check, record the server Pod's UID and restart count, then restart
only the worker and repeat the same request:

```bash
kubectl -n multi-process-cpu-server get pods -o wide
kubectl -n multi-process-cpu-server get pods -l app=lmcache-cpu-server \
  -o jsonpath='{range .items[*]}{.metadata.uid}{" "}{.status.containerStatuses[0].restartCount}{"\n"}{end}'
kubectl -n multi-process-cpu-server rollout restart deploy/vllm-cpu-server
kubectl -n multi-process-cpu-server rollout status deploy/vllm-cpu-server --timeout=900s
```

The worker UID should change; the server UID/restart count must stay unchanged.
The next identical request must still increase cache reads. Also send a distinct
prompt with no full cache chunk in common and check that reads do not increase.
Compare generated text across cold, warm and restarted requests and investigate
any differences. Save full responses, logs, metrics, Pod specifications and image
digests before cleanup. This checks functionality and cache survival across a
**worker** restart; it does not show cache-server persistence or a speedup.

## Cleanup

```bash
kubectl delete namespace multi-process-cpu-server --wait=true
kubectl label node "$GPU_NODE" lmcache.ai/cpu-server-example-
```

If you rented the node, delete the rental and any associated billable resources
separately; deleting the Kubernetes namespace does not stop those charges.
