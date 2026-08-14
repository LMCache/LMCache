Deployment Guide
================

This page covers deploying LMCache multiprocess mode in Docker and Kubernetes
environments, along with production best practices.

.. contents::
   :local:
   :depth: 2

Docker
------

**LMCache container:**

.. code-block:: bash

    docker run --runtime nvidia --gpus all \
        --network host \
        --ipc host \
        lmcache/standalone:nightly \
        /opt/venv/bin/lmcache server \
        --l1-size-gb 60 --eviction-policy LRU --max-workers 4 --port 6555

**vLLM container:**

.. code-block:: bash

    docker run --runtime nvidia --gpus all \
        --network host \
        --ipc host \
        lmcache/vllm-openai:latest-nightly \
        Qwen/Qwen3-14B \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.port": 6555}}'

Required Docker flags:

- ``--network host`` -- Allows the vLLM container to reach LMCache on localhost.
- ``--ipc host`` -- Required for CUDA IPC shared memory transfers between
  containers.
- ``--runtime nvidia --gpus all`` -- GPU access via the NVIDIA container
  runtime.

**HTTP server variant:**

For health-check and cache management API support (useful with container
orchestrators), use the HTTP server entry point:

.. code-block:: bash

    docker run --runtime nvidia --gpus all \
        --network host \
        --ipc host \
        lmcache/standalone:nightly \
        /opt/venv/bin/lmcache server \
        --l1-size-gb 60 --eviction-policy LRU --max-workers 4 --port 6555

Kubernetes
----------

LMCache is designed for a **DaemonSet + Deployment** pattern: one LMCache
server per node (DaemonSet) shared by multiple vLLM pods (Deployment).

Example YAML files are provided in ``examples/multi_process/``.

Prerequisites
~~~~~~~~~~~~~

- Kubernetes cluster with GPU support (NVIDIA GPU Operator installed)
- At least 4 GPUs per node
- ``kubectl`` configured to access your cluster

Step-by-Step
~~~~~~~~~~~~

**Step 1: Create namespace**

.. code-block:: bash

    kubectl create namespace multi-process

**Step 2: Deploy LMCache DaemonSet**

.. code-block:: bash

    kubectl apply -f examples/multi_process/lmcache-daemonset.yaml

**Step 3: Deploy vLLM**

.. code-block:: bash

    kubectl apply -f examples/multi_process/vllm-deployment.yaml

.. note::
   The default model is ``Qwen/Qwen3-14B``.  For gated models (e.g., Llama),
   create a Secret with your Hugging Face token:

   .. code-block:: bash

       kubectl create secret generic vllm-secrets \
         --from-literal=hf_token=your_hf_token_here \
         -n multi-process

   Then add the ``HF_TOKEN`` environment variable to the vLLM container spec.

**Step 4: Monitor deployment**

.. code-block:: bash

    # DaemonSet status
    kubectl get daemonset -n multi-process
    kubectl get pods -n multi-process -l app=lmcache-server

    # vLLM status
    kubectl get pods -n multi-process -l app=vllm-deployment -w

    # LMCache logs (for a specific node)
    VLLM_NODE=$(kubectl get pod -n multi-process -l app=vllm-deployment \
        -o jsonpath='{.items[0].spec.nodeName}')
    LMCACHE_POD=$(kubectl get pod -n multi-process -l app=lmcache-server \
        --field-selector spec.nodeName=$VLLM_NODE \
        -o jsonpath='{.items[0].metadata.name}')
    kubectl logs -n multi-process $LMCACHE_POD -f

**Step 5: Send test requests**

.. code-block:: bash

    kubectl port-forward -n multi-process deployment/vllm-deployment 8000:8000

    curl -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"Qwen/Qwen3-14B\",
            \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
            \"max_tokens\": 10
        }"

Architecture Notes
~~~~~~~~~~~~~~~~~~

- **DaemonSet uses ``hostNetwork: true``** so vLLM pods discover the LMCache
  server via ``status.hostIP``.
- **Both containers mount ``/dev/shm``** from the host to enable CUDA IPC
  memory sharing.
- **GPUs are NOT requested in the DaemonSet** -- this allows GPUs to remain
  exclusively allocated to vLLM pods.  The NVIDIA container runtime
  automatically provides GPU access for IPC-based memory transfers.
- **Multiple vLLM pods** on the same node automatically connect to the same
  LMCache DaemonSet instance.

.. note::
   LMCache pods on nodes without GPUs will crash with CUDA initialization
   errors.  This is expected -- LMCache only needs to run on GPU nodes where
   vLLM pods are scheduled.

Health Checking (HTTP Server)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Kubernetes liveness/readiness probes, deploy the HTTP server variant
instead.  Use the ``/healthcheck`` endpoint:

.. code-block:: yaml

    livenessProbe:
      httpGet:
        path: /healthcheck
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 30
    readinessProbe:
      httpGet:
        path: /healthcheck
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 10

Monitoring Integration
~~~~~~~~~~~~~~~~~~~~~~

Prometheus metrics are enabled by default on port 9090.  Add a
``ServiceMonitor`` or Prometheus scrape annotation to collect metrics from the
LMCache DaemonSet pods.  See :doc:`observability/index` for metric details.

Cleanup
~~~~~~~

.. code-block:: bash

    kubectl delete -f examples/multi_process/vllm-deployment.yaml
    kubectl delete -f examples/multi_process/lmcache-daemonset.yaml
    kubectl delete namespace multi-process

Production Best Practices
-------------------------

**Worker count (``--max-workers``, ``--max-gpu-workers``, ``--max-cpu-workers``):**
``--max-workers`` sets both the GPU affinity pool and CPU normal pool sizes
(default 1).  Use ``--max-gpu-workers`` to override the GPU pool independently
--- set it to at least the number of vLLM instances sharing the cache server so
each instance gets its own dedicated thread.  Use ``--max-cpu-workers`` to
override the CPU pool for lookup and other non-GPU operations.

**L1 memory sizing (``--l1-size-gb``):**
Allocate as much CPU memory as available after accounting for the OS and vLLM.
A larger L1 cache means fewer L2 round-trips.

**Eviction tuning:**

- ``--eviction-trigger-watermark 0.8`` (default) triggers eviction when L1 is
  80% full.
- ``--eviction-ratio 0.2`` (default) frees 20% of allocated memory per
  eviction cycle.
- Lower the watermark or increase the ratio if you observe frequent evictions
  under steady load.

**Logging:**
Use ``LMCACHE_LOG_LEVEL=DEBUG`` during initial setup to verify L2 store/load
activity.  Switch to ``INFO`` (default) for production to reduce log volume.

Transfer Mode (``--supported-transfer-mode``, ``--shm-name``)
-------------------------------------------------------------

LMCache supports two worker → server transfer paths: an
**lmcache-driven** path (server pulls/pushes via CUDA IPC or CPU SHM,
used for STORE/RETRIEVE) and an **engine-driven** path
(PREPARE/COMMIT, used by CPU-only or non-CUDA accelerator workers).
The server picks which paths to load via ``--supported-transfer-mode``:

- ``auto`` *(default)* -- load both paths.  Workers of either device
  type can connect without manual configuration; the server has no
  upfront knowledge of the connecting worker's device.
- ``lmcache_driven`` -- load only the server-driven transfer path.
  Supports CUDA devices (IPC) and CPU devices (SHM).  Use to skip
  allocating the engine-driven prepare/commit resources (pickle codec).
- ``engine_driven`` -- load only the engine-driven path.  Use when
  serving CPU-only or non-CUDA accelerator workers.

When the engine-driven path is loaded (``auto`` or ``engine_driven``),
LMCache by default creates a shared-memory (SHM) pool for KV transfers
between the server and vLLM workers.  The ``--shm-name`` option lets
you control this behavior:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Effect
   * - *(not set)* (default)
     - Auto-allocate a SHM pool (current default behavior).
   * - ``""`` (empty string)
     - Disable the SHM pool entirely and fall back to the pickle-based
       transfer path.  Useful when ``/dev/shm`` is unavailable or when
       running without ``--ipc host`` in Docker.
   * - ``"my_pool"`` (any non-empty name)
     - Use that exact name for the SHM segment instead of the
       auto-generated one.  Handy when you need a deterministic,
       human-readable segment name for monitoring or debugging.

**Examples:**

.. code-block:: bash

    # Force pickle (no SHM):
    lmcache server --l1-size-gb 60 --eviction-policy LRU --shm-name ""

    # Named SHM segment:
    lmcache server --l1-size-gb 60 --eviction-policy LRU --shm-name "lmcache_pool"

Dynamo
------

Kubernetes recipe for running `NVIDIA Dynamo <https://github.com/ai-dynamo/dynamo>`_
aggregated vLLM serving with KV cache offloaded to a per-node LMCache MP
DaemonSet, sharing tensors with the worker via cross-Pod CUDA IPC.

.. rubric:: Step 1 — Install the Dynamo platform

.. code-block:: bash

    helm install dynamo-platform oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform \
      --version my-version \
      --namespace dynamo-system --create-namespace \
      --wait

.. dropdown:: How it works today

   NGC hasn't published a helm chart with the
   PR #8414 CRD fix yet, so the clean ``helm install`` above doesn't work
   as-is.  Two pieces are needed: the **chart** (carries the updated CRD) and
   the **operator image** (whose baked-in CRDs an init container re-applies on
   every restart).  Until NGC ships both, the workaround is:

   1. **Chart from local clone**: ``git clone https://github.com/ai-dynamo/dynamo.git``
      and check out commit
      `e7eb1c565f <https://github.com/ai-dynamo/dynamo/commit/e7eb1c565f>`_
      (the version validated for this recipe), then install from the local
      path with the NGC ``1.1.1`` operator image:

      .. code-block:: bash

          # Required for `helm dependency build` -- the platform chart pulls in
          # NATS and Bitnami subcharts and helm needs to know their repos.
          helm repo add nats https://nats-io.github.io/k8s/helm/charts/
          helm repo add bitnami https://charts.bitnami.com/bitnami
          helm repo update

          cd dynamo/deploy/helm/charts
          helm dependency build ./platform/
          helm install dynamo-platform ./platform/ \
            --namespace dynamo-system --create-namespace \
            --set "dynamo-operator.controllerManager.manager.image.tag=1.1.1" \
            --wait

   2. **Patch out the operator's CRD-applying init container** (it would
      otherwise overwrite the chart's new CRDs on every restart):

      .. code-block:: bash

          kubectl -n dynamo-system patch deployment dynamo-platform-dynamo-operator-controller-manager \
            --type=json -p='[{"op":"replace","path":"/spec/template/spec/initContainers","value":[]}]'

   3. **Replace the CRDs by hand** (helm doesn't update CRDs on
      install/upgrade):

      .. code-block:: bash

          cd ../../..
          kubectl replace -f platform/components/operator/crds/nvidia.com_dynamographdeployments.yaml
          kubectl replace -f platform/components/operator/crds/nvidia.com_dynamocomponentdeployments.yaml

   Alternative: build the operator image yourself from a commit that includes
   PR #8414 (so its init container re-applies the new CRDs) and point the
   chart at it via ``--set ...image.tag=<your-tag>``.  Avoids steps 2 and 3,
   but adds a registry of your own.

Verify the install:

.. code-block:: bash

    kubectl get crd | grep nvidia.com
    # expect:
    #   dynamocomponentdeployments.nvidia.com
    #   dynamographdeployments.nvidia.com

.. rubric:: Step 2 -- Install the LMCache operator

.. code-block:: bash

    kubectl apply -f https://github.com/LMCache/LMCache/releases/download/operator-latest/install.yaml

.. note::
   Tested with the LMCache operator image ``lmcache/lmcache-operator:v0.1.1``.

Verify the install:

.. code-block:: bash

    kubectl get crd | grep lmcache
    # expect:
    #   lmcacheengines.lmcache.lmcache.ai

.. rubric:: Step 3 -- Create the namespace

.. code-block:: bash

    kubectl create namespace dynamo-lmcache

.. rubric:: Step 4 -- Create the HF token Secret

Both ``Frontend`` and ``VllmDecodeWorker`` reference ``hf-token-secret`` via
``envFromSecret``.  The Secret must exist or the pods fail to start with
``secret "hf-token-secret" not found``.

.. code-block:: bash

    # Replace the dummy token below with a real HF token.
    kubectl apply -f - <<'EOF'
    apiVersion: v1
    kind: Secret
    metadata:
      name: hf-token-secret
      namespace: dynamo-lmcache
    type: Opaque
    stringData:
      HF_TOKEN: "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    EOF

.. rubric:: Step 5 -- Deploy the LMCacheEngine

Replace ``my-tag`` below with the ``lmcache/vllm-openai`` image tag you want
to run.

.. code-block:: bash

    kubectl apply -f - <<'EOF'
    apiVersion: lmcache.lmcache.ai/v1alpha1
    kind: LMCacheEngine
    metadata:
      name: lmcache-mp
      namespace: dynamo-lmcache
    spec:
      image:
        repository: lmcache/vllm-openai
        tag: my-tag
        pullPolicy: IfNotPresent
      # L1 (CPU RAM) cache size -- bump for production workloads.
      l1:
        sizeGB: 16
    EOF

.. note::
   Validated ``my-tag`` against ``nightly-2026-04-25`` (lmcache
   ``0.4.5.dev31``).  This pre-stable build is wire-compatible with the
   ``vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3`` worker image (which ships
   lmcache ``0.4.4``).

Verify with:

.. code-block:: bash

    kubectl -n dynamo-lmcache get lmcacheengine lmcache-mp
    # expect: STATUS  Running

.. rubric:: Step 6 -- Deploy the Dynamo worker

Edit ``examples/backends/vllm/deploy/agg_lmcache.yaml`` in the Dynamo repo:
replace ``nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag`` (on both ``Frontend``
and ``VllmDecodeWorker``) with your Dynamo vllm-runtime image.

.. code-block:: bash

    kubectl apply -n dynamo-lmcache -f examples/backends/vllm/deploy/agg_lmcache.yaml

.. note::
   Validated ``my-tag`` against
   ``1.2.0-deepseek-v4-cuda13-dev.3``
   (vLLM ``0.20.1``, lmcache ``0.4.4``).  Pair with the LMCacheEngine pin
   from Step 5.

Verify with:

.. code-block:: bash

    kubectl -n dynamo-lmcache get pods -l nvidia.com/dynamo-component-type=worker
    # expect: READY 1/1, STATUS Running

.. rubric:: Step 7 -- Verify

Send the same long prompt twice:

.. code-block:: bash

    kubectl -n dynamo-lmcache port-forward svc/vllm-agg-lmcache-frontend 8000:8000 >/dev/null &
    PF_PID=$!
    sleep 4

    PROMPT=$(python3 -c "print('the quick brown fox jumps over the lazy dog '*60)")
    REQ="{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":5}"

    for label in cold warm; do
      echo "--- $label ---"
      curl -s http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" -d "$REQ" \
        | python3 -m json.tool | grep -E "prompt_tokens|cached_tokens"
    done

    kill $PF_PID

Then check LMCache server metrics:

.. code-block:: bash

    LMC=$(kubectl -n dynamo-lmcache get pod -l app.kubernetes.io/instance=lmcache-mp -o name | head -1)
    kubectl -n dynamo-lmcache exec "$LMC" -- curl -s localhost:9090/metrics | grep '^lmcache_mp_'

Expected: warm response shows ``cached_tokens > 0``, and
``lmcache_mp_lookup_hit_tokens_total > 0``.

.. dropdown:: Metrics endpoint by lmcache version

    The LMCache operator declares three container ports on every DaemonSet pod:
    ``5555`` (ZMQ), ``8080`` (control HTTP), ``9090`` (Prometheus).  It also
    creates two Services: ``lmcache-mp`` (5555 + 8080) and a headless
    ``lmcache-mp-metrics`` (9090).  Which port actually serves ``/metrics``
    depends on the image:

    .. list-table::
      :header-rows: 1
      :widths: 35 15 15 35

      * - lmcache version
        - ``/metrics`` on ``:9090``
        - ``/metrics`` on ``:8080``
        - Metric labels
      * - ``nightly-2026-04-25`` (lmcache ``0.4.5.dev31``)
        - ✅
        - 404
        - bare counters, no labels
      * - v0.4.5 stable (released 2026-05-15) and later
        - ✅
        - ✅
        - labeled with ``{cache_salt, model_name}``

.. rubric:: Cleanup

.. code-block:: bash

    kubectl delete -n dynamo-lmcache -f examples/backends/vllm/deploy/agg_lmcache.yaml
    kubectl -n dynamo-lmcache delete lmcacheengine lmcache-mp
    kubectl delete namespace dynamo-lmcache
    kubectl delete -f https://github.com/LMCache/LMCache/releases/download/operator-latest/install.yaml
    helm uninstall dynamo-platform -n dynamo-system
    kubectl delete namespace dynamo-system lmcache-operator-system
