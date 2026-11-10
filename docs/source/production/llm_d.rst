llm-d Integration
=================

`llm-d <https://github.com/llm-d/llm-d>`_ is an open-source, Kubernetes-native
stack for distributed LLM inference. Its router selects a vLLM pod for each
request. LMCache adds a KV cache layer behind those pods: each vLLM pod stores
KV cache in the LMCache server on its node and reads it back for a repeated
prefix.

The integration needs no change to llm-d. It uses the released llm-d router
chart, router image and P/D sidecar.

.. contents::
   :local:
   :depth: 2

How it fits together
--------------------

- The **llm-d Router** (the Endpoint Picker, EPP) scores the model-server pods
  for each request and picks one. In standalone mode, an Envoy sidecar in the
  EPP pod receives the requests, so no Gateway API provider is necessary.
- The **InferencePool** selects the model-server pods by label and names the
  port that the router sends requests to.
- The **model servers** are vLLM Deployments. LMCache connects here: each pod
  runs the ``lmcache/vllm-openai`` image and passes the LMCache connector
  config to vLLM with ``--kv-transfer-config``.
- The **LMCache server** runs once on each GPU node. The vLLM pods on that node
  share it.

.. code-block:: text

   client -> Envoy (8081) -> EPP picks a pod -> vLLM pod (8000)
                                                    |
                                                    | LMCacheMPConnector (CUDA IPC)
                                                    v
                                         LMCache server on the same node

Validated versions
~~~~~~~~~~~~~~~~~~

The steps on this page were run on one GPU node with these versions:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Component
     - Version
   * - llm-d router chart ``llm-d-router-standalone``
     - ``v1.0.0``
   * - llm-d P/D sidecar ``llm-d-router-disagg-sidecar``
     - ``v1.0.0``
   * - Gateway API Inference Extension (InferencePool CRD)
     - ``v1.6.2``
   * - LMCache server and vLLM image ``lmcache/vllm-openai``
     - ``v0.5.6``

Prerequisites
-------------

- A Kubernetes cluster with at least one GPU node and the NVIDIA GPU Operator.
  P/D disaggregation needs two GPUs on one node.
- ``kubectl`` and Helm 3.8 or later.
- A Hugging Face token if the model is gated. ``Qwen/Qwen3-0.6B`` is not gated.

Prefix KV reuse
---------------

Create the namespace
~~~~~~~~~~~~~~~~~~~~

The LMCache server, the router and the model servers share one namespace. A
pod can read a ConfigMap only from its own namespace.

.. code-block:: bash

   kubectl create namespace llm-d

Deploy the LMCache server
~~~~~~~~~~~~~~~~~~~~~~~~~

``lmcache_engine.yaml`` has three objects:

- A DaemonSet that runs ``lmcache server`` with ``--isolated-ipc`` on each GPU
  node.
- The ``lmcache-engine`` Service with ``internalTrafficPolicy: Local``. Each
  vLLM pod connects to the server on its own node, because CUDA IPC works only
  on one node.
- The ``lmcache-engine-connection`` ConfigMap. Its ``kv-transfer-config.json``
  key has the ``LMCacheMPConnector`` settings. The two other keys are for
  `P/D disaggregation`_.

.. literalinclude:: ../../../examples/llm_d_integration/kubernetes/lmcache_engine.yaml
   :language: yaml
   :caption: lmcache_engine.yaml

Apply the manifest from the root of the LMCache repository, and wait for the
server:

.. code-block:: bash

   kubectl apply -f examples/llm_d_integration/kubernetes/lmcache_engine.yaml
   kubectl -n llm-d rollout status ds/lmcache-engine
   kubectl -n llm-d logs ds/lmcache-engine | grep "cache server is running"
   # LMCache zmq cache server is running on 0.0.0.0:5555

Make sure that each server pod sees the GPUs of its node. The server opens the
KV cache of vLLM over CUDA IPC, so it must see every GPU, and it requests none:

.. code-block:: bash

   for p in $(kubectl -n llm-d get pod \
       -l app.kubernetes.io/instance=lmcache-engine -o name); do
     kubectl -n llm-d exec "$p" -- nvidia-smi -L | wc -l
   done

Each pod prints the GPU count of its node. If a pod prints an NVML error,
uncomment ``securityContext.privileged`` in the manifest and apply it again.

.. note::
   The ``lmcache.mp.isolated_ipc`` value in the ConfigMap must agree with the
   ``--isolated-ipc`` flag of the server. See the *Isolated IPC* section of the
   :doc:`/mp/deployment` guide. On a cluster without the ``nvidia``
   RuntimeClass, see *GPU Operator NRI/CDI* in the same guide.

Install the InferencePool CRD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The llm-d Router creates an ``InferencePool``. Its CRD comes from the Gateway
API Inference Extension release that llm-d v1.0.0 uses:

.. code-block:: bash

   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.6.2/v1-manifests.yaml
   kubectl wait --for=condition=Established \
       crd/inferencepools.inference.networking.k8s.io --timeout=60s

Install the llm-d Router
~~~~~~~~~~~~~~~~~~~~~~~~

The router selects the model-server pods by label and sends requests to the
vLLM port:

.. literalinclude:: ../../../examples/llm_d_integration/kubernetes/router_values.yaml
   :language: yaml
   :caption: router_values.yaml

.. code-block:: bash

   helm install llm-d oci://ghcr.io/llm-d/charts/llm-d-router-standalone \
       --version v1.0.0 --namespace llm-d \
       -f examples/llm_d_integration/kubernetes/router_values.yaml --wait

The chart names its objects after the release. ``llm-d-epp`` is the router
Deployment and Service, and ``llm-d`` is the InferencePool. Port 8081 of the
Service is the Envoy listener for clients.

.. note::
   The chart requests 4 CPUs and 8 Gi of memory for each of the EPP and Envoy
   containers. On a small cluster, set ``router.epp.resources`` and
   ``router.proxy.resources`` to lower values.

   With ``router.inferencePool.create: false``, the chart creates no
   InferencePool and the EPP selects the pods by label. The CRD is then not
   necessary.

Deploy the vLLM decode model server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The llm-d guides deploy their model servers from kustomize overlays.
``decode.yaml`` has the same shape as their ``decode`` base, with the LMCache
settings added:

.. literalinclude:: ../../../examples/llm_d_integration/kubernetes/decode.yaml
   :language: yaml
   :caption: decode.yaml

.. code-block:: bash

   kubectl apply -f examples/llm_d_integration/kubernetes/decode.yaml
   kubectl -n llm-d rollout status deploy/decode --timeout=15m

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Reason
   * - ``image: lmcache/vllm-openai``
     - The llm-d overlays run ``vllm/vllm-openai``, which has no LMCache
       client. The client and the server use a versioned protocol, so use the
       same image tag for both.
   * - ``PYTHONHASHSEED=0``
     - LMCache needs the same token hashes in all processes.
   * - ``LMCACHE_KV_CONFIG`` and ``$(LMCACHE_KV_CONFIG)``
     - Kubernetes reads the connector JSON from the ConfigMap when the pod
       starts and puts it into the vLLM arguments.
   * - ``llm-d.ai/guide: lmcache``
     - The label that ``modelServers.matchLabels`` of the router selects.
   * - ``--no-enable-prefix-caching``
     - With the prefix cache of vLLM on, vLLM serves a repeated prompt itself
       and the request does not reach LMCache. Keep the flag during
       verification. Remove it in production to use the two cache layers.
   * - Memory-backed ``/dev/shm``
     - The vLLM workers need more than the 64 Mi default of the container.

Verify
~~~~~~

Send requests to the router, not to the model pod:

.. code-block:: bash

   kubectl -n llm-d port-forward svc/llm-d-epp 8081:8081 &

Send one long prompt two times. The prompt must be longer than one LMCache
chunk (256 tokens by default):

.. code-block:: bash

   PROMPT="$(printf 'Tell me a long story about a brave knight who lived in a faraway kingdom. %.0s' $(seq 1 90))"
   for i in 1 2; do
     curl -s localhost:8081/v1/completions -H 'Content-Type: application/json' \
       -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"prompt\":\"$PROMPT\",\"max_tokens\":16}" \
       | jq -r '.choices[0].text'
   done

The LMCache server logs the store of the first request and the retrieve of
the second request:

.. code-block:: bash

   kubectl -n llm-d logs -l app.kubernetes.io/instance=lmcache-engine --tail=200 \
     | grep -E "Stored|Retrieved"
   # Stored 1280 tokens in 0.003 seconds
   # Retrieved 1280 tokens in 0.002 seconds

A ``Registered KV cache for GPU ID`` line in the same log shows that the
CUDA IPC handshake with vLLM is complete.

P/D disaggregation
------------------

The llm-d Router can send the prefill and the decode of each request to
different pods. One LMCache server is the KV cache layer of the two pods:

1. The router picks a decode pod and a prefill pod for each request. It selects
   them by the ``llm-d.ai/role`` label.
2. A routing sidecar in the decode pod sends the prompt to the prefill pod
   with ``max_tokens=1``.
3. The prefill pod stores the prefix in LMCache. For a repeated prompt, it
   reads the prefix from LMCache.
4. The sidecar sends the request to the vLLM of the decode pod, which pulls
   the KV cache from the prefill pod over NIXL and generates the output.

See :ref:`mp_disaggregated_prefill` for the connector design. This section
uses the namespace and the LMCache server of `Prefix KV reuse`_. Each vLLM pod
requests one GPU.

Install the router with the P/D plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These values follow the llm-d v1.0.0 ``pd-disaggregation`` guide. This release
creates no InferencePool: its EPP selects the pods by label.

.. literalinclude:: ../../../examples/llm_d_integration/kubernetes/pd_router_values.yaml
   :language: yaml
   :caption: pd_router_values.yaml

.. code-block:: bash

   helm upgrade --install llm-d-pd oci://ghcr.io/llm-d/charts/llm-d-router-standalone \
       --version v1.0.0 --namespace llm-d \
       -f examples/llm_d_integration/kubernetes/pd_router_values.yaml --wait

Deploy the prefill and decode pods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each pod reads the config of its role from the connection ConfigMap. Each
config is a ``MultiConnector`` with ``NixlConnector`` first and
``LMCacheMPConnector`` second. The decode pod also runs the llm-d routing
sidecar, which listens on port 8000 and sends requests to vLLM on port 8200.

.. literalinclude:: ../../../examples/llm_d_integration/kubernetes/pd_pods.yaml
   :language: yaml
   :caption: pd_pods.yaml

.. code-block:: bash

   kubectl apply -f examples/llm_d_integration/kubernetes/pd_pods.yaml
   kubectl -n llm-d rollout status deploy/prefill --timeout=20m
   kubectl -n llm-d rollout status deploy/decode-pd --timeout=20m

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Reason
   * - ``--disable-hybrid-kv-cache-manager``
     - ``LMCacheMPConnector`` needs it in this configuration. The llm-d guide
       does not set it, because its pods use ``NixlConnector`` only.
   * - ``VLLM_NIXL_SIDE_CHANNEL_HOST``
     - The NIXL side channel must listen on the pod IP, not on localhost.
   * - ``--no-enable-prefix-caching`` on prefill only
     - A repeated prompt goes to LMCache, so the retrieve shows in the server
       log. Remove it in production to use the two cache layers.
   * - ``VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120``
     - The sidecar keeps idle connections for 90 s. A shorter keep-alive of
       vLLM causes connection resets.
   * - ``--enable-prompt-tokens-details`` on decode
     - The response then reports ``cached_tokens``.

Verify P/D
~~~~~~~~~~

Send one long prompt two times through the P/D router:

.. code-block:: bash

   kubectl -n llm-d port-forward svc/llm-d-pd-epp 8082:8081 &
   PROMPT="$(printf 'Tell me a long story about a brave knight who lived in a faraway kingdom. %.0s' $(seq 1 90))"
   for i in 1 2; do
     curl -s localhost:8082/v1/completions -H 'Content-Type: application/json' \
       -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"prompt\":\"$PROMPT\",\"max_tokens\":16}" \
       | jq '.usage.prompt_tokens_details.cached_tokens'
   done
   # 0       first request
   # 1280    second request (the number depends on the prompt length)

Then examine each part of the path:

.. code-block:: bash

   # The decode pod pulled the KV cache over NIXL.
   kubectl -n llm-d logs deploy/decode-pd -c modelserver \
     | grep "NIXL compatibility check passed"

   # The prefill pod stored the prefix, then read it for the second request.
   kubectl -n llm-d logs -l app.kubernetes.io/instance=lmcache-engine --tail=200 \
     | grep -E "Stored|Retrieved"
   # Stored 1280 tokens in 0.004 seconds
   # Retrieved 1280 tokens in 0.001 seconds

   # Each pod served the two requests (vLLM request counters).
   for r in prefill:8000 decode:8200; do
     p=$(kubectl -n llm-d get pod \
         -l llm-d.ai/guide=lmcache-pd,llm-d.ai/role=${r%%:*} \
         -o jsonpath='{.items[0].metadata.name}')
     echo "${r%%:*}: $(kubectl get --raw \
         /api/v1/namespaces/llm-d/pods/$p:${r##*:}/proxy/metrics \
       | awk '/^vllm:request_success_total/ {s+=$2} END {print s}') requests"
   done
   # prefill: 2 requests
   # decode: 2 requests

Prefix-aware routing with two or more pods
------------------------------------------

The ``prefix-cache-scorer`` of the router compares token blocks, so the EPP
must tokenize each prompt. The default plugins of the chart have no tokenizer.
Without one, the scorer gives 0 to every pod and the router does not send a
repeated prefix to the pod that holds it.

Add the tokenizer sidecar of the chart and the ``token-producer`` plugin to
``router_values.yaml``. Use the model name of the decode pods, so that the
router and vLLM tokenize in the same way:

.. code-block:: yaml

   router:
     tokenizer:
       enabled: true
       flavor: python
       modelName: Qwen/Qwen3-0.6B
       port: 8000                      # loopback port in the EPP pod
       env: []                         # a gated model needs HF_TOKEN here
       resources:
         requests: { cpu: "1", memory: 2Gi }
         limits: { memory: 4Gi }
     epp:
       pluginsConfigFile: lmcache-plugins.yaml
       pluginsConfig:                  # the default plugins of the chart plus the token-producer
         apiVersion: llm-d.ai/v1
         kind: EndpointPickerConfig
         plugins:
           - type: token-producer
             parameters:
               modelName: Qwen/Qwen3-0.6B
               vllm: { url: http://localhost:8000, timeout: 10s }
           - type: queue-scorer
           - type: kv-cache-utilization-scorer
           - type: prefix-cache-scorer
           - type: metrics-data-source
             parameters: { scheme: http, path: /metrics, insecureSkipVerify: true }
           - type: core-metrics-extractor
         schedulingProfiles:
           - name: default
             plugins:
               - { pluginRef: queue-scorer, weight: 2 }
               - { pluginRef: kv-cache-utilization-scorer, weight: 2 }
               - { pluginRef: prefix-cache-scorer, weight: 3 }

With two decode pods on one node and these values, five repeats of one prompt
went to one pod: the scorer gave 1 to that pod and 0 to the other pod. The
same prompt, sent directly to the other pod, was read from the LMCache server
(``Retrieved`` in the server log), because the two pods share the server.

The scorer is approximate. It records the pod that received each prefix, and
it has no data about the KV cache that the LMCache server holds.

Use the LMCache Operator
------------------------

The :doc:`LMCache Operator </mp/operator>` creates the same DaemonSet, Service
and ConfigMap from an ``LMCacheEngine`` resource. The ConfigMap name is
``<engine-name>-connection``. With ``spec.pd``, the ConfigMap also has the
prefiller and decoder keys. Use that ConfigMap name in ``decode.yaml`` and
``pd_pods.yaml``.

The injection webhook of the operator does not change a container that sets
``command``. The llm-d model-server overlays set
``command: ["vllm", "serve"]``, so this page sets the connector config in the
Deployment. The steps on this page were run with ``lmcache_engine.yaml``, not
with the operator.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Symptom
     - Cause
     - Fix
   * - ``curl`` to the router returns 503 immediately after the install.
     - The EPP does not yet see a Ready model-server pod.
     - Wait until the decode Deployment is Ready.
   * - 503 continues, and the EPP log reports no endpoints.
     - ``modelServers.matchLabels`` does not agree with the pod labels, or
       ``targetPorts`` is not the vLLM port.
     - Use the same ``llm-d.ai/guide`` value on the two sides.
   * - ``no matches for kind "InferencePool"`` during the Helm install.
     - The InferencePool CRD is not installed.
     - Install the CRD, or set ``router.inferencePool.create=false``.
   * - vLLM stops at ``Registering kv caches`` and restarts. The LMCache
       server logs ``Device UUID ... not found in the discovered devices``.
     - The LMCache server pod does not see the GPUs of the node.
     - Run the server as privileged (see `Deploy the LMCache server`_).
   * - The LMCache server logs no ``Stored`` line.
     - The prompt is shorter than one chunk, or the request went to a server
       on a different node.
     - Use a longer prompt, and read the log of the server on the node of the
       vLLM pod.

Cleanup
-------

.. code-block:: bash

   kubectl -n llm-d delete -f examples/llm_d_integration/kubernetes/pd_pods.yaml
   kubectl -n llm-d delete -f examples/llm_d_integration/kubernetes/decode.yaml
   helm -n llm-d uninstall llm-d-pd llm-d
   kubectl delete -f examples/llm_d_integration/kubernetes/lmcache_engine.yaml
   kubectl delete namespace llm-d
