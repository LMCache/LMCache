# LMCache Disaggregated Prefill Example upon K8S (xP1D)

# The deployment consists of:
#   - Prefiller StatefulSet: Handles prefill operations (configurable replicas, default: 2)
#   - Decoder StatefulSet: Handles decode operations (single replica)
#   - Proxy Deployment: Routes requests between prefillers and decoder
#   - Services: Provides network connectivity between components

# Prerequisites:
#   - Kubernetes cluster with GPU support
#   - kubectl configured to access your cluster

# Steps:
cd examples/disagg_prefill/k8s

# 1. Create ConfigMaps for LMCache configuration files
kubectl create configmap lmcache-decoder-cfg --from-file=lmcache-decoder-config.yaml
kubectl create configmap lmcache-prefiller-cfg --from-file=lmcache-prefiller-config.yaml

# 2. Create ConfigMap for proxy code
# Note: Future will load script directly from built-in code in image
kubectl create configmap proxy-code --from-file=disagg_proxy_server_first_token_from_decoder.py

# 3. Deploy the application
# This creates:
#   - vllm-prefiller as StatefulSet (2 replicas by default)
#   - vllm-decoder as StatefulSet (1 replica)
#   - pd-proxy as Deployment
#   - Associated K8S Services
# The default configuration is 2P+1D (2 prefillers + 1 decoder)
kubectl apply -f xp1d.yaml

# Alternative: Single Prefiller Setup (1P1D)
# If you want to try 1P+1D:
# a) Edit xp1d.yaml: set replicas=1 for vllm-prefiller StatefulSet
# b) Edit xp1d.yaml: change prefillers value to "vllm-prefiller-0.vllm-prefiller-svc"
# c) Then run: kubectl apply -f xp1d.yaml

# 4. Wait until all pods become running
kubectl get pods -w
# Expected output:
# NAME                        READY   STATUS    RESTARTS   AGE
# pd-proxy-769f77cdd6-65v42   1/1     Running   0          8m8s
# vllm-decoder-0              1/1     Running   0          8m8s
# vllm-prefiller-0            1/1     Running   0          8m8s
# vllm-prefiller-1            1/1     Running   0          6m46s

# 5. Test the API
kubectl exec -it deploy/pd-proxy -- curl -X POST http://localhost:9000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/models/Qwen-2.5-0.5b",
    "prompt": "This is Greeting from LMCache",
    "max_tokens": 50
  }'

# 6. Monitor cache performance
# Check decoder logs for cache hit information:
kubectl logs vllm-decoder-0 | grep LMCache
# Expected log output:
# LMCache INFO: Reqid: cmpl-xxx, Total tokens 7, LMCache hit tokens: 7, need to load: 6
# LMCache DEBUG: Scheduled to load 7 tokens for request cmpl-xxx
# LMCache DEBUG: Retrieved 7 out of 7 out of total 7 tokens


# Cleanup (uncomment to run):
# kubectl delete -f xp1d.yaml
# kubectl delete configmap lmcache-decoder-cfg lmcache-prefiller-cfg proxy-code

# Notes:
# - Future versions will be updated once xPyD is supported

