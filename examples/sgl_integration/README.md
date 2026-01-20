# SGLang & LMCache Integration

This example shows how to use SGLang & LMCache Integration.

## Install
This project depends on a pending pull request in the SGLang repository. Until PR is merged, please use the code from that specific branch instead of the SGLang main branch.
```bash
git clone https://github.com/Oasis-Git/sglang/tree/lmcache
cd sglang

pip install --upgrade pip
pip install -e "python[all]"
```

## Server script
To start SGLang server with LMCache, run
```bash
export LMCACHE_CONFIG_FILE=lmcache_config.yaml
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000 --tp 2 --page-size 32 --enable-lmcache
```
If you hope to run the benchmark, please refer to `https://github.com/sgl-project/sglang/tree/main/benchmark/hicache`



## Kubernetes Deployment

This guide shows how to deploy SGLang with LMCache on Kubernetes using a custom Docker image with compiled CUDA extensions.

### Prerequisites

- Kubernetes cluster with GPU nodes
- Docker (with buildx for ARM64 builds)
- Container registry (e.g., AWS ECR, Docker Hub)
- kubectl configured to access your cluster

### Environment Variables

Set these environment variables before starting:

```bash
export REGISTRY=your-registry.example.com
export IMAGE_NAME=lmcache-sglang
export IMAGE_TAG=latest
# Pin versions for reproducibility
export SGLANG_VERSION=v0.5.7
export LMCACHE_VERSION=0.3.12
```

**Note:** Use `SGLANG_VERSION=latest` for the latest SGLang version, and `LMCACHE_VERSION=dev` to build from local source. If environment variables are not set, defaults are `latest` and `dev` respectively.

### Building the Docker Image

Build and push the Docker image to your container registry.

Below are example steps for Amazon ECR:

**Step 1: Authenticate with Amazon ECR**

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ${REGISTRY}
```

**Step 2: Build and push the image**

For standard Linux/x86_64 environments:

```bash
docker build -f docker/Dockerfile.sglang \
  --build-arg SGLANG_VERSION=${SGLANG_VERSION} \
  --build-arg LMCACHE_VERSION=${LMCACHE_VERSION} \
  -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .
docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
```

For ARM64 (e.g., macOS) cross-platform builds:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.sglang \
  --build-arg SGLANG_VERSION=${SGLANG_VERSION} \
  --build-arg LMCACHE_VERSION=${LMCACHE_VERSION} \
  -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \
  --push .
```

### Deploying to Kubernetes

**Step 1: Update the deployment YAML with your image**

Replace the placeholder image with your actual image:

```bash
# For Linux
sed -i "s|<YOUR_REGISTRY>/<YOUR_IMAGE_NAME>:<YOUR_TAG>|${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}|g" \
  examples/sgl_integration/sglang-deployment.yaml

# For macOS
sed -i '' "s|<YOUR_REGISTRY>/<YOUR_IMAGE_NAME>:<YOUR_TAG>|${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}|g" \
  examples/sgl_integration/sglang-deployment.yaml
```

**Step 2: Apply the deployment**

```bash
kubectl create namespace sgl-integration
kubectl apply -f examples/sgl_integration/sglang-deployment.yaml
```

**Step 3: Wait for pod to be ready**

```bash
kubectl get pods -n sgl-integration -w
# Wait until STATUS shows "Running" and READY shows "1/1"
```

### Testing the Deployment

**Step 1: Port forward to the service**

```bash
kubectl port-forward -n sgl-integration svc/sglang-lmcache 8000:8000 &
```

**Step 2: Send test requests**

Send a completion request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-14B", "prompt": "Explain the concept of machine learning", "max_tokens": 50}'
```

**Step 3: Verify LMCache is active**

Check for LMCache activity in the logs:

```bash
POD=$(kubectl get pods -n sgl-integration -l app=sglang-lmcache -o jsonpath='{.items[0].metadata.name}')
kubectl logs -n sgl-integration $POD | grep -i lmcache
```

You should see log messages indicating LMCache is storing KV cache data:

```
[2026-01-03 10:32:15] LMCache DEBUG: Finished offloading layer 0
[2026-01-03 10:32:15] LMCache DEBUG: Finished offloading layer 1
...
```

#### Understanding Cache Behavior

LMCache stores KV cache data to CPU memory, which you can observe through "Finished offloading layer X" messages in the logs. This enables efficient memory management and cache sharing across requests.

### Cleanup

To remove the deployment and clean up resources:

```bash
kubectl delete -f examples/sgl_integration/sglang-deployment.yaml
kubectl delete namespace sgl-integration
```
