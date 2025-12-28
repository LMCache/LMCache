# LMCache Kubernetes Integration Tests

Python-based integration tests for LMCache in Kubernetes environments.

> **Note**: These tests are skipped in CI by default (require K8s cluster with GPU nodes). Set `RUN_K8S_TESTS=1` to enable them when a K8s cluster is available in CI/CD.

## Directory Structure

```
tests/k8s/
├── common.py                    # Shared test utilities and helpers
├── conftest.py                  # Pytest configuration and markers
├── test_integration.py          # Main integration test file
├── Dockerfile                   # Container for running tests
├── README.md                    # This file
└── manifests/                   # K8s deployment manifests
    ├── base.yaml                # Base test manifest
    └── sagemaker-hyperpod.yaml  # SageMaker HyperPod test manifest
```

## Test Cases

### Base Integration Test
Tests basic LMCache functionality with local CPU caching.

**Cluster Requirements:**
- K8s cluster with GPU nodes

### SageMaker HyperPod Integration Test
Tests LMCache with SageMaker HyperPod's ai-toolkit shared memory cache.

**Cluster Requirements:**
- SageMaker HyperPod EKS cluster with GPU nodes
- Tiered storage enabled (see [AWS documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/managed-tier-checkpointing-setup.html))

## Prerequisites

- Docker
- kubectl
- For Amazon EKS clusters: AWS credentials configured (required for authentication)

## Running Tests

### Authenticate with the cluster

For Amazon EKS clusters, configure kubectl:
```bash
# Set cluster information
export AWS_REGION="us-west-2"
export CLUSTER_NAME="your-cluster-name"

# Configure kubectl for your EKS cluster
aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME

# Verify cluster access
kubectl cluster-info
```

### Set environment variables

```bash
export LMCACHE_IMAGE="lmcache/vllm-openai:latest"  # or your custom image
```

### Build the test container
```bash
docker build --platform linux/amd64 -t lmcache-k8s-tests -f tests/k8s/Dockerfile tests/k8s
```

### Run all tests
```bash
docker run --rm \
  --platform linux/amd64 \
  -v ~/.kube:/root/.kube:ro \
  -v ~/.aws:/root/.aws:ro \
  -v $(pwd)/test-logs:/logs \
  -e LMCACHE_IMAGE \
  lmcache-k8s-tests
```

### Run specific tests
```bash
# Base test
docker run --rm \
  --platform linux/amd64 \
  -v ~/.kube:/root/.kube:ro \
  -v ~/.aws:/root/.aws:ro \
  -v $(pwd)/test-logs:/logs \
  -e LMCACHE_IMAGE \
  lmcache-k8s-tests pytest -m base -v -s

# SageMaker HyperPod test
docker run --rm \
  --platform linux/amd64 \
  -v ~/.kube:/root/.kube:ro \
  -v ~/.aws:/root/.aws:ro \
  -v $(pwd)/test-logs:/logs \
  -e LMCACHE_IMAGE \
  lmcache-k8s-tests pytest -m sagemaker_hyperpod -v -s
```

### Debug mode
Skip cleanup on failure to inspect resources:
```bash
docker run --rm \
  --platform linux/amd64 \
  -v ~/.kube:/root/.kube:ro \
  -v ~/.aws:/root/.aws:ro \
  -v $(pwd)/test-logs:/logs \
  -e LMCACHE_IMAGE \
  -e K8S_SKIP_CLEANUP=1 \
  lmcache-k8s-tests pytest -m base -v -s
```

Test logs are saved to `./test-logs/` directory with timestamps for debugging.
