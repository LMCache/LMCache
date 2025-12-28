# SPDX-License-Identifier: Apache-2.0
"""
Kubernetes Integration Tests for LMCache

This module contains integration tests for LMCache in different K8s environments.

Test Cases:
- test_base_integration: Base test with local CPU cache (any K8s cluster with GPU)
- test_sagemaker_hyperpod_integration: SageMaker HyperPod with ai-toolkit shared memory

Prerequisites:
- kubectl configured with access to a K8s cluster
- Cluster has GPU nodes available

Environment Variables:
- LMCACHE_IMAGE: Container image to use (default varies by test)
- K8S_SKIP_CLEANUP: Set to "1" to skip cleanup on failure (for debugging)
"""

# Standard
from pathlib import Path
import json
import subprocess

# Third Party
from common import K8sTestConfig, K8sTestHelper, prepare_manifest, run_inference_test
import pytest


@pytest.fixture(scope="function")
def check_prerequisites():
    """Check test prerequisites - fail if not met"""
    # Check kubectl is available
    try:
        subprocess.run(
            ["kubectl", "version", "--client"],
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.fail("kubectl not found. Please install kubectl.")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"kubectl not configured properly.\nError: {e.stderr}")

    # Check cluster connectivity
    try:
        subprocess.run(
            ["kubectl", "cluster-info"],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Cannot connect to Kubernetes cluster.\n"
            f"Error: {e.stderr}\n"
            f"Please ensure kubectl is configured with valid cluster credentials."
        )


def deploy_and_test(
    config: K8sTestConfig,
    manifest_path: Path,
    model: str,
):
    """
    Common deployment and testing logic

    Args:
        config: K8s test configuration
        manifest_path: Path to deployment YAML
        model: Model name for inference test
    """
    helper = K8sTestHelper(config)
    manifest_content = ""

    try:
        # Load and prepare manifest
        manifest_content = prepare_manifest(
            manifest_path, config.lmcache_image, config.hf_token
        )

        # Apply manifest
        helper.apply_manifest(manifest_content)

        # Wait for deployment pod to be ready
        helper.wait_for_pod_ready(f"app={config.deployment_name}", config.timeout)

        # Get pod IP
        pod_ip_result = helper.run_kubectl(
            [
                "get",
                "pod",
                "-n",
                config.namespace,
                f"-l app={config.deployment_name}",
                "-o",
                "jsonpath={.items[0].status.podIP}",
            ]
        )
        pod_ip = pod_ip_result.stdout.strip()

        # Wait for debug pod to be ready
        helper.wait_for_pod_ready(f"app={config.debug_pod_name}", 60)

        # Use pod IP directly
        service_url = f"http://{pod_ip}:8000/v1/completions"

        # Test inference
        response_data = run_inference_test(
            helper=helper,
            service_url=service_url,
            model=model,
            prompt="What is the capital of France?",
            max_tokens=100,
        )

        # Log success
        print("✅ Inference test passed")
        print(f"Response: {json.dumps(response_data, indent=2)}")

    finally:
        # Cleanup
        helper.cleanup_resources(manifest_content)


@pytest.mark.k8s
@pytest.mark.integration
@pytest.mark.base
def test_base_integration(check_prerequisites):
    """
    Test LMCache with local CPU cache

    Requirements:
    - Any K8s cluster with GPU nodes
    - No special setup required

    Configuration:
    - Local CPU cache (5GB)
    - Chunk size: 256 tokens
    - Model: Qwen/Qwen3-14B
    - Image: lmcache/vllm-openai:latest
    """
    config = K8sTestConfig(
        namespace="lmcache-base-test",
        deployment_name="lmcache-base",
        service_name="lmcache-base",
        debug_pod_name="lmcache-base-test-client",
    )

    manifest_path = Path(__file__).parent / "manifests" / "base.yaml"

    deploy_and_test(
        config=config,
        manifest_path=manifest_path,
        model="Qwen/Qwen3-14B",
    )


@pytest.mark.k8s
@pytest.mark.integration
@pytest.mark.sagemaker_hyperpod
def test_sagemaker_hyperpod_integration(check_prerequisites):
    """
    Test LMCache with SageMaker HyperPod ai-toolkit

    Requirements:
    - SageMaker HyperPod EKS cluster
    - ai-toolkit DaemonSet running on GPU nodes
    - Tiered storage enabled (20% memory allocation)
    - ai-toolkit shared memory cache at /dev/shm/ai_toolkit_cache

    Configuration:
    - Remote storage via ai-toolkit shared memory
    - Chunk size: 256 tokens
    - Model: Qwen/Qwen3-14B
    - Image: Custom ECR image with LMCache

    Setup:
    1. Enable tiered storage on HyperPod cluster:
       cd cluster-setup && ./enable-tiered-storage.sh

    2. Verify ai-toolkit is running:
       kubectl get ds -n aws-hyperpod ai-toolkit
       kubectl get pods -n aws-hyperpod -l app=ai-toolkit

    3. Verify shared memory cache exists:
       kubectl exec -n aws-hyperpod <ai-toolkit-pod> -- ls -lh /dev/shm/ai_toolkit_cache
    """
    config = K8sTestConfig(
        namespace="lmcache-sagemaker-hyperpod-test",
        deployment_name="lmcache-sagemaker-hyperpod",
        service_name="lmcache-sagemaker-hyperpod",
        debug_pod_name="lmcache-sagemaker-hyperpod-test-client",
    )

    manifest_path = Path(__file__).parent / "manifests" / "sagemaker-hyperpod.yaml"

    deploy_and_test(
        config=config,
        manifest_path=manifest_path,
        model="Qwen/Qwen3-14B",
    )


if __name__ == "__main__":
    # Allow running as standalone script
    pytest.main([__file__, "-v", "-s"])
