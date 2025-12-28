# SPDX-License-Identifier: Apache-2.0
"""
Common utilities for K8s integration tests
"""

# Standard
from datetime import datetime
from pathlib import Path
from typing import Optional
import base64
import json
import logging
import os
import subprocess


# Setup logging
def setup_logging(log_dir: str = "/logs") -> logging.Logger:
    """Setup logging to both console and file"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"k8s_test_{timestamp}.log"

    logger = logging.getLogger("k8s_test")
    logger.setLevel(logging.DEBUG)

    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler - summary only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to {log_file}")
    return logger


logger = setup_logging()


class K8sTestConfig:
    """Configuration for K8s integration tests"""

    def __init__(
        self,
        namespace: str,
        deployment_name: str,
        service_name: str,
        debug_pod_name: str,
        timeout: int = 1800,  # 30 minutes for image pull
        test_timeout: int = 60,
    ):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.service_name = service_name
        self.debug_pod_name = debug_pod_name
        self.timeout = timeout
        self.test_timeout = test_timeout
        self.lmcache_image = os.getenv("LMCACHE_IMAGE")
        if not self.lmcache_image:
            raise ValueError(
                "LMCACHE_IMAGE environment variable must be set. "
                "Example: export LMCACHE_IMAGE='lmcache/vllm-openai:latest'"
            )
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.skip_cleanup = os.getenv("K8S_SKIP_CLEANUP", "0") == "1"

        logger.info(
            f"Test configuration: namespace={namespace}, "
            f"image={self.lmcache_image}, timeout={timeout}s"
        )


class K8sTestHelper:
    """Helper class for K8s operations"""

    def __init__(self, config: K8sTestConfig):
        self.config = config

    def run_kubectl(
        self, args: list[str], check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run kubectl command"""
        cmd = ["kubectl"] + args
        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.debug(f"Command failed with exit code {result.returncode}")
            logger.debug(f"stdout: {result.stdout}")
            logger.debug(f"stderr: {result.stderr}")
        if check and result.returncode != 0:
            raise RuntimeError(
                f"kubectl command failed: {' '.join(cmd)}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        return result

    def apply_manifest(self, manifest_content: str) -> None:
        """Apply K8s manifest"""
        logger.info("Applying K8s manifest")
        logger.debug(f"Manifest content:\n{manifest_content}")
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=manifest_content,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error(f"Failed to apply manifest: {result.stderr}")
            raise RuntimeError(
                f"Failed to apply manifest:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        logger.info("Manifest applied successfully")

    def wait_for_pod_ready(self, label_selector: str, timeout: int) -> None:
        """Wait for pod to be ready"""
        logger.info(
            f"Waiting for pod with label {label_selector} to be ready "
            f"(timeout: {timeout}s)"
        )
        result = self.run_kubectl(
            [
                "wait",
                "--for=condition=ready",
                "pod",
                f"-l{label_selector}",
                f"-n{self.config.namespace}",
                f"--timeout={timeout}s",
            ],
            check=False,
        )
        if result.returncode != 0:
            logger.error(f"Pod failed to become ready within {timeout}s")
            # Get pod status for debugging
            pod_status = self.run_kubectl(
                [
                    "get",
                    "pods",
                    "-n",
                    self.config.namespace,
                    f"-l{label_selector}",
                    "-o",
                    "wide",
                ],
                check=False,
            )
            pod_describe = self.run_kubectl(
                [
                    "describe",
                    "pods",
                    "-n",
                    self.config.namespace,
                    f"-l{label_selector}",
                ],
                check=False,
            )
            pod_logs = self.run_kubectl(
                [
                    "logs",
                    "-n",
                    self.config.namespace,
                    f"-l{label_selector}",
                    "--tail=100",
                    "--all-containers=true",
                ],
                check=False,
            )

            # Log detailed debugging info
            logger.debug(f"Pod status:\n{pod_status.stdout}")
            logger.debug(f"Pod describe:\n{pod_describe.stdout}")
            logger.debug(f"Pod logs:\n{pod_logs.stdout}")

            raise RuntimeError(
                f"Pod failed to become ready within {timeout}s\n"
                f"Wait command error:\n{result.stderr}\n\n"
                f"Pod status:\n{pod_status.stdout}\n\n"
                f"Pod describe:\n{pod_describe.stdout}\n\n"
                f"Pod logs (last 100 lines):\n{pod_logs.stdout}\n"
                f"Pod logs stderr:\n{pod_logs.stderr}"
            )
        logger.info("Pod is ready")

    def exec_in_pod(
        self, pod_name: str, command: list[str]
    ) -> subprocess.CompletedProcess:
        """Execute command in pod"""
        return self.run_kubectl(
            ["exec", pod_name, "-n", self.config.namespace, "--"] + command
        )

    def cleanup_resources(self, manifest_content: str) -> None:
        """Cleanup all test resources using the manifest"""
        if self.config.skip_cleanup:
            logger.info("Skipping cleanup (K8S_SKIP_CLEANUP=1)")
            return

        logger.info("Cleaning up test resources")
        result = subprocess.run(
            ["kubectl", "delete", "-f", "-"],
            input=manifest_content,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                f"Cleanup had errors (this is often normal): {result.stderr}"
            )
        logger.info("Cleanup complete")


def prepare_manifest(
    manifest_path: Path,
    lmcache_image: str,
    hf_token: str,
    extra_substitutions: Optional[dict[str, str]] = None,
) -> str:
    """
    Load and prepare K8s manifest with variable substitution

    Args:
        manifest_path: Path to the manifest YAML file
        lmcache_image: Container image to use
        hf_token: HuggingFace token
        extra_substitutions: Additional key-value pairs for substitution

    Returns:
        Prepared manifest content with substitutions applied
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_content = f.read()

    # Encode HF_TOKEN as base64 for Secret (if present in manifest)
    if "${HF_TOKEN}" in manifest_content:
        hf_token_b64 = base64.b64encode(hf_token.encode()).decode()
        manifest_content = manifest_content.replace("${HF_TOKEN}", hf_token_b64)

    # Substitute image
    manifest_content = manifest_content.replace("${LMCACHE_IMAGE}", lmcache_image)

    # Apply extra substitutions
    if extra_substitutions:
        for key, value in extra_substitutions.items():
            manifest_content = manifest_content.replace(f"${{{key}}}", value)

    return manifest_content


def run_inference_test(
    helper: K8sTestHelper,
    service_url: str,
    model: str,
    prompt: str = "What is the capital of France?",
    max_tokens: int = 100,
) -> dict:
    """
    Test inference endpoint from debug pod

    Args:
        helper: K8s test helper
        service_url: Service URL to test
        model: Model name
        prompt: Inference prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Response data as dict

    Raises:
        AssertionError: If inference test fails
    """
    logger.info(f"Testing inference at {service_url}")
    # Prepare inference request
    request_data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    logger.debug(f"Request data: {json.dumps(request_data, indent=2)}")

    # Send request from debug pod
    curl_cmd = [
        "curl",
        "--max-time",
        str(helper.config.test_timeout),
        "-w",
        "%{http_code}",
        "-o",
        "/tmp/response.json",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(request_data),
        service_url,
    ]

    result = helper.exec_in_pod(helper.config.debug_pod_name, curl_cmd)
    http_status = result.stdout.strip()
    logger.debug(f"HTTP status: {http_status}")

    # Verify HTTP status
    if http_status != "200":
        # Get curl stderr for more details
        curl_stderr = result.stderr if result.stderr else "No stderr output"
        logger.error(f"Inference request failed with HTTP {http_status}")
        logger.debug(f"curl stderr: {curl_stderr}")
        raise AssertionError(
            f"Expected HTTP 200, got {http_status}\n"
            f"curl stderr: {curl_stderr}\n"
            f"Service URL: {service_url}"
        )

    # Get response body
    response_result = helper.exec_in_pod(
        helper.config.debug_pod_name, ["cat", "/tmp/response.json"]
    )
    response_text = response_result.stdout
    logger.debug(f"Response: {response_text}")

    # Parse and validate response
    try:
        response_data = json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse response JSON: {e}")
        raise AssertionError(
            f"Failed to parse response JSON: {e}\nResponse: {response_text}"
        ) from e

    # Validate response structure
    assert "choices" in response_data, "Response missing 'choices' field"
    assert len(response_data["choices"]) > 0, "Response has no choices"
    assert "text" in response_data["choices"][0], (
        "Response missing 'choices[0].text' field"
    )

    logger.info("Inference test passed")
    return response_data
