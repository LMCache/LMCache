# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration for K8s integration tests
"""

# Standard
import os

# Third Party
import pytest


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "k8s: mark test as requiring Kubernetes cluster access"
    )
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line(
        "markers", "base: mark test as base integration test (any K8s cluster)"
    )
    config.addinivalue_line(
        "markers",
        "sagemaker_hyperpod: mark test as requiring SageMaker HyperPod environment",
    )


def pytest_collection_modifyitems(config, items):
    """Skip K8s tests in CI unless explicitly enabled"""
    # Skip K8s tests in CI by default (unless RUN_K8S_TESTS=1)
    if os.getenv("RUN_K8S_TESTS") != "1":
        skip_k8s = pytest.mark.skip(
            reason="K8s tests require cluster access. Set RUN_K8S_TESTS=1 to run them."
        )
        for item in items:
            if "k8s" in item.keywords:
                item.add_marker(skip_k8s)
