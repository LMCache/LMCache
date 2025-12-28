# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration for K8s integration tests
"""

# Third Party


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
