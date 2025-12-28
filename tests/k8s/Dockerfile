# SPDX-License-Identifier: Apache-2.0
# Dockerfile for LMCache K8s Integration Tests
#
# This container includes kubectl, AWS CLI, and all Python dependencies needed
# to run the K8s integration tests out of the box.

FROM python:3.10-slim

# Version pinning for reproducibility
ARG KUBECTL_VERSION=v1.35.0
ARG AWSCLI_VERSION=2.32.24

# Install kubectl, AWS CLI, and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl (pinned version)
RUN curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Install AWS CLI v2 (pinned version)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-${AWSCLI_VERSION}.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Set working directory
WORKDIR /workspace

# Copy test files
COPY common.py conftest.py test_integration.py requirements.txt ./
COPY manifests/ ./manifests/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV RUN_K8S_TESTS=1

# Default command runs pytest with verbose output
CMD ["pytest", "test_integration.py", "-v", "-s"]
