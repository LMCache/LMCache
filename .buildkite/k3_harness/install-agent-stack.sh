#!/usr/bin/env bash
# Install Buildkite agent-stack-k8s. Requires: setup-cluster.sh already ran.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.env"
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

TOKEN="${1:?Usage: $0 <BUILDKITE_AGENT_TOKEN>}"

if helm status agent-stack-k8s -n buildkite &>/dev/null; then
    echo "✓ agent-stack-k8s already installed. Use 'helm upgrade' to update."
    exit 0
fi

helm install agent-stack-k8s oci://ghcr.io/buildkite/helm/agent-stack-k8s \
    --namespace buildkite --create-namespace \
    --set agentToken="${TOKEN}" \
    --set config.org="${BUILDKITE_ORG}" \
    --set config.queue="${BUILDKITE_QUEUE}" \
    --wait --timeout 3m

echo "✓ agent-stack-k8s installed (queue=${BUILDKITE_QUEUE}, org=${BUILDKITE_ORG})"
kubectl get pods -n buildkite
