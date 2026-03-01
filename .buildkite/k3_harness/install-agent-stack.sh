#!/usr/bin/env bash
# Install Buildkite agent-stack-k8s with git SSH credentials.
#
# Usage:
#   install-agent-stack.sh <BUILDKITE_AGENT_TOKEN> <SSH_PRIVATE_KEY_PATH>
#
# Arguments:
#   BUILDKITE_AGENT_TOKEN  — from Buildkite cluster settings
#   SSH_PRIVATE_KEY_PATH   — private key authorized to clone the repo from GitHub
#
# The queue name defaults to "k8s". Override with BUILDKITE_QUEUE env var.
set -euo pipefail

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

TOKEN="${1:?Usage: $0 <BUILDKITE_AGENT_TOKEN> <SSH_PRIVATE_KEY_PATH>}"
SSH_KEY="${2:?Usage: $0 <BUILDKITE_AGENT_TOKEN> <SSH_PRIVATE_KEY_PATH>}"
QUEUE="${BUILDKITE_QUEUE:-k8s}"

if [[ ! -f "$SSH_KEY" ]]; then
    echo "Error: SSH key not found: $SSH_KEY"
    exit 1
fi

# Create (or update) the K8s secret with the SSH private key
kubectl create secret generic buildkite-git-ssh \
    --from-file=SSH_PRIVATE_ED25519_KEY="$SSH_KEY" \
    -n buildkite --dry-run=client -o yaml | kubectl apply -f -

# Install or upgrade agent-stack-k8s
helm upgrade --install agent-stack-k8s oci://ghcr.io/buildkite/helm/agent-stack-k8s \
    --namespace buildkite --create-namespace \
    --set agentToken="${TOKEN}" \
    --set config.queue="${QUEUE}" \
    --set-json 'config.pod-spec-patch={"containers":[{"name":"checkout","envFrom":[{"secretRef":{"name":"buildkite-git-ssh"}}]}]}' \
    --wait --timeout 3m

echo "✓ agent-stack-k8s installed (queue=${QUEUE})"
kubectl get pods -n buildkite
