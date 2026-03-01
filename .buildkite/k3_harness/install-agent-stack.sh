#!/usr/bin/env bash
# Install Buildkite agent-stack-k8s with GitHub token for HTTPS repo access.
#
# Usage:
#   install-agent-stack.sh <BUILDKITE_AGENT_TOKEN> <GITHUB_TOKEN>
#
# Arguments:
#   BUILDKITE_AGENT_TOKEN  — from Buildkite cluster settings
#   GITHUB_TOKEN           — GitHub PAT or fine-grained token with repo read/write
#
# The queue name defaults to "k8s". Override with BUILDKITE_QUEUE env var.
set -euo pipefail

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

TOKEN="${1:?Usage: $0 <BUILDKITE_AGENT_TOKEN> <GITHUB_TOKEN>}"
GH_TOKEN="${2:?Usage: $0 <BUILDKITE_AGENT_TOKEN> <GITHUB_TOKEN>}"
QUEUE="${BUILDKITE_QUEUE:-k8s}"

# Create (or update) the K8s secret with GitHub credentials.
# Two keys:
#   git-credentials  — used by agent-stack-k8s checkout container (HTTPS clone)
#   GITHUB_TOKEN     — injected into job containers for git push (e.g. baselines)
kubectl create secret generic buildkite-git-creds \
    --from-literal=git-credentials="https://x-access-token:${GH_TOKEN}@github.com" \
    --from-literal=GITHUB_TOKEN="${GH_TOKEN}" \
    -n buildkite --dry-run=client -o yaml | kubectl apply -f -

# Install or upgrade agent-stack-k8s
# - git-credentials-secret: checkout container uses HTTPS instead of SSH
# - pod-spec-patch: injects GITHUB_TOKEN into job containers for push operations
helm upgrade --install agent-stack-k8s oci://ghcr.io/buildkite/helm/agent-stack-k8s \
    --namespace buildkite --create-namespace \
    --set agentToken="${TOKEN}" \
    --set config.queue="${QUEUE}" \
    --set-json 'config.git-credentials-secret={"name":"buildkite-git-creds","key":"git-credentials"}' \
    --set-json 'config.pod-spec-patch={"containers":[{"name":"container-0","env":[{"name":"GITHUB_TOKEN","valueFrom":{"secretKeyRef":{"name":"buildkite-git-creds","key":"GITHUB_TOKEN"}}}]}]}' \
    --wait --timeout 3m

echo "agent-stack-k8s installed (queue=${QUEUE})"
kubectl get pods -n buildkite
