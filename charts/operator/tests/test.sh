#!/usr/bin/env bash
# Helm chart template rendering tests for the LMCache operator chart.
#
# Each test renders the chart with a specific values file and asserts
# that the expected resources are present (or absent), and that key
# fields have the correct values.  No Kubernetes cluster is needed.
#
# Usage:  bash charts/operator/tests/test.sh
#
# Exits 0 if all tests pass, 1 otherwise.

set -uo pipefail

CHART_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VALUES_DIR="$CHART_DIR/tests/values"
RELEASE="lmcache-operator"
NAMESPACE="lmcache-operator-system"
FAIL=0
PASS=0

# ── Helpers ──────────────────────────────────────────────────────────────────

# Render the chart with a values file and print all rendered manifests.
render() {
  local values_file="$1"
  helm template "$RELEASE" "$CHART_DIR" \
    --namespace "$NAMESPACE" \
    --values "$values_file" 2>/dev/null || true
}

# Render the chart with a values file and show only resource kinds.
kinds() {
  render "$1" | grep '^kind: ' | sed 's/kind: //' | sort
}

# Assert that a string IS present in rendered output.
assert_contains() {
  local label="$1" values_file="$2" pattern="$3"
  if render "$values_file" | grep -q "$pattern"; then
    PASS=$((PASS + 1))
    printf "  ✓ %s\n" "$label"
  else
    FAIL=$((FAIL + 1))
    printf "  ✗ %s — expected \"%s\" in %s\n" "$label" "$pattern" "$(basename "$values_file")"
  fi
}

# Assert that a string is NOT present in rendered output.
assert_not_contains() {
  local label="$1" values_file="$2" pattern="$3"
  if render "$values_file" | grep -q "$pattern"; then
    FAIL=$((FAIL + 1))
    printf "  ✗ %s — did not expect \"%s\" in %s\n" "$label" "$pattern" "$(basename "$values_file")"
  else
    PASS=$((PASS + 1))
    printf "  ✓ %s\n" "$label"
  fi
}

# Assert a resource kind IS present.
assert_has_kind() {
  local label="$1" values_file="$2" kind="$3"
  if kinds "$values_file" | grep -qx "$kind"; then
    PASS=$((PASS + 1))
    printf "  ✓ %s\n" "$label"
  else
    FAIL=$((FAIL + 1))
    printf "  ✗ %s — expected kind %s in %s\n" "$label" "$kind" "$(basename "$values_file")"
  fi
}

# Assert a resource kind is NOT present.
assert_no_kind() {
  local label="$1" values_file="$2" kind="$3"
  if kinds "$values_file" | grep -qx "$kind"; then
    FAIL=$((FAIL + 1))
    printf "  ✗ %s — did not expect kind %s in %s\n" "$label" "$kind" "$(basename "$values_file")"
  else
    PASS=$((PASS + 1))
    printf "  ✓ %s\n" "$label"
  fi
}

# ── Tests ────────────────────────────────────────────────────────────────────

echo "Running LMCache operator Helm chart tests..."
echo ""

# ── Scenario: Default (all features enabled) ─────────────────────────────────
echo "── Scenario: default (all features on) ──"

assert_has_kind "ServiceAccount created" \
  "$VALUES_DIR/minimal.yaml" "ServiceAccount"
assert_has_kind "ClusterRole (manager) created" \
  "$VALUES_DIR/minimal.yaml" "ClusterRole"
assert_has_kind "ClusterRoleBinding created" \
  "$VALUES_DIR/minimal.yaml" "ClusterRoleBinding"
assert_has_kind "Role (leader election) created" \
  "$VALUES_DIR/minimal.yaml" "Role"
assert_has_kind "RoleBinding (leader election) created" \
  "$VALUES_DIR/minimal.yaml" "RoleBinding"
assert_has_kind "Service (metrics) created" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "Service"
assert_has_kind "Service (webhook) created" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "Service"
assert_has_kind "Deployment created" \
  "$VALUES_DIR/minimal.yaml" "Deployment"
assert_has_kind "MutatingWebhookConfiguration created" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "MutatingWebhookConfiguration"
assert_has_kind "Issuer (cert-manager) created" \
  "$VALUES_DIR/no-metrics.yaml" "Issuer"
assert_has_kind "Certificate (cert-manager) created" \
  "$VALUES_DIR/no-metrics.yaml" "Certificate"
assert_has_kind "CRD (LMCacheEngine) created" \
  "$VALUES_DIR/minimal.yaml" "CustomResourceDefinition"
assert_contains "Leader election arg present" \
  "$VALUES_DIR/minimal.yaml" "\-\-leader-elect"
assert_contains "Metrics bind address arg present" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "\-\-metrics-bind-address=:8443"
assert_contains "Webhook cert path arg present" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "\-\-webhook-cert-path=/tmp/k8s-webhook-server/serving-certs"
assert_contains "cert-manager CA injection annotation present" \
  "$VALUES_DIR/no-metrics.yaml" "cert-manager.io/inject-ca-from"
assert_contains "Default image reference" \
  "$VALUES_DIR/minimal.yaml" "lmcache/lmcache-operator:v0.1.1"
assert_contains "Release name in Deployment" \
  "$VALUES_DIR/minimal.yaml" "name: $RELEASE-controller-manager"
assert_contains "Namespace in Deployment" \
  "$VALUES_DIR/minimal.yaml" "namespace: $NAMESPACE"

echo ""

# ── Scenario: Minimal (all optional features off) ─────────────────────────────
echo "── Scenario: minimal (webhook/metrics/cert-manager/crd-roles off) ──"

assert_no_kind "No MutatingWebhookConfiguration" \
  "$VALUES_DIR/minimal.yaml" "MutatingWebhookConfiguration"
assert_no_kind "No Issuer" \
  "$VALUES_DIR/minimal.yaml" "Issuer"
assert_no_kind "No Certificate" \
  "$VALUES_DIR/minimal.yaml" "Certificate"
assert_no_kind "No Service" \
  "$VALUES_DIR/minimal.yaml" "Service"
assert_not_contains "No --metrics-bind-address arg" \
  "$VALUES_DIR/minimal.yaml" "\-\-metrics-bind-address"
assert_not_contains "No --webhook-cert-path arg" \
  "$VALUES_DIR/minimal.yaml" "\-\-webhook-cert-path"
assert_not_contains "No cert-manager annotation" \
  "$VALUES_DIR/minimal.yaml" "cert-manager.io/inject-ca-from"
assert_not_contains "No metrics-auth-role" \
  "$VALUES_DIR/minimal.yaml" "metrics-auth-role"
assert_not_contains "No metrics-reader" \
  "$VALUES_DIR/minimal.yaml" "metrics-reader"
assert_not_contains "No admin role" \
  "$VALUES_DIR/minimal.yaml" "lmcacheengine-admin-role"
assert_has_kind "Deployment still created" \
  "$VALUES_DIR/minimal.yaml" "Deployment"
assert_has_kind "ClusterRole (manager) still created" \
  "$VALUES_DIR/minimal.yaml" "ClusterRole"
assert_has_kind "ServiceAccount still created" \
  "$VALUES_DIR/minimal.yaml" "ServiceAccount"
assert_contains "Leader election still enabled" \
  "$VALUES_DIR/minimal.yaml" "\-\-leader-elect"

echo ""

# ── Scenario: CRD install disabled ─────────────────────────────────────────────
echo "── Scenario: crdInstall=false ──"

assert_no_kind "No CRDs when crdInstall=false" \
  "$VALUES_DIR/no-crds.yaml" "CustomResourceDefinition"
assert_has_kind "Deployment still created" \
  "$VALUES_DIR/no-crds.yaml" "Deployment"
assert_has_kind "ServiceAccount still created" \
  "$VALUES_DIR/no-crds.yaml" "ServiceAccount"
assert_has_kind "ClusterRole still created" \
  "$VALUES_DIR/no-crds.yaml" "ClusterRole"

echo ""

# ── Scenario: Webhook on, cert-manager off ────────────────────────────────────
echo "── Scenario: webhook on, cert-manager off ──"

assert_has_kind "MutatingWebhookConfiguration created" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "MutatingWebhookConfiguration"
assert_no_kind "No Issuer" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "Issuer"
assert_no_kind "No Certificate" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "Certificate"
assert_not_contains "No cert-manager CA injection annotation" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "cert-manager.io/inject-ca-from"
assert_has_kind "Service (webhook) still created" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "Service"
assert_contains "Webhook cert path still in args" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "\-\-webhook-cert-path"

echo ""

# ── Scenario: Metrics off, webhook+cert-manager on ───────────────────────────
echo "── Scenario: metrics off, webhook+cert-manager on ──"

assert_not_contains "No metrics Service name" \
  "$VALUES_DIR/no-metrics.yaml" "metrics-service"
assert_not_contains "No --metrics-bind-address arg" \
  "$VALUES_DIR/no-metrics.yaml" "\-\-metrics-bind-address"
assert_contains "Webhook Service still present" \
  "$VALUES_DIR/no-metrics.yaml" "webhook-service"
assert_not_contains "No metrics-auth-role" \
  "$VALUES_DIR/no-metrics.yaml" "metrics-auth-role"
assert_not_contains "No metrics-reader" \
  "$VALUES_DIR/no-metrics.yaml" "metrics-reader"
assert_has_kind "MutatingWebhookConfiguration still present" \
  "$VALUES_DIR/no-metrics.yaml" "MutatingWebhookConfiguration"
assert_has_kind "Issuer still present" \
  "$VALUES_DIR/no-metrics.yaml" "Issuer"

echo ""

# ── Scenario: Custom image, resources, replicas, scheduling ───────────────────
echo "── Scenario: custom image, resources, replicas, scheduling ──"

assert_contains "Custom image repository:tag" \
  "$VALUES_DIR/custom-image-resources.yaml" "myregistry.io/lmcache-operator:v0.2.0"
assert_contains "Image pull policy Always" \
  "$VALUES_DIR/custom-image-resources.yaml" "imagePullPolicy: Always"
assert_contains "Replicas set to 3" \
  "$VALUES_DIR/custom-image-resources.yaml" "replicas: 3"
assert_contains "Custom CPU limit" \
  "$VALUES_DIR/custom-image-resources.yaml" "cpu: \"1\""
assert_contains "Custom memory limit" \
  "$VALUES_DIR/custom-image-resources.yaml" "memory: 256Mi"
assert_contains "Custom CPU request" \
  "$VALUES_DIR/custom-image-resources.yaml" "cpu: 200m"
assert_contains "Custom nodeSelector" \
  "$VALUES_DIR/custom-image-resources.yaml" "node-role.kubernetes.io/control-plane"
assert_contains "Custom toleration key" \
  "$VALUES_DIR/custom-image-resources.yaml" "node-role.kubernetes.io/control-plane"
assert_contains "Custom toleration effect" \
  "$VALUES_DIR/custom-image-resources.yaml" "NoSchedule"
assert_contains "Custom pod annotation" \
  "$VALUES_DIR/custom-image-resources.yaml" "custom.annotation/example"
assert_contains "Custom pod label" \
  "$VALUES_DIR/custom-image-resources.yaml" "custom-label: value"
assert_contains "Common label propagated" \
  "$VALUES_DIR/custom-image-resources.yaml" "team: ml-platform"
assert_contains "Extra env var LOG_LEVEL" \
  "$VALUES_DIR/custom-image-resources.yaml" "LOG_LEVEL"
assert_contains "Extra arg zap-log-level" \
  "$VALUES_DIR/custom-image-resources.yaml" "\-\-zap-log-level=5"
assert_contains "Extra arg zap-encoder" \
  "$VALUES_DIR/custom-image-resources.yaml" "\-\-zap-encoder=json"
assert_contains "Custom termination grace period" \
  "$VALUES_DIR/custom-image-resources.yaml" "terminationGracePeriodSeconds: 30"

echo ""

# ── Scenario: Custom webhook port and failure policy ──────────────────────────
echo "── Scenario: custom webhook port and failure policy ──"

assert_contains "Custom webhook container port" \
  "$VALUES_DIR/webhook-custom-port.yaml" "containerPort: 8443"
assert_contains "Custom failurePolicy Fail" \
  "$VALUES_DIR/webhook-custom-port.yaml" "failurePolicy: Fail"

echo ""

# ── Scenario: Leader election disabled ────────────────────────────────────────
echo "── Scenario: leader election disabled ──"

DEPLOY_ARGS=$(render "$VALUES_DIR/no-leader-election.yaml" | awk '/kind: Deployment/,/^---/' | grep -c "leader-elect" || true)
if [ "$DEPLOY_ARGS" -eq 0 ]; then
  PASS=$((PASS + 1))
  printf "  ✓ No --leader-elect arg\n"
else
  FAIL=$((FAIL + 1))
  printf "  ✗ No --leader-elect arg — did not expect leader-elect in Deployment\n"
fi
assert_has_kind "Deployment still created" \
  "$VALUES_DIR/no-leader-election.yaml" "Deployment"

echo ""

# ── Scenario: Image pull secrets ─────────────────────────────────────────────
echo "── Scenario: image pull secrets on ServiceAccount ──"

assert_contains "Image pull secret name on ServiceAccount" \
  "$VALUES_DIR/image-pull-secrets.yaml" "name: my-registry-secret"

echo ""

# ── Scenario: Extra volumes and volumeMounts ─────────────────────────────────
echo "── Scenario: extra volumes and volumeMounts ──"

assert_contains "Extra volume 'config'" \
  "$VALUES_DIR/extra-volumes.yaml" "name: config"
assert_contains "Extra volumeMount /etc/custom-config" \
  "$VALUES_DIR/extra-volumes.yaml" "mountPath: /etc/custom-config"
assert_contains "Webhook certs volume still present" \
  "$VALUES_DIR/extra-volumes.yaml" "name: webhook-certs"
assert_contains "Webhook cert mount still present" \
  "$VALUES_DIR/extra-volumes.yaml" "mountPath: /tmp/k8s-webhook-server/serving-certs"

echo ""

# ── Scenario: Custom metrics address and port ────────────────────────────────
echo "── Scenario: custom metrics bind address and service port ──"

assert_contains "Custom metrics bind address :8080" \
  "$VALUES_DIR/custom-metrics.yaml" "\-\-metrics-bind-address=:8080"
assert_contains "Custom metrics service port 8080" \
  "$VALUES_DIR/custom-metrics.yaml" "port: 8080"

echo ""

# ── Scenario: Custom security context ───────────────────────────────────────
echo "── Scenario: custom pod and container security context ──"

assert_contains "Custom runAsUser 1000" \
  "$VALUES_DIR/custom-security-context.yaml" "runAsUser: 1000"
assert_contains "Custom runAsGroup 1000" \
  "$VALUES_DIR/custom-security-context.yaml" "runAsGroup: 1000"
assert_contains "Custom fsGroup 1000" \
  "$VALUES_DIR/custom-security-context.yaml" "fsGroup: 1000"

echo ""

# ── CRD symlink verification ──────────────────────────────────────────────────
echo "── Scenario: CRDs via symlinks from operator/config/crd/bases ──"

assert_contains "LMCacheEngine CRD present" \
  "$VALUES_DIR/minimal.yaml" "lmcacheengines.lmcache.lmcache.ai"
assert_contains "CacheBlendEngine CRD present" \
  "$VALUES_DIR/minimal.yaml" "cacheblendengines.lmcache.lmcache.ai"
assert_contains "LMCacheCoordinator CRD present" \
  "$VALUES_DIR/minimal.yaml" "lmcachecoordinators.lmcache.lmcache.ai"

echo ""

# ── Namespace / Release.Name propagation ─────────────────────────────────────
echo "── Scenario: Release.Namespace and Release.Name propagation ──"

assert_contains "Release name in ClusterRole" \
  "$VALUES_DIR/minimal.yaml" "name: $RELEASE-manager-role"
assert_contains "Release name in ClusterRoleBinding" \
  "$VALUES_DIR/minimal.yaml" "name: $RELEASE-manager-rolebinding"
assert_contains "Release name in Role" \
  "$VALUES_DIR/minimal.yaml" "name: $RELEASE-leader-election-role"
assert_contains "Release name in RoleBinding" \
  "$VALUES_DIR/minimal.yaml" "name: $RELEASE-leader-election-rolebinding"
assert_contains "Release name in webhook service" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "name: $RELEASE-webhook-service"
assert_contains "Release name in mutating webhook" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "name: $RELEASE-mutating-webhook-configuration"
assert_contains "Release namespace in webhook service" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "namespace: $NAMESPACE"
assert_contains "Release namespace in cert-manager cert" \
  "$VALUES_DIR/no-metrics.yaml" "namespace: $NAMESPACE"
assert_contains "Release name in cert-manager cert DNS" \
  "$VALUES_DIR/no-metrics.yaml" "$RELEASE-webhook-service.$NAMESPACE.svc"
assert_contains "Release namespace in namespaceSelector exclusion" \
  "$VALUES_DIR/webhook-no-certmanager.yaml" "lmcache-operator-system$"

echo ""

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────"
echo "  Tests: $((PASS + FAIL))  Passed: $PASS  Failed: $FAIL"
echo "─────────────────────────────────────────"

if [ "$FAIL" -ne 0 ]; then
  exit 1
fi

exit 0
