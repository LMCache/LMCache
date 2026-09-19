// SPDX-License-Identifier: Apache-2.0

// Package chart tests the public Helm rendering and deployment contract.
package chart

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	. "github.com/onsi/gomega"
	admissionv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"sigs.k8s.io/yaml"
)

const chartPath = "../../charts/lmcache-operator"

func helm(t *testing.T, args ...string) []byte {
	t.Helper()
	binary := os.Getenv("HELM")
	if binary == "" {
		binary = "helm"
	}
	output, err := exec.Command(binary, args...).CombinedOutput()
	NewWithT(t).Expect(err).NotTo(HaveOccurred(), "helm %v: %s", args, output)
	return output
}

func render(t *testing.T, release, namespace string, args ...string) []unstructured.Unstructured {
	t.Helper()
	command := append([]string{"template", release, chartPath, "--namespace", namespace}, args...)
	decoder := utilyaml.NewYAMLOrJSONDecoder(bytes.NewReader(helm(t, command...)), 4096)
	var objects []unstructured.Unstructured
	for {
		var object unstructured.Unstructured
		err := decoder.Decode(&object)
		if err == io.EOF {
			return objects
		}
		NewWithT(t).Expect(err).NotTo(HaveOccurred())
		if object.GetKind() != "" {
			objects = append(objects, object)
		}
	}
}

func objectOf(t *testing.T, objects []unstructured.Unstructured, kind, name string) unstructured.Unstructured {
	t.Helper()
	for _, object := range objects {
		if object.GetKind() == kind && object.GetName() == name {
			return object
		}
	}
	t.Fatalf("missing %s %s", kind, name)
	return unstructured.Unstructured{}
}

func decode(t *testing.T, object unstructured.Unstructured, target any) {
	t.Helper()
	NewWithT(t).Expect(runtime.DefaultUnstructuredConverter.FromUnstructured(object.Object, target)).To(Succeed())
}

func TestGeneratedAPIsAndPermissions(t *testing.T) {
	g := NewWithT(t)
	objects := render(t, "lmcache-operator", "lmcache-operator-system")
	crds, err := filepath.Glob(filepath.Join(chartPath, "files", "crds", "*.yaml"))
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(crds).To(HaveLen(3))
	for _, path := range crds {
		source, err := os.ReadFile(path)
		g.Expect(err).NotTo(HaveOccurred())
		var expected, actual apiextensionsv1.CustomResourceDefinition
		g.Expect(yaml.Unmarshal(source, &expected)).To(Succeed())
		decode(t, objectOf(t, objects, "CustomResourceDefinition", expected.Name), &actual)
		g.Expect(actual.Spec).To(Equal(expected.Spec), "chart must ship the generated API schema")
		g.Expect(actual.Annotations).To(HaveKeyWithValue("helm.sh/resource-policy", "keep"))
	}
	var expected, actual rbacv1.ClusterRole
	source, err := os.ReadFile(filepath.Join(chartPath, "files", "role.yaml"))
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(yaml.Unmarshal(source, &expected)).To(Succeed())
	decode(t, objectOf(t, objects, "ClusterRole", "lmcache-operator-manager-role"), &actual)
	g.Expect(actual.Rules).To(Equal(expected.Rules))
	for _, object := range objects {
		// Helm uninstall must not own the namespace or user-managed instances.
		g.Expect(object.GetKind()).NotTo(BeElementOf("Namespace", "LMCacheEngine", "CacheBlendEngine", "LMCacheCoordinator"))
	}
}

func TestNamespaceAndWebhookWiring(t *testing.T) {
	g := NewWithT(t)
	objects := render(t, "cache-control", "cache-system")
	var deployment appsv1.Deployment
	decode(t, objectOf(t, objects, "Deployment", "cache-control-controller-manager"), &deployment)
	g.Expect(deployment.Namespace).To(Equal("cache-system"))
	g.Expect(deployment.Spec.Template.Spec.ServiceAccountName).To(Equal("cache-control-controller-manager"))
	g.Expect(deployment.Spec.Template.Spec.Containers[0].Args).To(ContainElement("--leader-elect"))
	var service corev1.Service
	decode(t, objectOf(t, objects, "Service", "cache-control-webhook-service"), &service)
	for key, value := range service.Spec.Selector {
		g.Expect(deployment.Spec.Template.Labels).To(HaveKeyWithValue(key, value))
	}
	certificate := objectOf(t, objects, "Certificate", "cache-control-serving-cert")
	names, found, err := unstructured.NestedStringSlice(certificate.Object, "spec", "dnsNames")
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(found).To(BeTrue())
	g.Expect(names).To(ContainElement("cache-control-webhook-service.cache-system.svc"))
	secret, _, err := unstructured.NestedString(certificate.Object, "spec", "secretName")
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(deployment.Spec.Template.Spec.Volumes[0].Secret.SecretName).To(Equal(secret))
	var webhooks admissionv1.MutatingWebhookConfiguration
	webhookObject := objectOf(t, objects, "MutatingWebhookConfiguration", "cache-control-mutating-webhook-configuration")
	decode(t, webhookObject, &webhooks)
	g.Expect(webhooks.Annotations).To(HaveKeyWithValue(
		"cert-manager.io/inject-ca-from", "cache-system/cache-control-serving-cert"))
	g.Expect(webhooks.Webhooks).To(HaveLen(2))
	for _, webhook := range webhooks.Webhooks {
		g.Expect(webhook.ClientConfig.Service.Name).To(Equal(service.Name))
		g.Expect(webhook.ClientConfig.Service.Namespace).To(Equal(service.Namespace))
		g.Expect(webhook.ObjectSelector.MatchLabels).To(HaveLen(1))
		g.Expect(webhook.NamespaceSelector.MatchExpressions[0].Values).To(ContainElement("cache-system"))
	}
	for _, object := range objects {
		if object.GetKind() != "RoleBinding" && object.GetKind() != "ClusterRoleBinding" {
			continue
		}
		subjects, _, err := unstructured.NestedSlice(object.Object, "subjects")
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(subjects).To(HaveLen(1))
		g.Expect(subjects[0]).To(HaveKeyWithValue("namespace", "cache-system"))
		g.Expect(subjects[0]).To(HaveKeyWithValue("name", deployment.Spec.Template.Spec.ServiceAccountName))
	}
}

func TestOperatorConfiguration(t *testing.T) {
	g := NewWithT(t)
	values := filepath.Join(t.TempDir(), "values.yaml")
	g.Expect(os.WriteFile(values, []byte(`image:
  repository: registry.example:5000/lmcache/operator
  tag: custom
  pullPolicy: Always
replicaCount: 2
imagePullSecrets:
  - name: registry-auth
podAnnotations:
  example.com/team: serving
nodeSelector:
  pool: control
tolerations:
  - key: dedicated
    operator: Exists
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: pool
              operator: In
              values: [control]
resources:
  requests:
    memory: 96Mi
metrics:
  serviceMonitor:
    enabled: true
    interval: 45s
    labels:
      prometheus: main
`), 0o600)).To(Succeed())
	objects := render(t, "lmcache-operator", "lmcache-operator-system", "-f", values)
	var deployment appsv1.Deployment
	decode(t, objectOf(t, objects, "Deployment", "lmcache-operator-controller-manager"), &deployment)
	pod := deployment.Spec.Template
	g.Expect(*deployment.Spec.Replicas).To(Equal(int32(2)))
	g.Expect(pod.Spec.Containers[0].Image).To(Equal("registry.example:5000/lmcache/operator:custom"))
	g.Expect(pod.Spec.Containers[0].ImagePullPolicy).To(Equal(corev1.PullAlways))
	g.Expect(pod.Spec.Containers[0].Resources.Requests.Memory().String()).To(Equal("96Mi"))
	g.Expect(pod.Spec.ImagePullSecrets).To(Equal([]corev1.LocalObjectReference{{Name: "registry-auth"}}))
	g.Expect(pod.Annotations).To(HaveKeyWithValue("example.com/team", "serving"))
	g.Expect(pod.Spec.NodeSelector).To(HaveKeyWithValue("pool", "control"))
	g.Expect(pod.Spec.Tolerations).To(HaveLen(1))
	terms := pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
	g.Expect(terms).To(HaveLen(1))
	monitor := objectOf(t, objects, "ServiceMonitor", "lmcache-operator-controller-manager-metrics-monitor")
	g.Expect(monitor.GetLabels()).To(HaveKeyWithValue("prometheus", "main"))
}

func TestLongReleaseNames(t *testing.T) {
	g := NewWithT(t)
	objects := render(t, strings.Repeat("a", 53), "operators", "--set", "metrics.serviceMonitor.enabled=true")
	identities := make(map[string]bool, len(objects))
	for _, object := range objects {
		identity := object.GetKind() + "/" + object.GetNamespace() + "/" + object.GetName()
		g.Expect(identities).NotTo(HaveKey(identity), "each rendered object needs a distinct identity")
		identities[identity] = true
		if object.GetKind() != "CustomResourceDefinition" {
			g.Expect(len(object.GetName())).To(BeNumerically("<=", 63))
		}
	}
}
