//go:build e2e

/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package e2e

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	admissionv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/test/utils"
)

// This spec temporarily stops reconciliation cluster-wide, so it runs alone
// and only on a dedicated Kind cluster created by the smoke test targets.
var _ = Describe("Operator Helm lifecycle (no-GPU)", Serial, func() {
	const namespace = "lmcache-operator-system"

	BeforeEach(func() {
		if skipImageLoad || os.Getenv("KIND_CLUSTER") == "" {
			Skip("Helm release lifecycle requires the dedicated Kind smoke cluster")
		}
	})

	AfterEach(func() {
		recordOnFailure(namespace)
	})

	It("upgrades CRDs, retains cache resources, reinstalls, and adopts a YAML installation", func() {
		ctx := context.Background()
		lmc, err := utils.NewLMCFromFixture("lmc_minimal.yaml", namespace, "helm-retained-cache")
		Expect(err).NotTo(HaveOccurred())
		// No GPU workload is needed to verify release ownership and reconciliation.
		lmc.Spec.NodeSelector = map[string]string{"helm-lifecycle-test": "unscheduled"}
		key := client.ObjectKeyFromObject(lmc)
		restoreOperator := false
		var restoreDeployArgs []string
		DeferCleanup(func() {
			if restoreOperator {
				deployHelmOperator(restoreDeployArgs...)
			}
			Expect(utils.DeleteLMCAndWaitGC(ctx, k8sClient, key, time.Minute)).To(Succeed())
		})

		By("creating a cache in the operator namespace")
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, time.Minute)).To(Succeed())
		retained := captureHelmRetainedObjects(ctx, key)

		By("upgrading the release with a changed CRD schema and operator pod annotation")
		chartPath, originalDescription, upgradedDescription := chartWithUpgradedSchema()
		assertHelmCRDSchemaDescription(ctx, originalDescription)
		restoreOperator = true
		_, err = utils.RunMake("deploy", fmt.Sprintf("IMG=%s", managerImage),
			"CHART="+chartPath,
			"HELM_EXTRA_ARGS=--set-string podAnnotations.helm-lifecycle=upgraded")
		Expect(err).NotTo(HaveOccurred())
		waitHelmOperatorRollout()
		assertHelmCRDSchemaDescription(ctx, upgradedDescription)
		deployment := &appsv1.Deployment{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{
			Namespace: namespace, Name: "lmcache-operator-controller-manager",
		}, deployment)).To(Succeed())
		Expect(deployment.Spec.Template.Annotations).To(HaveKeyWithValue("helm-lifecycle", "upgraded"))
		assertHelmObjectsRetained(ctx, retained)
		operatorObjects := helmOperatorObjects(namespace)
		for _, object := range operatorObjects {
			Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(object), object)).To(Succeed())
		}

		By("uninstalling the operator release")
		_, err = utils.RunMake("undeploy")
		Expect(err).NotTo(HaveOccurred())
		assertHelmOperatorRemoved(ctx, operatorObjects)

		By("retaining the namespace, CRDs, cache CR, and owned cache resources")
		assertHelmObjectsRetained(ctx, retained)
		assertHelmCRDSchemaDescription(ctx, upgradedDescription)

		By("changing the cache while the operator is absent")
		Expect(utils.PatchLMCSpec(ctx, k8sClient, key, func(spec *lmcachev1alpha1.LMCacheEngineSpec) {
			if spec.Server == nil {
				spec.Server = &lmcachev1alpha1.ServerSpec{}
			}
			port := int32(6555)
			spec.Server.Port = &port
		})).To(Succeed())

		By("reinstalling the release and reconciling the retained cache")
		deployHelmOperator()
		restoreOperator = false
		assertHelmCRDSchemaDescription(ctx, originalDescription)
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, time.Minute)).To(Succeed())
		Eventually(func(g Gomega) {
			cfg, err := utils.GetConnectionConfig(ctx, k8sClient, key)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(cfg.KVConnectorExtraConfig.Port).To(Equal("6555"))
		}, time.Minute, time.Second).Should(Succeed())
		assertHelmObjectsRetained(ctx, retained)

		By("installing the standalone YAML with externally managed CRDs")
		restoreOperator = true
		restoreDeployArgs = []string{"HELM_EXTRA_ARGS=--take-ownership"}
		_, err = utils.RunMake("undeploy")
		Expect(err).NotTo(HaveOccurred())
		assertHelmOperatorRemoved(ctx, operatorObjects)
		adoptedObjects := helmOperatorObjects(namespace)
		for _, retainedObject := range retained {
			crd, ok := retainedObject.object.(*apiextensionsv1.CustomResourceDefinition)
			if !ok {
				continue
			}
			Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(crd), crd)).To(Succeed())
			patch := client.MergeFrom(crd.DeepCopy())
			delete(crd.Annotations, "meta.helm.sh/release-name")
			delete(crd.Annotations, "meta.helm.sh/release-namespace")
			Expect(k8sClient.Patch(ctx, crd, patch)).To(Succeed())
			adoptedObjects = append(adoptedObjects, crd)
		}
		_, err = utils.RunMake("build-installer", fmt.Sprintf("IMG=%s", managerImage))
		Expect(err).NotTo(HaveOccurred())
		_, err = utils.RunFromOperator(exec.Command("kubectl", "apply", "-f", "dist/install.yaml"))
		Expect(err).NotTo(HaveOccurred())
		waitHelmOperatorRollout()
		assertHelmObjectsRetained(ctx, retained)

		By("adopting the YAML installation into Helm without replacing cache resources")
		deployHelmOperator("HELM_EXTRA_ARGS=--take-ownership")
		restoreOperator = false
		assertHelmObjectsRetained(ctx, retained)
		for _, object := range adoptedObjects {
			Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(object), object)).To(Succeed())
			Expect(object.GetAnnotations()).To(HaveKeyWithValue("meta.helm.sh/release-name", "lmcache-operator"))
			Expect(object.GetAnnotations()).To(HaveKeyWithValue("meta.helm.sh/release-namespace", namespace))
		}
	})
})

type helmRetainedObject struct {
	object client.Object
	uid    types.UID
}

// chartWithUpgradedSchema copies the installable chart and changes a harmless
// schema description so the live API can prove that Helm upgraded the CRD.
func chartWithUpgradedSchema() (chartPath, originalDescription, upgradedDescription string) {
	GinkgoHelper()
	chartPath, err := os.MkdirTemp("", "lmcache-operator-chart-upgrade-")
	Expect(err).NotTo(HaveOccurred())
	DeferCleanup(func() {
		Expect(os.RemoveAll(chartPath)).To(Succeed())
	})
	source := filepath.Join(utils.OperatorRoot(), "charts", "lmcache-operator")
	Expect(os.CopyFS(chartPath, os.DirFS(source))).To(Succeed())

	crdPath := filepath.Join(chartPath, "files", "crds", "lmcache.lmcache.ai_lmcacheengines.yaml")
	data, err := os.ReadFile(crdPath)
	Expect(err).NotTo(HaveOccurred())
	crd := &apiextensionsv1.CustomResourceDefinition{}
	Expect(yaml.Unmarshal(data, crd)).To(Succeed())
	Expect(crd.Spec.Versions).NotTo(BeEmpty())
	Expect(crd.Spec.Versions[0].Schema).NotTo(BeNil())
	Expect(crd.Spec.Versions[0].Schema.OpenAPIV3Schema).NotTo(BeNil())
	crdSchema := crd.Spec.Versions[0].Schema.OpenAPIV3Schema
	originalDescription = crdSchema.Description
	upgradedDescription = originalDescription + " Helm lifecycle schema upgrade probe."
	crdSchema.Description = upgradedDescription
	data, err = yaml.Marshal(crd)
	Expect(err).NotTo(HaveOccurred())
	Expect(os.WriteFile(crdPath, data, 0o600)).To(Succeed())
	return chartPath, originalDescription, upgradedDescription
}

func assertHelmCRDSchemaDescription(ctx context.Context, expected string) {
	GinkgoHelper()
	crd := &apiextensionsv1.CustomResourceDefinition{}
	Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "lmcacheengines.lmcache.lmcache.ai"}, crd)).To(Succeed())
	Expect(crd.Spec.Versions).NotTo(BeEmpty())
	Expect(crd.Spec.Versions[0].Schema).NotTo(BeNil())
	Expect(crd.Spec.Versions[0].Schema.OpenAPIV3Schema).NotTo(BeNil())
	Expect(crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Description).To(Equal(expected))
}

func captureHelmRetainedObjects(ctx context.Context, key types.NamespacedName) []helmRetainedObject {
	GinkgoHelper()
	meta := metav1.ObjectMeta{Namespace: key.Namespace, Name: key.Name}
	objects := make([]client.Object, 0, 8)
	objects = append(objects,
		&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: key.Namespace}},
		&lmcachev1alpha1.LMCacheEngine{ObjectMeta: meta},
		&appsv1.DaemonSet{ObjectMeta: meta},
		&corev1.Service{ObjectMeta: meta},
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: key.Namespace, Name: key.Name + "-connection"}},
	)
	for _, name := range []string{
		"lmcacheengines.lmcache.lmcache.ai",
		"cacheblendengines.lmcache.lmcache.ai",
		"lmcachecoordinators.lmcache.lmcache.ai",
	} {
		objects = append(objects, &apiextensionsv1.CustomResourceDefinition{ObjectMeta: metav1.ObjectMeta{Name: name}})
	}
	retained := make([]helmRetainedObject, 0, len(objects))
	for _, object := range objects {
		Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(object), object)).To(Succeed())
		retained = append(retained, helmRetainedObject{object: object, uid: object.GetUID()})
	}
	return retained
}

func assertHelmObjectsRetained(ctx context.Context, objects []helmRetainedObject) {
	GinkgoHelper()
	for _, retained := range objects {
		key := client.ObjectKeyFromObject(retained.object)
		Expect(k8sClient.Get(ctx, key, retained.object)).To(Succeed(), "retain %T %s", retained.object, key)
		Expect(retained.object.GetUID()).To(Equal(retained.uid), "preserve %T %s identity", retained.object, key)
		Expect(retained.object.GetDeletionTimestamp()).To(BeNil(), "retain %T %s without deletion", retained.object, key)
	}
}

func helmOperatorObjects(namespace string) []client.Object {
	manager := metav1.ObjectMeta{Namespace: namespace, Name: "lmcache-operator-controller-manager"}
	return []client.Object{
		&appsv1.Deployment{ObjectMeta: manager},
		&corev1.ServiceAccount{ObjectMeta: manager},
		&corev1.Service{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "lmcache-operator-webhook-service"}},
		&rbacv1.ClusterRole{ObjectMeta: metav1.ObjectMeta{Name: "lmcache-operator-manager-role"}},
		&rbacv1.ClusterRoleBinding{ObjectMeta: metav1.ObjectMeta{Name: "lmcache-operator-manager-rolebinding"}},
		&admissionv1.MutatingWebhookConfiguration{ObjectMeta: metav1.ObjectMeta{
			Name: "lmcache-operator-mutating-webhook-configuration",
		}},
	}
}

func assertHelmOperatorRemoved(ctx context.Context, objects []client.Object) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		for _, object := range objects {
			key := client.ObjectKeyFromObject(object)
			g.Expect(apierrors.IsNotFound(k8sClient.Get(ctx, key, object))).To(BeTrue(), "remove %T %s", object, key)
		}
		pods := &corev1.PodList{}
		g.Expect(k8sClient.List(ctx, pods, client.InNamespace("lmcache-operator-system"),
			client.MatchingLabels{"control-plane": "controller-manager"})).To(Succeed())
		g.Expect(pods.Items).To(BeEmpty(), "operator pods must stop before exercising offline CR changes")
	}, time.Minute, time.Second).Should(Succeed())
}

func deployHelmOperator(extraArgs ...string) {
	GinkgoHelper()
	args := append([]string{"deploy", fmt.Sprintf("IMG=%s", managerImage)}, extraArgs...)
	_, err := utils.RunMake(args...)
	Expect(err).NotTo(HaveOccurred(), "Failed to reinstall the operator Helm release")
	waitHelmOperatorRollout()
}

func waitHelmOperatorRollout() {
	GinkgoHelper()
	_, err := utils.RunFromOperator(exec.Command("kubectl", "rollout", "status",
		"deployment/lmcache-operator-controller-manager", "-n", "lmcache-operator-system", "--timeout=180s"))
	Expect(err).NotTo(HaveOccurred(), "Operator rollout did not complete")
}
