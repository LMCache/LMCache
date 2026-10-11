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

package controller

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

var _ = DescribeTable("Pod labels preserve workload selectors", func(kind string, existing bool) {
	const resourceName = "pod-labels"
	namespace := mustCreateNS(uniqueNS("pod-labels"))
	key := types.NamespacedName{Name: resourceName, Namespace: namespace}
	metadata := metav1.ObjectMeta{Name: key.Name, Namespace: key.Namespace}
	podLabels := map[string]string{
		"example.com/team":            "inference",
		"app.kubernetes.io/component": "custom-component",
	}
	if !existing {
		podLabels["app.kubernetes.io/name"] = "custom-name"
		podLabels["app.kubernetes.io/instance"] = "custom-instance"
		podLabels["app.kubernetes.io/managed-by"] = "custom-manager"
	}

	var (
		resource  client.Object
		workload  client.Object
		r         reconcile.Reconciler
		setLabels func(map[string]string)
	)
	switch kind {
	case "LMCacheEngine":
		engine := &lmcachev1alpha1.LMCacheEngine{
			ObjectMeta: metadata,
			Spec: lmcachev1alpha1.LMCacheEngineSpec{
				L1:        lmcachev1alpha1.L1BackendSpec{SizeGB: 1},
				PodLabels: podLabels,
			},
		}
		resource = engine
		workload = resources.BuildDaemonSet(engine)
		r = &LMCacheEngineReconciler{Client: k8sClient, Scheme: k8sClient.Scheme()}
		setLabels = func(value map[string]string) { engine.Spec.PodLabels = value }
	case "CacheBlendEngine":
		payloadRepository := "lmcache/cacheblend-plugin"
		engine := &lmcachev1alpha1.CacheBlendEngine{
			ObjectMeta: metadata,
			Spec: lmcachev1alpha1.CacheBlendEngineSpec{
				L1:        lmcachev1alpha1.L1BackendSpec{SizeGB: 1},
				PodLabels: podLabels,
				Injection: &lmcachev1alpha1.InjectionSpec{
					PayloadImage: &lmcachev1alpha1.ImageSpec{Repository: &payloadRepository},
				},
			},
		}
		resource = engine
		workload = resources.BuildCBEngineDaemonSet(engine)
		r = &CacheBlendEngineReconciler{Client: k8sClient, Scheme: k8sClient.Scheme()}
		setLabels = func(value map[string]string) { engine.Spec.PodLabels = value }
	case "LMCacheCoordinator":
		coordinator := &lmcachev1alpha1.LMCacheCoordinator{
			ObjectMeta: metadata,
			Spec:       lmcachev1alpha1.LMCacheCoordinatorSpec{PodLabels: podLabels},
		}
		resource = coordinator
		workload = resources.BuildCoordinatorDeployment(coordinator)
		r = &LMCacheCoordinatorReconciler{Client: k8sClient, Scheme: k8sClient.Scheme()}
		setLabels = func(value map[string]string) { coordinator.Spec.PodLabels = value }
	}
	Expect(k8sClient.Create(ctx, resource)).To(Succeed())
	DeferCleanup(func() {
		Expect(k8sClient.Get(ctx, key, resource)).To(Succeed())
		resource.SetFinalizers(nil)
		Expect(k8sClient.Update(ctx, resource)).To(Succeed())
		Expect(k8sClient.Delete(ctx, resource)).To(Succeed())
		for _, object := range []client.Object{&appsv1.DaemonSet{}, &appsv1.Deployment{}, &corev1.Service{}, &corev1.ConfigMap{}} {
			Expect(k8sClient.DeleteAllOf(ctx, object, client.InNamespace(namespace))).To(Succeed())
		}
	})

	if existing {
		By("creating a workload with an additional immutable selector label")
		selector, template := podLabelWorkloadFields(workload)
		selector.MatchLabels["example.com/legacy-selector"] = "original"
		template.Labels["example.com/legacy-selector"] = "original"
		Expect(k8sClient.Create(ctx, workload)).To(Succeed())
		podLabels["example.com/legacy-selector"] = "overridden"
		Expect(k8sClient.Get(ctx, key, resource)).To(Succeed())
		setLabels(podLabels)
		Expect(k8sClient.Update(ctx, resource)).To(Succeed())
	}
	selector, _ := podLabelWorkloadFields(workload)
	expectedSelector := selector.DeepCopy()

	for _, team := range []string{"inference", "updated"} {
		if team == "updated" {
			By("updating custom labels without changing the workload selector")
			podLabels["example.com/team"] = team
			Expect(k8sClient.Get(ctx, key, resource)).To(Succeed())
			setLabels(podLabels)
			Expect(k8sClient.Update(ctx, resource)).To(Succeed())
		}
		_, err := r.Reconcile(ctx, reconcile.Request{NamespacedName: key})
		Expect(err).NotTo(HaveOccurred())
		// Repeated reconciliation must preserve the same labels.
		_, err = r.Reconcile(ctx, reconcile.Request{NamespacedName: key})
		Expect(err).NotTo(HaveOccurred())
		Expect(k8sClient.Get(ctx, key, workload)).To(Succeed())
		selector, template := podLabelWorkloadFields(workload)
		Expect(selector).To(Equal(expectedSelector))
		Expect(labels.SelectorFromSet(selector.MatchLabels).Matches(labels.Set(template.Labels))).To(BeTrue())
		Expect(template.Labels).To(HaveKeyWithValue("example.com/team", team))
		Expect(template.Labels).To(HaveKeyWithValue("app.kubernetes.io/component", "custom-component"))
		if existing {
			Expect(selector.MatchLabels).To(HaveKeyWithValue("example.com/legacy-selector", "original"))
			Expect(template.Labels).To(HaveKeyWithValue("example.com/legacy-selector", "original"))
		}

		service := &corev1.Service{}
		Expect(k8sClient.Get(ctx, key, service)).To(Succeed())
		Expect(labels.SelectorFromSet(service.Spec.Selector).Matches(labels.Set(template.Labels))).To(BeTrue())
	}
},
	Entry("creates and updates LMCacheEngine pods", "LMCacheEngine", false),
	Entry("creates and updates CacheBlendEngine pods", "CacheBlendEngine", false),
	Entry("creates and updates LMCacheCoordinator pods", "LMCacheCoordinator", false),
	Entry("preserves an existing LMCacheEngine selector", "LMCacheEngine", true),
	Entry("preserves an existing CacheBlendEngine selector", "CacheBlendEngine", true),
	Entry("preserves an existing LMCacheCoordinator selector", "LMCacheCoordinator", true),
)

// podLabelWorkloadFields returns the selector and pod template of a workload.
func podLabelWorkloadFields(workload client.Object) (*metav1.LabelSelector, *corev1.PodTemplateSpec) {
	switch object := workload.(type) {
	case *appsv1.DaemonSet:
		return object.Spec.Selector, &object.Spec.Template
	case *appsv1.Deployment:
		return object.Spec.Selector, &object.Spec.Template
	default:
		panic("unsupported test workload")
	}
}
