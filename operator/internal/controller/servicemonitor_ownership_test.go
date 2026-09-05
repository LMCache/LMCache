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
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

const (
	monitorOwnerEngine      = "LMCacheEngine"
	monitorOwnerBlend       = "CacheBlendEngine"
	monitorOwnerCoordinator = "LMCacheCoordinator"
)

var _ = Describe("ServiceMonitor ownership when monitoring is disabled", func() {
	for _, kind := range []string{monitorOwnerEngine, monitorOwnerBlend, monitorOwnerCoordinator} {
		Context(kind, func() {
			DescribeTable("only deletes monitors controlled by the reconciled resource",
				func(ownership string, deleted bool) {
					ctx := context.Background()
					scheme := runtime.NewScheme()
					Expect(clientgoscheme.AddToScheme(scheme)).To(Succeed())
					Expect(lmcachev1alpha1.AddToScheme(scheme)).To(Succeed())
					Expect(monitoringv1.AddToScheme(scheme)).To(Succeed())

					meta := metav1.ObjectMeta{
						Name: "cache", Namespace: "default", UID: types.UID("current-owner"),
					}
					var owner client.Object
					switch kind {
					case monitorOwnerEngine:
						owner = &lmcachev1alpha1.LMCacheEngine{
							ObjectMeta: meta,
							Spec: lmcachev1alpha1.LMCacheEngineSpec{
								L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 1},
							},
						}
					case monitorOwnerBlend:
						owner = &lmcachev1alpha1.CacheBlendEngine{
							ObjectMeta: meta,
							Spec: lmcachev1alpha1.CacheBlendEngineSpec{
								L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 1},
								Injection: &lmcachev1alpha1.InjectionSpec{
									PayloadImage: &lmcachev1alpha1.ImageSpec{Repository: ptr.To("example/plugin")},
								},
							},
						}
					case monitorOwnerCoordinator:
						owner = &lmcachev1alpha1.LMCacheCoordinator{ObjectMeta: meta}
					}

					monitor := &monitoringv1.ServiceMonitor{
						ObjectMeta: metav1.ObjectMeta{
							Name: meta.Name, Namespace: meta.Namespace,
							Labels:      map[string]string{"managed-by": "user"},
							Annotations: map[string]string{"description": "custom monitor"},
						},
						Spec: monitoringv1.ServiceMonitorSpec{
							Selector:  metav1.LabelSelector{MatchLabels: map[string]string{"app": "custom"}},
							Endpoints: []monitoringv1.Endpoint{{Port: "metrics", Path: "/custom-metrics"}},
						},
					}
					if ownership != "unowned" {
						ref := metav1.OwnerReference{
							APIVersion: lmcachev1alpha1.GroupVersion.String(),
							Kind:       kind, Name: meta.Name, UID: meta.UID, Controller: ptr.To(true),
						}
						switch ownership {
						case "another resource":
							ref.Kind = monitorOwnerCoordinator
							if kind == monitorOwnerCoordinator {
								ref.Kind = monitorOwnerEngine
							}
							ref.UID = "other-owner"
						case "previous incarnation":
							ref.UID = "previous-owner"
						case "non-controller reference":
							ref.Controller = ptr.To(false)
						case "unspecified controller reference":
							ref.Controller = nil
						}
						monitor.OwnerReferences = []metav1.OwnerReference{ref}
					}
					original := monitor.DeepCopy()
					k8s := fake.NewClientBuilder().WithScheme(scheme).
						WithStatusSubresource(owner).WithObjects(owner, monitor).Build()
					var reconciler reconcile.Reconciler
					switch kind {
					case monitorOwnerEngine:
						reconciler = &LMCacheEngineReconciler{Client: k8s, Scheme: scheme}
					case monitorOwnerBlend:
						reconciler = &CacheBlendEngineReconciler{Client: k8s, Scheme: scheme}
					case monitorOwnerCoordinator:
						reconciler = &LMCacheCoordinatorReconciler{Client: k8s, Scheme: scheme}
					}

					// Reconcile twice to cover both initial setup and steady-state cleanup.
					for range 2 {
						_, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(owner)})
						Expect(err).NotTo(HaveOccurred())
						got := &monitoringv1.ServiceMonitor{}
						err = k8s.Get(ctx, client.ObjectKeyFromObject(monitor), got)
						if deleted {
							Expect(apierrors.IsNotFound(err)).To(BeTrue())
						} else {
							Expect(err).NotTo(HaveOccurred())
							Expect(got.OwnerReferences).To(Equal(original.OwnerReferences))
							Expect(got.Spec).To(Equal(original.Spec))
							Expect(got.Labels).To(Equal(original.Labels))
							Expect(got.Annotations).To(Equal(original.Annotations))
						}
					}
				},
				Entry("owned monitor", "current resource", true),
				Entry("user-managed monitor", "unowned", false),
				Entry("monitor owned by another resource with the same name", "another resource", false),
				Entry("monitor owned by an earlier resource with the same name", "previous incarnation", false),
				Entry("non-controller owner reference", "non-controller reference", false),
				Entry("owner reference without a controller flag", "unspecified controller reference", false),
			)
		})
	}
})
