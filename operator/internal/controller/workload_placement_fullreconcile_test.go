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
	"testing"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

// wlasReconcileScheme builds a full test scheme for driving Reconcile
// directly against the fake client: the real reconcile handles DaemonSets,
// Services, ConfigMaps, Secrets and ServiceMonitors, as well as the engine
// CR and Node/Pod objects, and each must be registered in the Scheme.
func wlasReconcileScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	sc := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(sc); err != nil {
		t.Fatal(err)
	}
	if err := monitoringv1.AddToScheme(sc); err != nil {
		t.Fatal(err)
	}
	if err := lmcachev1alpha1.AddToScheme(sc); err != nil {
		t.Fatal(err)
	}
	return sc
}

func wlasReconcileClient(t *testing.T, objs ...client.Object) client.Client {
	t.Helper()
	return fake.NewClientBuilder().
		WithScheme(wlasReconcileScheme(t)).
		WithObjects(objs...).
		WithStatusSubresource(&lmcachev1alpha1.LMCacheEngine{}).
		Build()
}

// TestDisableOrdering_ReconcilePurgesOnlyAfterSelectorIsClean is the
// full-Reconcile ordering guard the reviewer asked for: enable → label +
// selector on the DS, then disable → ONE Reconcile must reconcile the
// label-free DaemonSet selector BEFORE purging node labels, so the DS
// controller never sees a selector referencing a label no node carries
// (which would delete every engine pod and drop every warm KV cache).
func TestDisableOrdering_ReconcilePurgesOnlyAfterSelectorIsClean(t *testing.T) {
	ctx := context.Background()
	engine := newWlasEngine("engine", 300)
	pod := boundPod(wlasTestNamespace, "p1", "gpu-a", map[string]string{"app": "predictor"})
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), pod)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "engine")
	req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: wlasTestNamespace, Name: "engine"}}

	// --- Enabled Reconcile: label applied, DS selector carries the key.
	if _, err := r.Reconcile(ctx, req); err != nil {
		t.Fatalf("enabled reconcile: %v", err)
	}
	ds := &appsv1.DaemonSet{}
	if err := c.Get(ctx, req.NamespacedName, ds); err != nil {
		t.Fatal(err)
	}
	if ds.Spec.Template.Spec.NodeSelector[key] != resources.WorkloadLabelValueBase {
		t.Fatalf("DS selector must reference the managed label while enabled")
	}
	node := &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-a"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != resources.WorkloadLabelValueBase {
		t.Fatalf("node label missing after enabled reconcile")
	}

	// --- Disable.
	disableEngine(t, ctx, c, engine)
	if _, err := r.Reconcile(ctx, req); err != nil {
		t.Fatalf("disable reconcile: %v", err)
	}

	// Post-conditions of the ONE disable reconcile: the live DS selector no
	// longer references the key …
	if err := c.Get(ctx, req.NamespacedName, ds); err != nil {
		t.Fatal(err)
	}
	if ds.Spec.Template.Spec.NodeSelector[key] != "" {
		t.Fatalf("DS selector still references the managed label after disable reconcile")
	}
	// … AND the labels were happily purged. The ordering contract is
	// enforced by purgeWorkloadPlacement's live re-read (tested below) —
	// this test asserts the happy-path final state.
	node = &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-a"}, node); err != nil {
		t.Fatal(err)
	}
	if _, still := node.Labels[key]; still {
		t.Fatalf("labels must be purged after the selector went label-free")
	}
	// Finalizer dropped → CR deletion can proceed without stripping.
	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, req.NamespacedName, e); err != nil {
		t.Fatal(err)
	}
	for _, f := range e.Finalizers {
		if f == workloadPlacementFinalizer {
			t.Fatal("finalizer must be removed on the disabled-path purge")
		}
	}
}

// TestDisableOrdering_GuardRefusesWhileSelectorReferencesThem targets the
// enforcement mechanism itself: with the live DS still selecting on the key
// (e.g. reconcileDaemonSet's update conflict blocked, or a watch race), the
// cleanup must REFUSE — no purge, no error, requeue — rather than strip
// labels the selector still depends on.
func TestDisableOrdering_GuardRefusesWhileSelectorReferencesThem(t *testing.T) {
	ctx := context.Background()
	engine := newWlasEngine("engine", 300)
	pod := boundPod(wlasTestNamespace, "p1", "gpu-a", map[string]string{"app": "predictor"})
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), pod)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "engine")
	req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: wlasTestNamespace, Name: "engine"}}

	if _, err := r.Reconcile(ctx, req); err != nil {
		t.Fatal(err)
	}
	disableEngine(t, ctx, c, engine)

	// SABOTAGE: force the live DS to keep referencing the managed key,
	// simulating a reconcileDaemonSet update that did not land.
	ds := &appsv1.DaemonSet{}
	if err := c.Get(ctx, req.NamespacedName, ds); err != nil {
		t.Fatal(err)
	}
	patch := client.MergeFrom(ds.DeepCopy())
	if ds.Spec.Template.Spec.NodeSelector == nil {
		ds.Spec.Template.Spec.NodeSelector = map[string]string{}
	}
	ds.Spec.Template.Spec.NodeSelector[key] = resources.WorkloadLabelValueBase
	if err := c.Patch(ctx, ds, patch); err != nil {
		t.Fatal(err)
	}

	fresh := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, req.NamespacedName, fresh); err != nil {
		t.Fatal(err)
	}
	done, err := r.purgeWorkloadPlacement(ctx, fresh)
	if err != nil {
		t.Fatalf("guard must not error on refusal: %v", err)
	}
	if done {
		t.Fatal("guard must refuse while the live selector references the label")
	}
	node := &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-a"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != resources.WorkloadLabelValueBase {
		t.Fatal("refusing guard must not purge labels")
	}
}

// TestFullReconcile_PlacementStatusSurvivesUpdateStatus closes the
// reviewer's "status wipe" concern at the integration level: NodeAbsentSince/
// WorkloadNodes/WorkloadLabelKey written by the placement step must survive
// updateStatus's re-fetch + second Status().Update in the SAME Reconcile and
// remain persisted on a subsequent Reconcile — or release windows would
// restart on every pass and never expire.
func TestFullReconcile_PlacementStatusSurvivesUpdateStatus(t *testing.T) {
	ctx := context.Background()
	engine := newWlasEngine("engine", 300)
	pod := boundPod(wlasTestNamespace, "p1", "gpu-a", map[string]string{"app": "predictor"})
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), nodeNamed("gpu-b"), pod)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "engine")
	req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: wlasTestNamespace, Name: "engine"}}

	// Pass 1: workload leaves gpu-b after having been there (injected as a
	// pre-labeled node carrying no wanted bookkeeping — a node the workload
	// left BEFORE the operator ever saw it).
	pre := nodeNamed("gpu-b")
	pre.Labels[key] = resources.WorkloadLabelValueBase
	if err := c.Update(ctx, pre); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Reconcile(ctx, req); err != nil {
		t.Fatal(err)
	}
	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, req.NamespacedName, e); err != nil {
		t.Fatal(err)
	}
	if e.Status.WorkloadLabelKey != key {
		t.Fatalf("workloadLabelKey not persisted through updateStatus's re-fetch")
	}
	if len(e.Status.WorkloadNodes) != 2 || e.Status.WorkloadNodes[0] != "gpu-a" || e.Status.WorkloadNodes[1] != "gpu-b" {
		t.Fatalf("workloadNodes (visible set must keep gpu-b inside its release window): %v", e.Status.WorkloadNodes)
	}
	if ts, ok := e.Status.NodeAbsentSince["gpu-b"]; !ok || ts.IsZero() {
		t.Fatalf("absence window bookkeeping for gpu-b not persisted: %+v", e.Status.NodeAbsentSince)
	}

	// Pass 2: a second full Reconcile that must not change pass-1's
	// bookkeeping (no restart of release windows).
	savedSince := e.Status.NodeAbsentSince["gpu-b"]
	if _, err := r.Reconcile(ctx, req); err != nil {
		t.Fatal(err)
	}
	if err := c.Get(ctx, req.NamespacedName, e); err != nil {
		t.Fatal(err)
	}
	ts2, ok := e.Status.NodeAbsentSince["gpu-b"]
	if !ok || !ts2.Equal(&savedSince) {
		t.Fatalf("release window restarted between reconciles: %v -> %v", savedSince, ts2)
	}
}

// --- helpers ---------------------------------------------------------------

func disableEngine(t *testing.T, ctx context.Context, c client.Client, engine *lmcachev1alpha1.LMCacheEngine) {
	t.Helper()
	fresh := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: engine.Namespace}, fresh); err != nil {
		t.Fatal(err)
	}
	fresh.Spec.WorkloadAwareScheduling.Enabled = false
	if err := c.Update(ctx, fresh); err != nil {
		t.Fatal(err)
	}
	if err := c.Status().Update(ctx, fresh); err != nil {
		t.Fatal(err)
	}
}
