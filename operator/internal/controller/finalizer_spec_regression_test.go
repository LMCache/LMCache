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
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// specSnapshot returns the STORED spec of the engine (fetched fresh, not
// the reconciler's in-memory copy).
func specSnapshot(ctx context.Context, t *testing.T, c client.Client, name string) lmcachev1alpha1.LMCacheEngineSpec {
	t.Helper()
	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: name}, e); err != nil {
		t.Fatal(err)
	}
	return e.Spec
}

// TestEnableDisableNeverWritesSpec is the regression for the two
// stored-spec corruption bugs:
//
//  1. Enabling WLAS ran AddFinalizer + full-object Update AFTER
//     SetDefaults(), persisting defaulted fields (gpuVendor, logLevel,
//     releaseDelaySeconds, injected nodeSelector) the user never wrote —
//     generation bumps and GitOps drift.
//  2. Disabling WLAS ran RemoveFinalizer + full-object Update after
//     reconcileDaemonSet had resolved spec.coordinator (ref AND url set
//     in memory), permanently failing validation on coordinator.ref
//     engines ("exactly one of ref or url must be set").
//
// Both finalizer changes are now metadata-only patches; the stored spec
// must be byte-identical before enable, after enable, and after disable.
func TestEnableDisableNeverWritesSpec(t *testing.T) {
	ctx := context.Background()
	engine := newWlasEngine("engine", 300)
	// Deliberately leave fields unset that SetDefaults fills in, so the
	// test can catch any writeback of defaulted values.
	engine.Spec.GPUVendor = nil
	engine.Spec.LogLevel = nil
	engine.Spec.NodeSelector = nil
	engine.Spec.WorkloadAwareScheduling.ReleaseDelaySeconds = nil

	pod := boundPod(wlasTestNamespace, "p1", "gpu-a", map[string]string{"app": "predictor"})
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), pod)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}
	req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: wlasTestNamespace, Name: "engine"}}

	before := specSnapshot(ctx, t, c, "engine")
	beforeWLAS := *before.WorkloadAwareScheduling

	// --- Enable path: finalizer added, spec untouched.
	if _, err := r.Reconcile(ctx, req); err != nil {
		t.Fatalf("enable reconcile: %v", err)
	}
	afterEnable := specSnapshot(ctx, t, c, "engine")
	if !reflect.DeepEqual(before, afterEnable) {
		t.Fatalf("enable wrote defaulted/mutated fields into the stored spec:\nbefore: %+v\nafter:  %+v", before, afterEnable)
	}
	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, req.NamespacedName, e); err != nil {
		t.Fatal(err)
	}
	hasFinalizer := false
	for _, f := range e.Finalizers {
		if f == workloadPlacementFinalizer {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		t.Fatal("finalizer must be added on the enable path")
	}

	// --- Disable path: finalizer removed, spec STILL untouched.
	disableEngine(t, ctx, c, e)
	if _, err := r.Reconcile(ctx, req); err != nil {
		t.Fatalf("disable reconcile: %v", err)
	}
	afterDisable := specSnapshot(ctx, t, c, "engine")
	// The user's one intentional change is Enabled=false; nothing else in
	// the spec (no defaulted fields, no resolved coordinator) may differ.
	wantWLAS := beforeWLAS
	wantWLAS.Enabled = false
	if !reflect.DeepEqual(afterDisable.WorkloadAwareScheduling, &wantWLAS) {
		t.Fatalf("disable left unexpected workloadAwareScheduling changes:\ngot:  %+v\nwant: %+v", afterDisable.WorkloadAwareScheduling, &wantWLAS)
	}
	afterDisable.WorkloadAwareScheduling = before.WorkloadAwareScheduling
	if !reflect.DeepEqual(before, afterDisable) {
		t.Fatalf("disable wrote mutated fields (coordinator resolution?) into the stored spec:\nbefore: %+v\nafter:  %+v", before, afterDisable)
	}
	if err := c.Get(ctx, req.NamespacedName, e); err != nil {
		t.Fatal(err)
	}
	for _, f := range e.Finalizers {
		if f == workloadPlacementFinalizer {
			t.Fatal("finalizer must be removed on the disable path")
		}
	}
}

// TestCoordinatorRefSurvivesDisable is the direct regression for the CR
// bricking bug: an engine using spec.coordinator.ref must remain valid
// after reconcileDaemonSet resolves the connection in memory and the
// disable path removes the finalizer.
func TestCoordinatorRefSurvivesDisable(t *testing.T) {
	ctx := context.Background()
	engine := newWlasEngine("engine", 300)
	// Coordinator via ref only — the shape the old full-Update corrupted.
	engine.Spec.Coordinator = &lmcachev1alpha1.CoordinatorConnectionSpec{
		Ref: &corev1.LocalObjectReference{Name: "coord"},
	}

	// Stub coordinator so reconcileDaemonSet can resolve the ref — the
	// resolution result (ref AND url in memory) is exactly what the old
	// bug persisted.
	coord := &lmcachev1alpha1.LMCacheCoordinator{
		ObjectMeta: metav1.ObjectMeta{Name: "coord", Namespace: wlasTestNamespace},
	}
	pod := boundPod(wlasTestNamespace, "p1", "gpu-a", map[string]string{"app": "predictor"})
	c := wlasReconcileClient(t, engine, coord, nodeNamed("gpu-a"), pod)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}
	req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: wlasTestNamespace, Name: "engine"}}

	if _, err := r.Reconcile(ctx, req); err != nil {
		t.Fatalf("enable reconcile: %v", err)
	}
	disableEngine(t, ctx, c, engine)
	if _, err := r.Reconcile(ctx, req); err != nil {
		t.Fatalf("disable reconcile: %v", err)
	}

	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, req.NamespacedName, e); err != nil {
		t.Fatal(err)
	}
	// The stored spec must still have ref-only (no resolved url persisted).
	if e.Spec.Coordinator == nil || e.Spec.Coordinator.Ref == nil || e.Spec.Coordinator.URL != nil {
		t.Fatalf("stored coordinator spec corrupted by reconcile: %+v", e.Spec.Coordinator)
	}
	// And the CR must still pass validation — the brick check.
	if errs := e.ValidateSpec(); len(errs) != 0 {
		t.Fatalf("engine permanently invalid after disable (the brick): %v", errs)
	}
}

// silence metav1 if only used in future edits of this file
var _ = metav1.Now
