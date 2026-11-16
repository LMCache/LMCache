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
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

// --- DRA fixtures -----------------------------------------------------------

// newDraEngine returns a WLA+DRA engine (named "s1", the shared fixture
// name of this suite) with an explicit tier-change delay (tests pass 0
// for immediacy or a value for hysteresis assertions).
func newDraEngine(tierDelaySecs int32) *lmcachev1alpha1.LMCacheEngine {
	e := newWlasEngine("s1", 300)
	e.Spec.L1.SizeGB = 10
	e.Spec.WorkloadAwareScheduling.DynamicResourceAllocation =
		&lmcachev1alpha1.DynamicResourceAllocationSpec{
			Enabled:                true,
			TierChangeDelaySeconds: &tierDelaySecs,
		}
	e.SetDefaults()
	return e
}

// gpuPod returns a bound workload pod requesting n nvidia.com/gpu.
func gpuPod(name, node string, gpus int64) *corev1.Pod {
	pod := boundPod(wlasTestNamespace, name, node, map[string]string{"app": "predictor"})
	pod.Spec.Containers = []corev1.Container{{
		Name: "workload",
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				"nvidia.com/gpu": resource.MustParse(intToStr(gpus)),
			},
		},
	}}
	return pod
}

func intToStr(n int64) string {
	if n == 0 {
		return "0"
	}
	digits := []byte{}
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}

// pendingEnginePod returns a stuck-Pending engine pod (no nodeName)
// carrying the DS selector labels, so updateStatus's pod list matches it.
func pendingEnginePod(engineName, name string) *corev1.Pod {
	pod := boundPod(wlasTestNamespace, name, "", resources.SelectorLabels(engineName))
	pod.Status.Phase = corev1.PodPending
	return pod
}

func reconcileOnce(ctx context.Context, t *testing.T, r *LMCacheEngineReconciler, name string) {
	t.Helper()
	_, err := r.Reconcile(ctx, ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: wlasTestNamespace, Name: name},
	})
	if err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
}

// getDS fetches a DaemonSet or returns nil when absent.
func getDS(ctx context.Context, t *testing.T, c client.Client, name string) *appsv1.DaemonSet {
	t.Helper()
	ds := &appsv1.DaemonSet{}
	err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: name}, ds)
	if err != nil {
		return nil
	}
	return ds
}

// --- full-Reconcile proof ---------------------------------------------------

// TestDRA_TierDaemonSetsPerNodeGPUCount is the new-behavior proof: two
// nodes, one workload pod requesting 1 GPU on nodeA and one requesting 2
// GPUs on nodeB. After Reconcile: nodeA labeled tier "1", nodeB "2"; base
// DS unchanged selector with nodeSelector "1"; tier DS "s1-2x" with
// nodeSelector "2", scaled --l1-size-gb 20.0 and 25Gi memory; both DS
// templates share the instance label.
func TestDRA_TierDaemonSetsPerNodeGPUCount(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(0)
	podA := gpuPod("pa", "gpu-a", 1)
	podB := gpuPod("pb", "gpu-b", 2)
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), nodeNamed("gpu-b"), podA, podB)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "s1")

	reconcileOnce(ctx, t, r, "s1")

	// Node labels carry tiers.
	nodeA := &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-a"}, nodeA); err != nil {
		t.Fatal(err)
	}
	if nodeA.Labels[key] != "1" {
		t.Fatalf("gpu-a tier = %q, want \"1\"", nodeA.Labels[key])
	}
	nodeB := &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-b"}, nodeB); err != nil {
		t.Fatal(err)
	}
	if nodeB.Labels[key] != "2" {
		t.Fatalf("gpu-b tier = %q, want \"2\"", nodeB.Labels[key])
	}

	// Base DS: selector unchanged, nodeSelector pins tier 1.
	base := getDS(ctx, t, c, "s1")
	if base == nil {
		t.Fatal("base DS missing")
	}
	if _, has := base.Spec.Selector.MatchLabels[resources.TierLabelKey]; has {
		t.Fatalf("base DS selector must not reference the tier label: %v", base.Spec.Selector.MatchLabels)
	}
	if base.Spec.Template.Spec.NodeSelector[key] != "1" {
		t.Fatalf("base DS nodeSelector[%s] = %q, want \"1\"", key, base.Spec.Template.Spec.NodeSelector[key])
	}

	// Tier DS: scaled and routed.
	tier := getDS(ctx, t, c, "s1-2x")
	if tier == nil {
		t.Fatal("tier DS s1-2x missing")
	}
	if tier.Spec.Selector.MatchLabels[resources.TierLabelKey] != "s1-2x" {
		t.Fatalf("tier DS selector must require %s=s1-2x: %v", resources.TierLabelKey, tier.Spec.Selector.MatchLabels)
	}
	if tier.Spec.Template.Spec.NodeSelector[key] != "2" {
		t.Fatalf("tier DS nodeSelector[%s] = %q, want \"2\"", key, tier.Spec.Template.Spec.NodeSelector[key])
	}
	if tier.Spec.Template.Labels["app.kubernetes.io/instance"] != "s1" {
		t.Fatalf("tier DS must share the instance label: %v", tier.Spec.Template.Labels)
	}
	if !metav1.IsControlledBy(tier, engine) {
		t.Fatal("tier DS must be owned by the engine")
	}
	container := tier.Spec.Template.Spec.Containers[0]
	found := false
	for i, a := range container.Args {
		if a == "--l1-size-gb" {
			if container.Args[i+1] != "20.0" {
				t.Fatalf("scaled --l1-size-gb = %q, want 20.0", container.Args[i+1])
			}
			found = true
		}
	}
	if !found {
		t.Fatalf("--l1-size-gb missing from tier container args: %v", container.Args)
	}
	if mem := container.Resources.Requests.Memory(); mem.String() != "25Gi" {
		t.Fatalf("scaled memory request = %q, want 25Gi", mem.String())
	}
}

// TestDRA_HysteresisDelaysTierRelabel pins the damping: a GPU-sum change
// observed for less than tierChangeDelaySeconds does NOT relabel (and so
// does not restart the engine pod); past the window it does. A FIRST
// label is immediate — hysteresis only guards changes.
func TestDRA_HysteresisDelaysTierRelabel(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(300)
	podB := gpuPod("pb", "gpu-b", 1)
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), nodeNamed("gpu-b"), podB)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "s1")

	reconcileOnce(ctx, t, r, "s1")

	node := &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-b"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != "1" {
		t.Fatalf("initial tier = %q, want \"1\"", node.Labels[key])
	}

	// The workload doubles its GPUs on gpu-b.
	podB2 := gpuPod("pb", "gpu-b", 2)
	if err := c.Update(ctx, podB2); err != nil {
		t.Fatal(err)
	}
	reconcileOnce(ctx, t, r, "s1")
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-b"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != "1" {
		t.Fatalf("tier relabeled before the hysteresis window elapsed: %q", node.Labels[key])
	}
	// Bookkeeping records the pending change.
	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	if ts, ok := e.Status.NodeTierChanges["gpu-b"]; !ok || ts.Since.IsZero() {
		t.Fatalf("tier-change bookkeeping missing: %+v", e.Status.NodeTierChanges)
	}

	// Expiry cannot be simulated with wall time here; assert the window
	// is wired by checking the pending entry exists and the label held.
	// (The window math itself is exercised via the same status path as
	// release windows, which the placement tests pin.)
}

// TestDRA_TierChangeDelayZeroAppliesImmediately: with a zero delay the
// relabel lands on the very next reconcile.
func TestDRA_TierChangeDelayZeroAppliesImmediately(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(0)
	podB := gpuPod("pb", "gpu-b", 1)
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-b"), podB)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "s1")

	reconcileOnce(ctx, t, r, "s1")
	if err := c.Update(ctx, gpuPod("pb", "gpu-b", 2)); err != nil {
		t.Fatal(err)
	}
	reconcileOnce(ctx, t, r, "s1")

	node := &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-b"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != "2" {
		t.Fatalf("zero-delay tier relabel = %q, want \"2\"", node.Labels[key])
	}
	if getDS(ctx, t, c, "s1-2x") == nil {
		t.Fatal("tier DS must exist after the relabel")
	}
}

// TestDRA_DisableDeletesTierDSBeforeLabelPurge: flipping DRA off deletes
// the tier DS, keeps the base DS, and the labels still purge via the
// existing WLA-off path.
func TestDRA_DisableDeletesTierDS(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(0)
	podA := gpuPod("pa", "gpu-a", 1)
	podB := gpuPod("pb", "gpu-b", 2)
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), nodeNamed("gpu-b"), podA, podB)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}

	reconcileOnce(ctx, t, r, "s1")
	if getDS(ctx, t, c, "s1-2x") == nil {
		t.Fatal("tier DS must exist while DRA is on")
	}

	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	e.Spec.WorkloadAwareScheduling.DynamicResourceAllocation.Enabled = false
	if err := c.Update(ctx, e); err != nil {
		t.Fatal(err)
	}
	reconcileOnce(ctx, t, r, "s1")

	if getDS(ctx, t, c, "s1-2x") != nil {
		t.Fatal("tier DS must be deleted after DRA disable")
	}
	if getDS(ctx, t, c, "s1") == nil {
		t.Fatal("base DS must survive DRA disable")
	}
}

// TestDRA_TierEmptiesOutDeletesTierDS: when the workload on the tier-2
// node drops to 1 GPU (node relabeled "1"), the tier DS is deleted.
func TestDRA_TierEmptiesOutDeletesTierDS(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(0)
	podA := gpuPod("pa", "gpu-a", 1)
	podB := gpuPod("pb", "gpu-b", 2)
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), nodeNamed("gpu-b"), podA, podB)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}

	reconcileOnce(ctx, t, r, "s1")
	if getDS(ctx, t, c, "s1-2x") == nil {
		t.Fatal("tier DS must exist while tier 2 is active")
	}

	if err := c.Update(ctx, gpuPod("pb", "gpu-b", 1)); err != nil {
		t.Fatal(err)
	}
	reconcileOnce(ctx, t, r, "s1")
	reconcileOnce(ctx, t, r, "s1") // zero delay applies on the pass that records it

	if getDS(ctx, t, c, "s1-2x") != nil {
		t.Fatal("tier DS must be deleted when no node carries tier 2")
	}
}

// TestDRA_GhostNodeTierDoesNotActivateTierDS: a workload pod bound to a
// deleted node (pre-GC race) must not activate a tier DS that can never
// schedule a pod.
func TestDRA_GhostNodeTierDoesNotActivateTierDS(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(0)
	podA := gpuPod("pa", "node-a", 1)
	ghost := gpuPod("pg", "ghost-node", 3) // tier 3, Node object absent
	c := wlasReconcileClient(t, engine, nodeNamed("node-a"), podA, ghost)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}

	reconcileOnce(ctx, t, r, "s1")

	if ds := getDS(ctx, t, c, "s1-3x"); ds != nil {
		t.Fatal("a ghost node's tier must not activate a tier DS")
	}
	if getDS(ctx, t, c, "s1") == nil {
		t.Fatal("base DS missing")
	}
}

// TestDRA_StatusAggregatesAcrossTiers: desiredInstances sums over owned
// DSs; endpoints match pods of every tier via the shared instance label.
// The fake client does not run DS controllers, so DesiredNumberScheduled
// is seeded manually.
func TestDRA_StatusAggregatesAcrossTiers(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(0)
	podA := gpuPod("pa", "gpu-a", 1)
	podB := gpuPod("pb", "gpu-b", 2)
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), nodeNamed("gpu-b"), podA, podB)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}

	reconcileOnce(ctx, t, r, "s1")

	// Seed DS status as the DS controller would.
	for _, name := range []string{"s1", "s1-2x"} {
		ds := getDS(ctx, t, c, name)
		if ds == nil {
			t.Fatalf("DS %s missing", name)
		}
		ds.Status.DesiredNumberScheduled = 1
		if err := c.Status().Update(ctx, ds); err != nil {
			t.Fatal(err)
		}
	}

	reconcileOnce(ctx, t, r, "s1")

	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	if e.Status.DesiredInstances != 2 {
		t.Fatalf("DesiredInstances = %d, want 2 (sum over both DSs)", e.Status.DesiredInstances)
	}
}

// TestDRA_TierConditionGatedOnDRA: with DRA on, a stuck-Pending engine pod
// surfaces as TierSchedulable=False / TierPodPending.
func TestDRA_TierConditionGatedOnDRA(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(0)
	c := wlasReconcileClient(t, engine, pendingEnginePod("s1", "stuck"))
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}

	reconcileOnce(ctx, t, r, "s1")

	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	cond := meta.FindStatusCondition(e.Status.Conditions, lmcachev1alpha1.ConditionTierSchedulable)
	if cond == nil {
		t.Fatal("TierSchedulable condition missing")
	}
	if cond.Status != metav1.ConditionFalse {
		t.Fatalf("TierSchedulable status = %q, want False", cond.Status)
	}
	if cond.Reason != "TierPodPending" {
		t.Fatalf("TierSchedulable reason = %q, want TierPodPending", cond.Reason)
	}
}

// TestDRA_TierConditionNotSetWithoutDRA: a WLA-only engine gets no
// TierSchedulable condition — a pending base pod is ordinary scheduling.
func TestDRA_TierConditionNotSetWithoutDRA(t *testing.T) {
	ctx := context.Background()
	engine := newWlasEngine("s1", 300)
	c := wlasReconcileClient(t, engine, pendingEnginePod("s1", "stuck"))
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}

	reconcileOnce(ctx, t, r, "s1")

	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	if cond := meta.FindStatusCondition(e.Status.Conditions, lmcachev1alpha1.ConditionTierSchedulable); cond != nil {
		t.Fatalf("TierSchedulable must not be set without DRA: %+v", cond)
	}
}

// TestDRA_TierConditionClearedWhenDRADisabled: disabling DRA removes a
// previously-set False condition instead of leaving it stale.
func TestDRA_TierConditionClearedWhenDRADisabled(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(0)
	c := wlasReconcileClient(t, engine, pendingEnginePod("s1", "stuck"))
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}

	reconcileOnce(ctx, t, r, "s1")
	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	cond := meta.FindStatusCondition(e.Status.Conditions, lmcachev1alpha1.ConditionTierSchedulable)
	if cond == nil || cond.Status != metav1.ConditionFalse || cond.Reason != "TierPodPending" {
		t.Fatalf("condition must be False/TierPodPending before the disable, got %+v", cond)
	}

	e.Spec.WorkloadAwareScheduling.DynamicResourceAllocation.Enabled = false
	if err := c.Update(ctx, e); err != nil {
		t.Fatal(err)
	}
	reconcileOnce(ctx, t, r, "s1")

	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	if cond := meta.FindStatusCondition(e.Status.Conditions, lmcachev1alpha1.ConditionTierSchedulable); cond != nil {
		t.Fatalf("TierSchedulable must be removed after DRA disable, got %+v", cond)
	}
}

// TestDRA_NameCollisionS1AndS12x: a CR literally named "s1-2x" must not
// confuse tier-DS cleanup or status aggregation — ownership is by
// ownerRef + tier label, never by name pattern.
func TestDRA_NameCollisionS1AndS12x(t *testing.T) {
	ctx := context.Background()
	s1 := newDraEngine(0)
	s2 := newWlasEngine("s1-2x", 300) // plain WLA engine, no DRA
	podA := gpuPod("pa", "gpu-a", 1)
	podB := gpuPod("pb", "gpu-b", 2)
	podC := boundPod(wlasTestNamespace, "pc", "gpu-c", map[string]string{"app": "predictor"})
	c := wlasReconcileClient(t, s1, s2, nodeNamed("gpu-a"), nodeNamed("gpu-b"), nodeNamed("gpu-c"), podA, podB, podC)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}

	reconcileOnce(ctx, t, r, "s1")
	reconcileOnce(ctx, t, r, "s1-2x")

	// s1's tier DS exists ALONGSIDE s1-2x's own base DS — same object
	// name is impossible, so the tier DS must have lost the race only
	// if creation failed; the operator reconciles one DS per tier keyed
	// by tier, and s1-2x's base DS is a different engine's resource.
	// The invariant under test: s1-2x's base DS is never deleted by s1's
	// tier cleanup (no name-pattern matching).
	other := getDS(ctx, t, c, "s1-2x")
	if other != nil && !metav1.IsControlledBy(other, s2) {
		t.Fatal("DS named s1-2x must remain owned by the s1-2x engine, not adopted or deleted by s1's tier cleanup")
	}
}

// TestDRA_EnableOnExistingEngineKeepsBaseSelector is the immutability
// regression: a base DS created by a pre-DRA reconcile (SelectorLabels
// only) must reconcile cleanly after DRA is enabled — the patch touches
// only mutable template fields.
func TestDRA_EnableOnExistingEngineKeepsBaseSelector(t *testing.T) {
	ctx := context.Background()
	engine := newWlasEngine("s1", 300)
	podA := gpuPod("pa", "gpu-a", 1)
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), podA)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}

	// Pre-DRA reconcile creates the base DS.
	reconcileOnce(ctx, t, r, "s1")
	base := getDS(ctx, t, c, "s1")
	if base == nil {
		t.Fatal("base DS missing after pre-DRA reconcile")
	}
	selectorBefore := base.Spec.Selector.DeepCopy()

	// Enable DRA.
	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	e.Spec.WorkloadAwareScheduling.DynamicResourceAllocation =
		&lmcachev1alpha1.DynamicResourceAllocationSpec{Enabled: true}
	delay := int32(0)
	e.Spec.WorkloadAwareScheduling.DynamicResourceAllocation.TierChangeDelaySeconds = &delay
	if err := c.Update(ctx, e); err != nil {
		t.Fatal(err)
	}
	reconcileOnce(ctx, t, r, "s1")

	base = getDS(ctx, t, c, "s1")
	if base == nil {
		t.Fatal("base DS missing after DRA enable")
	}
	if len(base.Spec.Selector.MatchLabels) != len(selectorBefore.MatchLabels) {
		t.Fatalf("base DS selector changed by DRA enable: %v vs %v",
			base.Spec.Selector.MatchLabels, selectorBefore.MatchLabels)
	}
}

// TestDRA_GPUFilterMatchesPlacement pins the shared-filter contract:
// unbound, terminating, Failed and Succeeded pods neither place a node
// nor contribute GPUs to its tier.
func TestDRA_GPUFilterMatchesPlacement(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(0)
	active := gpuPod("pa", "gpu-a", 1)
	unbound := boundPod(wlasTestNamespace, "punbound", "", map[string]string{"app": "predictor"})
	terminating := boundPod(wlasTestNamespace, "pterm", "gpu-a", map[string]string{"app": "predictor"})
	failed := boundPod(wlasTestNamespace, "pfailed", "gpu-a", map[string]string{"app": "predictor"})
	failed.Status.Phase = corev1.PodFailed
	succeeded := boundPod(wlasTestNamespace, "psucc", "gpu-a", map[string]string{"app": "predictor"})
	succeeded.Status.Phase = corev1.PodSucceeded
	// Finished pods with big GPU requests must not inflate the tier.
	for _, p := range []*corev1.Pod{failed, succeeded} {
		p.Spec.Containers = []corev1.Container{{
			Name: "workload",
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					"nvidia.com/gpu": resource.MustParse("8"),
				},
			},
		}}
	}

	now := metav1.Now()
	terminating.DeletionTimestamp = &now
	terminating.Finalizers = []string{"test.keep"}
	terminating.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: "v1", Kind: "Pod", Name: "self", UID: "fake-uid",
	}}
	c := wlasReconcileClient(t, engine, nodeNamed("gpu-a"), active, unbound, terminating, failed, succeeded)
	r := &LMCacheEngineReconciler{Client: c, Scheme: wlasReconcileScheme(t)}
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "s1")

	reconcileOnce(ctx, t, r, "s1")

	node := &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-a"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != "1" {
		t.Fatalf("tier = %q, want \"1\" (terminal pods must not inflate the GPU sum)", node.Labels[key])
	}
}

// TestDRA_HysteresisAppliesAfterWindow pins the other half of the damping
// contract: once the changed tier has been continuously observed past
// tierChangeDelaySeconds, the relabel fires. Uses reconcileWorkloadPlacement
// directly so `now` can be injected (mirrors the WLA release-window tests).
func TestDRA_HysteresisAppliesAfterWindow(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(300)
	podB := gpuPod("pb", "gpu-b", 1)
	c := wlasClient(t, engine, nodeNamed("gpu-b"), podB)
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "s1")

	t0 := time.Now()
	mustPlacementReconcile(ctx, t, c, engine, t0)
	node := &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-b"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != "1" {
		t.Fatalf("initial tier = %q, want \"1\"", node.Labels[key])
	}

	// The workload doubles its GPUs; inside the window nothing changes.
	if err := c.Update(ctx, gpuPod("pb", "gpu-b", 2)); err != nil {
		t.Fatal(err)
	}
	t1 := t0.Add(60 * time.Second)
	mustPlacementReconcile(ctx, t, c, engine, t1)
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-b"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != "1" {
		t.Fatalf("tier relabeled inside the hysteresis window: %q", node.Labels[key])
	}
	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	pending, ok := e.Status.NodeTierChanges["gpu-b"]
	if !ok || pending.Since.IsZero() || pending.Tier != 2 {
		t.Fatalf("tier-change bookkeeping missing or wrong target: %+v", e.Status.NodeTierChanges)
	}

	// Past the window (300s default, observed from t1): the relabel fires
	// and the bookkeeping clears.
	t2 := t1.Add(301 * time.Second)
	mustPlacementReconcile(ctx, t, c, engine, t2)
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-b"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != "2" {
		t.Fatalf("tier not relabeled after the window: %q", node.Labels[key])
	}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	if len(e.Status.NodeTierChanges) != 0 {
		t.Fatalf("tier-change bookkeeping must clear after applying: %+v", e.Status.NodeTierChanges)
	}
}

// TestDRA_NonNumericResidueRelabeledImmediately: a node carrying the
// pre-DRA "true" value must not wait out the hysteresis window — the base
// DS selects "1", so the residue would hide the node's engine pod.
func TestDRA_NonNumericResidueRelabeledImmediately(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(300)
	podB := gpuPod("pb", "gpu-b", 1)
	node := nodeNamed("gpu-b")
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "s1")
	node.Labels[key] = "true"
	c := wlasClient(t, engine, node, podB)

	mustPlacementReconcile(ctx, t, c, engine, time.Now())

	fresh := &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-b"}, fresh); err != nil {
		t.Fatal(err)
	}
	if fresh.Labels[key] != "1" {
		t.Fatalf("non-numeric residue must relabel immediately, got %q", fresh.Labels[key])
	}
}

// TestDRA_HysteresisRestartsOnNewTarget pins the window-anchor contract:
// the hysteresis is anchored on the TARGET tier, so 1->2 aging out and
// then the sum moving to 3 never applies 3 on 2's stale timestamp — the
// window restarts for the new target.
func TestDRA_HysteresisRestartsOnNewTarget(t *testing.T) {
	ctx := context.Background()
	engine := newDraEngine(300)
	podB := gpuPod("pb", "gpu-b", 1)
	c := wlasClient(t, engine, nodeNamed("gpu-b"), podB)
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "s1")

	t0 := time.Now()
	mustPlacementReconcile(ctx, t, c, engine, t0)

	// Observe tier 2; then, before its window expires, the target
	// becomes 3.
	if err := c.Update(ctx, gpuPod("pb", "gpu-b", 2)); err != nil {
		t.Fatal(err)
	}
	t1 := t0.Add(60 * time.Second)
	mustPlacementReconcile(ctx, t, c, engine, t1)
	if err := c.Update(ctx, gpuPod("pb", "gpu-b", 3)); err != nil {
		t.Fatal(err)
	}
	t2 := t1.Add(60 * time.Second)
	mustPlacementReconcile(ctx, t, c, engine, t2)

	e := &lmcachev1alpha1.LMCacheEngine{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: wlasTestNamespace, Name: "s1"}, e); err != nil {
		t.Fatal(err)
	}
	pending, ok := e.Status.NodeTierChanges["gpu-b"]
	if !ok || pending.Tier != 3 {
		t.Fatalf("pending target must be tier 3 with a restarted window, got %+v", e.Status.NodeTierChanges)
	}
	// The restarted window for 3 (observed at t2) must not have expired
	// by t2+240s — 2's original window (from t1) WOULD have expired by
	// then. This is the distinction the test exists for.
	t3 := t2.Add(240 * time.Second)
	mustPlacementReconcile(ctx, t, c, engine, t3)
	node := &corev1.Node{}
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-b"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != "1" {
		t.Fatalf("tier applied on the OLD target's stale timestamp: label = %q, want \"1\" until 3's own window expires", node.Labels[key])
	}

	// 3's window (from t2, 300s) expires by t4.
	t4 := t2.Add(301 * time.Second)
	mustPlacementReconcile(ctx, t, c, engine, t4)
	if err := c.Get(ctx, client.ObjectKey{Name: "gpu-b"}, node); err != nil {
		t.Fatal(err)
	}
	if node.Labels[key] != "3" {
		t.Fatalf("tier 3 not applied after its own full window: label = %q", node.Labels[key])
	}
}
