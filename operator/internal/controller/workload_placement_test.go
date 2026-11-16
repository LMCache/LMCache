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

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

// --- fixtures --------------------------------------------------------------

func wlasTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	sc := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(sc); err != nil {
		t.Fatal(err)
	}
	if err := lmcachev1alpha1.AddToScheme(sc); err != nil {
		t.Fatal(err)
	}
	return sc
}

// wlasTestNamespace is fixed because every fixture in this suite lives in
// one namespace; the cross-namespace rejection is exercised by creating a
// foreign-namespace pod, not by moving the engine.
const wlasTestNamespace = "mlservices"

func newWlasEngine(name string, delaySecs int32) *lmcachev1alpha1.LMCacheEngine {
	e := &lmcachev1alpha1.LMCacheEngine{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: wlasTestNamespace},
	}
	e.Spec.L1.SizeGB = 8
	e.Spec.WorkloadAwareScheduling = &lmcachev1alpha1.WorkloadAwareSchedulingSpec{
		Enabled: true,
		WorkloadSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{"app": "predictor"},
		},
		ReleaseDelaySeconds: &delaySecs,
	}
	e.SetDefaults()
	return e
}

func boundPod(ns, name, node string, lbls map[string]string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns, Labels: lbls},
		Spec:       corev1.PodSpec{NodeName: node},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning},
	}
}

func nodeNamed(name string) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"kubernetes.io/hostname": name}},
	}
}

func wlasClient(t *testing.T, objs ...client.Object) client.Client {
	t.Helper()
	return fake.NewClientBuilder().
		WithScheme(wlasTestScheme(t)).
		WithObjects(objs...).
		WithStatusSubresource(&lmcachev1alpha1.LMCacheEngine{}).
		Build()
}

func mustPlacementReconcile(ctx context.Context, t *testing.T, c client.Client, e *lmcachev1alpha1.LMCacheEngine, now time.Time) time.Duration {
	t.Helper()
	delay, _, err := reconcileWorkloadPlacement(ctx, c, e, now)
	if err != nil {
		t.Fatalf("reconcileWorkloadPlacement: %v", err)
	}
	return delay
}

func labelSnapshot(ctx context.Context, t *testing.T, c client.Client, key string) map[string]bool {
	t.Helper()
	out := map[string]bool{}
	nodes := &corev1.NodeList{}
	if err := c.List(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	for _, n := range nodes.Items {
		if v, ok := n.Labels[key]; ok && v == resources.WorkloadLabelValueBase {
			out[n.Name] = true
		}
	}
	return out
}

// --- tests -----------------------------------------------------------------

func TestWorkloadPlacement_LabelsNodesHostingBoundPods(t *testing.T) {
	ctx := context.Background()
	e := newWlasEngine("engine", 300)
	c := wlasClient(t,
		e, nodeNamed("gpu-a"), nodeNamed("gpu-b"), nodeNamed("gpu-c"),
		boundPod("mlservices", "p1", "gpu-a", map[string]string{"app": "predictor"}),
		boundPod("mlservices", "p2", "gpu-b", map[string]string{"app": "predictor"}),
		boundPod("mlservices", "p3", "gpu-c", map[string]string{"app": "other"}), // selector mismatch
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p4", Namespace: "mlservices", Labels: map[string]string{"app": "predictor"}},
			Spec: corev1.PodSpec{ /* unbound */ }, Status: corev1.PodStatus{Phase: corev1.PodPending}}, // not bound
	)
	if err := c.Create(ctx, boundPod("other-ns", "p5", "gpu-c", map[string]string{"app": "predictor"})); err != nil {
		t.Fatal(err) // must not count: other namespace
	}

	mustPlacementReconcile(ctx, t, c, e, time.Now())

	key := resources.WorkloadPlacementLabel("mlservices", "engine")
	got := labelSnapshot(ctx, t, c, key)
	want := map[string]bool{"gpu-a": true, "gpu-b": true}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("labels mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"gpu-a", "gpu-b"}, e.Status.WorkloadNodes); diff != "" {
		t.Fatalf("status.workloadNodes mismatch (-want +got):\n%s", diff)
	}
	if e.Status.WorkloadLabelKey != key {
		t.Fatalf("status.workloadLabelKey = %q, want %q", e.Status.WorkloadLabelKey, key)
	}
}

func TestWorkloadPlacement_KeepsLabelThroughReleaseWindow(t *testing.T) {
	ctx := context.Background()
	e := newWlasEngine("engine", 300)
	pod := boundPod("mlservices", "p1", "gpu-a", map[string]string{"app": "predictor"})
	c := wlasClient(t, e, nodeNamed("gpu-a"), pod)

	t0 := time.Now()
	mustPlacementReconcile(ctx, t, c, e, t0)
	key := resources.WorkloadPlacementLabel("mlservices", "engine")

	// Workload leaves; first absence is observed 60s in. The release window
	// starts at THIS observation, so a full 300s window remains — that is
	// the first-observed-absence semantics, not a "time since last seen"
	// countdown.
	if err := c.Delete(ctx, pod); err != nil {
		t.Fatal(err)
	}
	delay := mustPlacementReconcile(ctx, t, c, e, t0.Add(60*time.Second))
	if delay != 300*time.Second {
		t.Fatalf("expected a full 300s window from first-observed absence, got %v", delay)
	}
	if got := labelSnapshot(ctx, t, c, key); !got["gpu-a"] {
		t.Fatalf("label must be kept inside the window, got %v", got)
	}

	// Deterministic bookkeeping first observed at t0+60s: expires at t0+360s.
	mustPlacementReconcile(ctx, t, c, e, t0.Add(359*time.Second))
	if got := labelSnapshot(ctx, t, c, key); !got["gpu-a"] {
		t.Fatalf("one second before expiry the label must remain, got %v", got)
	}
	mustPlacementReconcile(ctx, t, c, e, t0.Add(361*time.Second))
	if got := labelSnapshot(ctx, t, c, key); got["gpu-a"] {
		t.Fatalf("label must be removed once the window elapses, got %v", got)
	}
}

// Regression: the release window runs from the FIRST OBSERVED absence, so a
// controller interruption longer than releaseDelaySeconds that misses the
// actual departure still preserves the workload's full release window.
func TestWorkloadPlacement_LongGapStillGrantsFullWindow(t *testing.T) {
	ctx := context.Background()
	e := newWlasEngine("engine", 300)
	pod := boundPod("mlservices", "p1", "gpu-a", map[string]string{"app": "predictor"})
	c := wlasClient(t, e, nodeNamed("gpu-a"), pod)

	t0 := time.Now()
	mustPlacementReconcile(ctx, t, c, e, t0)
	if err := c.Delete(ctx, pod); err != nil {
		t.Fatal(err)
	}

	// No reconciles happen for 400s — longer than the 300s delay and past
	// the actual deletion moment. The first reconcile back must NOT treat
	// the stale pre-gap bookkeeping as an elapsed window.
	delay := mustPlacementReconcile(ctx, t, c, e, t0.Add(400*time.Second))
	if delay <= 0 || delay > 300*time.Second {
		t.Fatalf("expected a full fresh window after the gap, got %v", delay)
	}
	key := resources.WorkloadPlacementLabel("mlservices", "engine")
	if got := labelSnapshot(ctx, t, c, key); !got["gpu-a"] {
		t.Fatalf("label must survive the interrupted-controller gap, got %v", got)
	}
}

// Regression: a node already carrying the label without bookkeeping (labeled
// by an operator version predating nodeAbsentSince) must not be cleaned up
// instantly — it gets one full release window from its first handling.
func TestWorkloadPlacement_UntrackedLabelGetsFullWindow(t *testing.T) {
	ctx := context.Background()
	e := newWlasEngine("engine", 300)
	key := resources.WorkloadPlacementLabel("mlservices", "engine")
	preLabeled := nodeNamed("gpu-a")
	preLabeled.Labels[key] = resources.WorkloadLabelValueBase
	c := wlasClient(t, e, preLabeled) // no workload pods at all

	t0 := time.Now()
	delay := mustPlacementReconcile(ctx, t, c, e, t0)
	if delay <= 0 || delay > 300*time.Second {
		t.Fatalf("expected a full window granted at first observation, got %v", delay)
	}
	if got := labelSnapshot(ctx, t, c, key); got["gpu-a"] == false {
		t.Fatal("untracked label must not be removed on first observation")
	}

	mustPlacementReconcile(ctx, t, c, e, t0.Add(301*time.Second))
	if got := labelSnapshot(ctx, t, c, key); got["gpu-a"] {
		t.Fatalf("untracked label must be removed after one full window, got %v", got)
	}
}

func TestWorkloadPlacement_ZeroDelayRemovesImmediately(t *testing.T) {
	ctx := context.Background()
	e := newWlasEngine("engine", 0)
	pod := boundPod("mlservices", "p1", "gpu-a", map[string]string{"app": "predictor"})
	c := wlasClient(t, e, nodeNamed("gpu-a"), pod)

	t0 := time.Now()
	mustPlacementReconcile(ctx, t, c, e, t0)
	if err := c.Delete(ctx, pod); err != nil {
		t.Fatal(err)
	}
	mustPlacementReconcile(ctx, t, c, e, t0.Add(time.Second))

	key := resources.WorkloadPlacementLabel("mlservices", "engine")
	if got := labelSnapshot(ctx, t, c, key); got["gpu-a"] {
		t.Fatalf("expected label removed with zero delay, got %v", got)
	}
}

func TestWorkloadPlacement_DisabledStripsResidue(t *testing.T) {
	ctx := context.Background()
	e := newWlasEngine("engine", 300)
	c := wlasClient(t, e, nodeNamed("gpu-a"),
		boundPod("mlservices", "p1", "gpu-a", map[string]string{"app": "predictor"}))
	key := resources.WorkloadPlacementLabel("mlservices", "engine")

	mustPlacementReconcile(ctx, t, c, e, time.Now())
	if got := labelSnapshot(ctx, t, c, key); !got["gpu-a"] {
		t.Fatal("precondition: gpu-a labeled")
	}

	e.Spec.WorkloadAwareScheduling.Enabled = false
	if err := c.Update(ctx, e); err != nil {
		t.Fatal(err)
	}
	if err := c.Status().Update(ctx, e); err != nil {
		t.Fatal(err)
	}
	mustPlacementReconcile(ctx, t, c, e, time.Now().Add(time.Second))
	if got := labelSnapshot(ctx, t, c, key); len(got) != 0 {
		t.Fatalf("disabling must strip every managed label, got %v", got)
	}
}

func TestWorkloadPlacement_IgnoresTerminatingAndTerminalPods(t *testing.T) {
	ctx := context.Background()
	e := newWlasEngine("engine", 0)
	terminating := boundPod("mlservices", "p1", "gpu-a", map[string]string{"app": "predictor"})
	now := metav1.Now()
	terminating.DeletionTimestamp = &now
	terminating.Finalizers = []string{"test/finalizer"} // fake client requires finalizers on terminating objects
	failed := boundPod("mlservices", "p2", "gpu-b", map[string]string{"app": "predictor"})
	failed.Status.Phase = corev1.PodFailed

	c := wlasClient(t, e, nodeNamed("gpu-a"), nodeNamed("gpu-b"), terminating)
	if err := c.Create(ctx, failed); err != nil {
		t.Fatal(err)
	}
	mustPlacementReconcile(ctx, t, c, e, time.Now())

	key := resources.WorkloadPlacementLabel("mlservices", "engine")
	if got := labelSnapshot(ctx, t, c, key); len(got) != 0 {
		t.Fatalf("terminating/terminal pods must not earn labels, got %v", got)
	}
}

func TestStripWorkloadPlacementLabels_PerEngineIsolation(t *testing.T) {
	ctx := context.Background()
	e1 := newWlasEngine("engine", 300)
	e2 := newWlasEngine("engine-2", 300)
	c := wlasClient(t, e1, e2, nodeNamed("gpu-a"), nodeNamed("gpu-b"),
		boundPod("mlservices", "p1", "gpu-a", map[string]string{"app": "predictor"}),
		boundPod("mlservices", "q1", "gpu-b", map[string]string{"app": "predictor"}))
	now := time.Now()
	mustPlacementReconcile(ctx, t, c, e1, now)
	mustPlacementReconcile(ctx, t, c, e2, now)

	key1 := resources.WorkloadPlacementLabel("mlservices", "engine")
	key2 := resources.WorkloadPlacementLabel("mlservices", "engine-2")
	if !labelSnapshot(ctx, t, c, key1)["gpu-a"] || !labelSnapshot(ctx, t, c, key2)["gpu-b"] {
		t.Fatal("precondition: both engines labeled their nodes")
	}

	if err := stripWorkloadPlacementLabels(ctx, c, e1); err != nil {
		t.Fatal(err)
	}
	if snap := labelSnapshot(ctx, t, c, key1); len(snap) != 0 {
		t.Fatalf("engine 1 labels must be gone, got %v", snap)
	}
	if snap := labelSnapshot(ctx, t, c, key2); !snap["gpu-b"] {
		t.Fatalf("engine 2 labels must be untouched, got %v", snap)
	}
}

func TestWorkloadMatches_MalformedSelectorNeverMatches(t *testing.T) {
	e := newWlasEngine("engine", 300)
	e.Spec.WorkloadAwareScheduling.WorkloadSelector = &metav1.LabelSelector{
		MatchExpressions: []metav1.LabelSelectorRequirement{
			{Key: "x", Operator: "BogusOperator", Values: []string{"y"}},
		},
	}
	if _, err := workloadSelectorFor(e); err == nil {
		t.Fatal("expected selector parse error")
	}
	if workloadMatches(e, map[string]string{"x": "y"}) {
		t.Fatal("malformed selector must not match anything")
	}
}

// Strengthened per review: the long-gap regression must exercise stale
// PRE-GAP absence bookkeeping, not just a gap from a never-absent state.
// An absence window observed BEFORE a controller outage must be honored
// exactly — expired across the gap → removed; within its window across the
// gap → kept. The window never restarts.
func TestWorkloadPlacement_PreGapAbsenceEntryHonoredAcrossGap(t *testing.T) {
	ctx := context.Background()

	// Expired-entry case: absence observed at t0-400 (releaseDelay 300 →
	// deadline t0-100), controller resumes only at t0 → the label must be
	// removed. First-observed-absence never re-grants once entries exist.
	t.Run("expired pre-gap entry is removed on resume", func(t *testing.T) {
		e := newWlasEngine("engine", 300)
		key := resources.WorkloadPlacementLabel(wlasTestNamespace, "engine")
		pre := nodeNamed("gpu-a")
		pre.Labels[key] = resources.WorkloadLabelValueBase
		c := wlasClient(t, e, pre) // no matching pods; label pre-seeded

		gapStart := time.Now().Add(-400 * time.Second)
		e.Status.NodeAbsentSince = map[string]metav1.Time{"gpu-a": {Time: gapStart}}
		if err := c.Status().Update(t.Context(), e); err != nil {
			t.Fatal(err)
		}
		mustPlacementReconcile(ctx, t, c, e, gapStart.Add(400*time.Second))
		if got := labelSnapshot(ctx, t, c, key); got["gpu-a"] {
			t.Fatal("an absence entry 100s past its deadline must release on the next observation, not re-grant a window")
		}
	})

	// Within-window case: absence observed at t0-60 with delay 300 →
	// controller out until t0+60 (i.e., 120s into the window) → kept,
	// honoring the pre-gap observation, full window never restarted.
	t.Run("in-window pre-gap entry survives and keeps its window", func(t *testing.T) {
		e := newWlasEngine("engine", 300)
		key := resources.WorkloadPlacementLabel(wlasTestNamespace, "engine")
		pre := nodeNamed("gpu-a")
		pre.Labels[key] = resources.WorkloadLabelValueBase
		c := wlasClient(t, e, pre)

		gapStart := time.Now().Add(-60 * time.Second)
		e.Status.NodeAbsentSince = map[string]metav1.Time{"gpu-a": {Time: gapStart}}
		if err := c.Status().Update(t.Context(), e); err != nil {
			t.Fatal(err)
		}
		delay := mustPlacementReconcile(ctx, t, c, e, gapStart.Add(120*time.Second))
		if got := labelSnapshot(ctx, t, c, key); !got["gpu-a"] {
			t.Fatal("the pre-gap window must still be in force")
		}
		// metav1.Time round-trips through status with second precision,
		// so the remaining window is 180s minus a sub-second remainder.
		if delay > 180*time.Second || delay < 179*time.Second {
			t.Fatalf("the pre-gap window must continue from its own start (~180s remaining), got %v", delay)
		}
	})
}

// NotFound on the add path: a desired node whose object is already gone
// (pod bound to a node that was just deleted before pod GC) must be SKIPPED,
// not returned as an error — otherwise a pre-GC pod wedges the whole
// reconcile (DaemonSet, Services, status) on a stale spec.nodeName.
func TestWorkloadPlacement_DesiredNodeNotFoundIsSkipped(t *testing.T) {
	ctx := context.Background()
	e := newWlasEngine("engine", 300)
	// p1 bound to a healthy node; p2 bound to a node that does not exist.
	c := wlasClient(t, e, nodeNamed("gpu-a"),
		boundPod(wlasTestNamespace, "p1", "gpu-a", map[string]string{"app": "predictor"}),
		boundPod(wlasTestNamespace, "p2", "gpu-ghost", map[string]string{"app": "predictor"}))
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "engine")

	if _, _, err := reconcileWorkloadPlacement(ctx, c, e, time.Now()); err != nil {
		t.Fatalf("missing node object must be skipped, not fail the reconcile: %v", err)
	}
	// The healthy node still got labeled (the bad node didn't block it).
	if got := labelSnapshot(ctx, t, c, key); !got["gpu-a"] {
		t.Fatal("a NotFound on one desired node must not block labeling the others")
	}
	// The ghost node must not have created a status entry.
	if _, ghost := e.Status.NodeAbsentSince["gpu-ghost"]; ghost {
		t.Fatal("skipped node must not carry absence bookkeeping")
	}
	if len(e.Status.WorkloadNodes) != 1 || e.Status.WorkloadNodes[0] != "gpu-a" {
		t.Fatalf("workloadNodes must contain only reachable nodes: %v", e.Status.WorkloadNodes)
	}
}
