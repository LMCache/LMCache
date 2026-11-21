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
	"testing"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/event"

	"github.com/LMCache/LMCache/internal/resources"
)

// podUpdate builds an UpdateEvent's old/new pair.
func podUpdate(old, new *corev1.Pod) event.UpdateEvent {
	return event.UpdateEvent{ObjectOld: old, ObjectNew: new}
}

// nodeUpdate builds an UpdateEvent's old/new pair for nodes.
func nodeUpdate(old, new *corev1.Node) event.UpdateEvent {
	return event.UpdateEvent{ObjectOld: old, ObjectNew: new}
}

// TestPodPredicate_TerminalPhaseTriggersReconcile is the regression test
// for the stranded-engine leak: a pod moving to Succeeded/Failed changes
// no watched field except Status.Phase, so the predicate must fire on it —
// otherwise a finished Job pod (lingering on ttlSecondsAfterFinished)
// keeps its node's label and engine forever, and inflates its DRA tier.
func TestPodPredicate_TerminalPhaseTriggersReconcile(t *testing.T) {
	p := workloadPlacementPodPredicate()
	base := boundPod(wlasTestNamespace, "p1", "gpu-a", map[string]string{"app": "predictor"})
	base.Status.Phase = corev1.PodRunning

	done := base.DeepCopy()
	done.Status.Phase = corev1.PodSucceeded
	if !p.Update(podUpdate(base, done)) {
		t.Fatal("phase-only update to Succeeded must trigger a reconcile")
	}

	failed := base.DeepCopy()
	failed.Status.Phase = corev1.PodFailed
	if !p.Update(podUpdate(base, failed)) {
		t.Fatal("phase-only update to Failed must trigger a reconcile")
	}

	// A non-terminal phase change (Pending -> Running) alone does not
	// count: placement reads binding, deletion, labels and terminality.
	running := base.DeepCopy()
	running.Status.Phase = corev1.PodRunning
	if p.Update(podUpdate(base, running)) {
		t.Fatal("same-phase update must not trigger a reconcile")
	}
}

// TestNodePredicate_OnlyWlasLabelChangesFire: kubelet status churn on a
// labeled node must not fan out to every engine reconcile — only a change
// in the lmcache.ai/wlas-* label subset may.
func TestNodePredicate_OnlyWlasLabelChangesFire(t *testing.T) {
	p := workloadPlacementNodePredicate()
	key := resources.WorkloadPlacementLabel(wlasTestNamespace, "engine")
	labeled := nodeNamed("gpu-a")
	labeled.Labels[key] = "1"

	churn := labeled.DeepCopy()
	churn.Status.Phase = "Ready" // irrelevant field changes
	churn.Labels["node.kubernetes.io/other"] = "x"
	if p.Update(nodeUpdate(labeled, churn)) {
		t.Fatal("non-wlas churn on a labeled node must not trigger a reconcile")
	}

	retiered := labeled.DeepCopy()
	retiered.Labels[key] = "2"
	if !p.Update(nodeUpdate(labeled, retiered)) {
		t.Fatal("wlas label value change must trigger a reconcile")
	}

	stripped := labeled.DeepCopy()
	delete(stripped.Labels, key)
	if !p.Update(nodeUpdate(labeled, stripped)) {
		t.Fatal("wlas label removal must trigger a reconcile")
	}

	// A node gaining the label from nothing.
	gained := stripped.DeepCopy()
	gained.Labels[key] = "1"
	if !p.Update(nodeUpdate(stripped, gained)) {
		t.Fatal("wlas label addition must trigger a reconcile")
	}

	// An entirely unrelated node's churn stays silent.
	plain := nodeNamed("gpu-b")
	if p.Update(nodeUpdate(plain, plain.DeepCopy())) {
		t.Fatal("unrelated node update must not trigger a reconcile")
	}
}
