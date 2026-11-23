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

package resources

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// wlLabelTestNS is the fixed test namespace for placement label tests.
const wlLabelTestNS = "mlservices"

// testNodeSelectorValue is the fixed user nodeSelector value under test.
const testNodeSelectorValue = "gpu-h100"

// wlLabelTestRelease is the release name shared by the placement-label
// fixtures (per-CR tests use WorkloadPlacementLabel directly).
const wlLabelTestRelease = "engine"

// workloadAwareEngine builds the single Engine fixture for this suite.
func workloadAwareEngine() *lmcachev1alpha1.LMCacheEngine {
	e := &lmcachev1alpha1.LMCacheEngine{
		ObjectMeta: metav1.ObjectMeta{Name: wlLabelTestRelease, Namespace: wlLabelTestNS},
	}
	e.Spec.L1.SizeGB = 8
	e.Spec.WorkloadAwareScheduling = &lmcachev1alpha1.WorkloadAwareSchedulingSpec{
		Enabled: true,
		WorkloadSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{"app": "predictor"},
		},
	}
	return e
}

// TestWorkloadPlacementLabelPinned pins the exact label for one
// namespace/name pair: the hash function and prefix must never silently
// change, or every already-labeled node's key would be orphaned on operator
// upgrade (the DaemonSet would start selecting a label no node carries).
func TestWorkloadPlacementLabelPinned(t *testing.T) {
	key := WorkloadPlacementLabel("mlservices", "my-engine")
	const want = "lmcache.ai/wlas-" + "ab3bc0b93e3a551c" // fnv64a("mlservices/my-engine"), hex
	if key != want {
		t.Fatalf("label key = %q, want %q — hash function changed, labels orphaned on upgrade!", key, want)
	}
	if len(key) > 63 {
		t.Fatalf("label key %q exceeds the 63-character label key budget", key)
	}
}

// TestWorkloadPlacementLabelDistinct asserts per-identity keys: two engines
// must never share one key or they would fight over each other's nodes.
func TestWorkloadPlacementLabelDistinct(t *testing.T) {
	k1 := WorkloadPlacementLabel("mlservices", "engine-a")
	k2 := WorkloadPlacementLabel("mlservices", "engine-b")
	k3 := WorkloadPlacementLabel("other-ns", "engine-a")
	if k1 == k2 || k1 == k3 || k2 == k3 {
		t.Fatalf("expected distinct labels, got %q / %q / %q", k1, k2, k3)
	}
}

// TestWorkloadPlacementLabelLongNameStaysInBudget asserts the 63-char budget
// survives pathological CR names.
func TestWorkloadPlacementLabelLongNameStaysInBudget(t *testing.T) {
	long := "an-extremely-long-engine-name-that-would-overflow-any-name-derived-key-scheme-if-we-used-one-aaaa"
	key := WorkloadPlacementLabel("very-long-namespace-name-to-be-safe", long)
	if len(key) > 63 {
		t.Fatalf("hash-derived key %q exceeded budget (%d)", key, len(key))
	}
}

// TestEffectiveNodeSelector_DisabledPassthrough: with the feature off, the
// builder must pass spec.nodeSelector through untouched.
func TestEffectiveNodeSelector_DisabledPassthrough(t *testing.T) {
	engine := workloadAwareEngine()
	engine.Spec.WorkloadAwareScheduling.Enabled = false
	engine.Spec.NodeSelector = map[string]string{"workload": testNodeSelectorValue}
	got := EffectiveNodeSelector(wlLabelTestNS, engine.Name, &engine.Spec)
	if len(got) != 1 || got["workload"] != testNodeSelectorValue {
		t.Fatalf("disabled passthrough broken: %v", got)
	}
}

// TestEffectiveNodeSelector_NilScheduling absent block behaves as Disabled.
func TestEffectiveNodeSelector_NilScheduling(t *testing.T) {
	engine := workloadAwareEngine()
	engine.Spec.WorkloadAwareScheduling = nil
	engine.Spec.NodeSelector = map[string]string{"workload": testNodeSelectorValue}
	got := EffectiveNodeSelector(wlLabelTestNS, engine.Name, &engine.Spec)
	if len(got) != 1 || got["workload"] != testNodeSelectorValue {
		t.Fatalf("absent-block passthrough broken: %v", got)
	}
}

// TestEffectiveNodeSelector_EnabledMergesAddedKey: enabled merges the user
// schedule with the managed key, later-wins on key conflict in the operator's
// favor (the user cannot pre-place the hash label themselves).
func TestEffectiveNodeSelector_EnabledMergesAddedKey(t *testing.T) {
	engine := workloadAwareEngine()
	engine.Spec.NodeSelector = map[string]string{"workload": "gpu-h100", "gpu-type": "h100"}
	got := EffectiveNodeSelector(wlLabelTestNS, engine.Name, &engine.Spec)
	key := WorkloadPlacementLabel("mlservices", "engine")
	if got["workload"] != testNodeSelectorValue || got["gpu-type"] != "h100" {
		t.Fatalf("user nodeSelector must survive the merge: %v", got)
	}
	if got[key] != WorkloadLabelValueBase {
		t.Fatalf("managed label %s=%q missing from merged selector: %v", key, got[key], got)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 keys in merged selector, got %d: %v", len(got), got)
	}
}

// TestEffectiveNodeSelector_EnabledEmptyUserSelector: even an empty user
// nodeSelector (rare, all-nodes) gains the managed key when enabled.
func TestEffectiveNodeSelector_EnabledEmptyUserSelector(t *testing.T) {
	engine := workloadAwareEngine()
	got := EffectiveNodeSelector(wlLabelTestNS, engine.Name, &engine.Spec)
	key := WorkloadPlacementLabel("mlservices", "engine")
	if got[key] != WorkloadLabelValueBase {
		t.Fatalf("managed label missing on empty user selector: %v", got)
	}
	if len(got) != 1 {
		t.Fatalf("expected only the managed key, got %v", got)
	}
}
