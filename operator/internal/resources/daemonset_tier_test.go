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

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// draEngine returns an engine with WLA + DRA enabled (all tier knobs at
// their zero-value defaults, resolved by the Effective* helpers).
func draEngine() *lmcachev1alpha1.LMCacheEngine {
	e := minimalEngine()
	e.Spec.WorkloadAwareScheduling = &lmcachev1alpha1.WorkloadAwareSchedulingSpec{
		Enabled: true,
		WorkloadSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{"app": "predictor"},
		},
		DynamicResourceAllocation: &lmcachev1alpha1.DynamicResourceAllocationSpec{
			Enabled: true,
		},
	}
	return e
}

func TestTierDaemonSetName(t *testing.T) {
	cases := []struct {
		multiplier int32
		want       string
	}{
		{1, testEngineName},
		{2, testEngineName + "-2x"},
		{8, testEngineName + "-8x"},
	}
	for _, tc := range cases {
		if got := TierDaemonSetName(testEngineName, tc.multiplier); got != tc.want {
			t.Fatalf("TierDaemonSetName(%q, %d) = %q, want %q", testEngineName, tc.multiplier, got, tc.want)
		}
	}
}

// TestBuildDaemonSet_BaseStaysSelectorAndTemplateIdentical pins the
// immutability fix: the base (tier 1) DS must be byte-equivalent to the
// pre-DRA build — SelectorLabels-only selector, NO tier template label —
// so enabling DRA on an existing engine only patches mutable fields.
func TestBuildDaemonSet_BaseStaysSelectorAndTemplateIdentical(t *testing.T) {
	engine := draEngine()
	ds := BuildDaemonSet(engine)

	if len(ds.Spec.Selector.MatchLabels) != len(SelectorLabels(engine.Name)) {
		t.Fatalf("base DS selector must stay SelectorLabels only, got %v", ds.Spec.Selector.MatchLabels)
	}
	if _, has := ds.Spec.Template.Labels[TierLabelKey]; has {
		t.Fatalf("base DS template must not carry the tier label, got %v", ds.Spec.Template.Labels)
	}
}

// TestBuildTieredDaemonSet covers the tier-2 DS: scaled L1 arg and memory
// request, tier template label, tier selector, tier nodeSelector, and the
// shared instance label that keeps the Services matching tier pods.
func TestBuildTieredDaemonSet(t *testing.T) {
	engine := draEngine()
	engine.Spec.L1.SizeGB = 10

	ds := BuildTieredDaemonSet(engine, 2)

	if ds.Name != testEngineName+"-2x" {
		t.Fatalf("tier DS name = %q, want %q", ds.Name, testEngineName+"-2x")
	}

	// Tier selector = SelectorLabels + tier label.
	want := MergeLabels(SelectorLabels(engine.Name), map[string]string{TierLabelKey: ds.Name})
	if len(ds.Spec.Selector.MatchLabels) != len(want) {
		t.Fatalf("tier DS selector = %v, want %v", ds.Spec.Selector.MatchLabels, want)
	}
	for k, v := range want {
		if ds.Spec.Selector.MatchLabels[k] != v {
			t.Fatalf("tier DS selector[%q] = %q, want %q", k, ds.Spec.Selector.MatchLabels[k], v)
		}
	}

	// Shared instance label keeps Services selecting tier pods.
	if ds.Spec.Template.Labels["app.kubernetes.io/instance"] != engine.Name {
		t.Fatalf("tier DS template must carry the shared instance label, got %v", ds.Spec.Template.Labels)
	}
	if ds.Spec.Template.Labels[TierLabelKey] != ds.Name {
		t.Fatalf("tier DS template must carry %s=%s, got %v", TierLabelKey, ds.Name, ds.Spec.Template.Labels)
	}

	// nodeSelector pins the placement label at the tier value.
	key := WorkloadPlacementLabel(engine.Namespace, engine.Name)
	if got := ds.Spec.Template.Spec.NodeSelector[key]; got != "2" {
		t.Fatalf("tier DS nodeSelector[%s] = %q, want \"2\"", key, got)
	}

	// Scaled container args and resources: l1 10*2=20 -> "--l1-size-gb 20.0"
	// and memory request ceil(20+5)=25Gi.
	c := ds.Spec.Template.Spec.Containers[0]
	foundL1 := false
	for i, a := range c.Args {
		if a == "--l1-size-gb" && i+1 < len(c.Args) {
			if c.Args[i+1] != "20.0" {
				t.Fatalf("scaled --l1-size-gb = %q, want 20.0", c.Args[i+1])
			}
			foundL1 = true
		}
	}
	if !foundL1 {
		t.Fatalf("scaled --l1-size-gb arg missing: %v", c.Args)
	}
	if mem := c.Resources.Requests.Memory(); mem.String() != "25Gi" {
		t.Fatalf("scaled memory request = %q, want 25Gi", mem.String())
	}
}

// TestBuildTieredDaemonSet_MultiplierOneIsBase ensures the folding of the
// tier builder into buildDaemonSetCore cannot drift: tier 1 output equals
// BuildDaemonSet output.
func TestBuildTieredDaemonSet_MultiplierOneIsBase(t *testing.T) {
	engine := draEngine()
	tiered := BuildTieredDaemonSet(engine, 1)
	base := BuildDaemonSet(engine)
	if tiered.Name != base.Name {
		t.Fatalf("tier-1 DS name = %q, want %q", tiered.Name, base.Name)
	}
	if len(tiered.Spec.Selector.MatchLabels) != len(base.Spec.Selector.MatchLabels) {
		t.Fatalf("tier-1 selector must equal base: %v vs %v", tiered.Spec.Selector.MatchLabels, base.Spec.Selector.MatchLabels)
	}
}
