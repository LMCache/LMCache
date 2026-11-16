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

package v1alpha1

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func enabledWAS() *WorkloadAwareSchedulingSpec {
	return &WorkloadAwareSchedulingSpec{
		Enabled: true,
		WorkloadSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{"app": "predictor"},
		},
	}
}

func TestValidate_WorkloadAwareSchedulingDisabledOrAbsent(t *testing.T) {
	// Absent block, disabled block, and disabled block with broken selector:
	// all must pass — the contract only applies when the feature is on.
	for name, w := range map[string]*WorkloadAwareSchedulingSpec{
		"absent":  nil,
		"off":     {Enabled: false},
		"off-bad": {Enabled: false, WorkloadSelector: &metav1.LabelSelector{}},
	} {
		t.Run(name, func(t *testing.T) {
			e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 8}}}
			e.Spec.WorkloadAwareScheduling = w
			if errs := e.ValidateSpec(); len(errs) != 0 {
				t.Fatalf("expected clean validation, got %v", errs)
			}
		})
	}
}

func TestValidate_WorkloadAwareSchedulingRequiresSelector(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 8}}}
	e.Spec.WorkloadAwareScheduling = &WorkloadAwareSchedulingSpec{Enabled: true}
	errs := e.ValidateSpec()
	if len(errs) == 0 {
		t.Fatal("expected validation error for enabled without selector")
	}
}

func TestValidate_WorkloadAwareSchedulingRejectsEmptySelector(t *testing.T) {
	// The match-everything footgun: an empty LabelSelector would label every
	// node running any pod in the namespace.
	for name, sel := range map[string]*metav1.LabelSelector{
		"empty-object":      {},
		"empty-labels":      {MatchLabels: map[string]string{}},
		"empty-expressions": {MatchExpressions: []metav1.LabelSelectorRequirement{}},
	} {
		t.Run(name, func(t *testing.T) {
			e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 8}}}
			e.Spec.WorkloadAwareScheduling = &WorkloadAwareSchedulingSpec{Enabled: true, WorkloadSelector: sel}
			if errs := e.ValidateSpec(); len(errs) == 0 {
				t.Fatal("expected validation error for empty selector")
			}
		})
	}
}

func TestValidate_WorkloadAwareSchedulingAcceptsExpressions(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 8}}}
	e.Spec.WorkloadAwareScheduling = &WorkloadAwareSchedulingSpec{
		Enabled: true,
		WorkloadSelector: &metav1.LabelSelector{
			MatchExpressions: []metav1.LabelSelectorRequirement{
				{Key: "component", Operator: metav1.LabelSelectorOpIn, Values: []string{"predictor"}},
				{Key: "serving.kserve.io/inferenceservice", Operator: metav1.LabelSelectorOpExists},
			},
		},
	}
	if errs := e.ValidateSpec(); len(errs) != 0 {
		t.Fatalf("expected clean validation, got %v", errs)
	}
}

func TestValidate_WorkloadAwareSchedulingRejectsMalformedSelectors(t *testing.T) {
	cases := map[string]*metav1.LabelSelector{
		"unknown-operator": {MatchExpressions: []metav1.LabelSelectorRequirement{
			{Key: "x", Operator: "Bogus", Values: []string{"y"}},
		}},
		"in-needs-values": {MatchExpressions: []metav1.LabelSelectorRequirement{
			{Key: "x", Operator: metav1.LabelSelectorOpIn},
		}},
		"exists-forbids-values": {MatchExpressions: []metav1.LabelSelectorRequirement{
			{Key: "x", Operator: metav1.LabelSelectorOpExists, Values: []string{"y"}},
		}},
		"invalid-label-key": {MatchLabels: map[string]string{"not a valid key": "v"}},
		"invalid-label-value": {MatchExpressions: []metav1.LabelSelectorRequirement{
			{Key: "x", Operator: metav1.LabelSelectorOpIn, Values: []string{"not\nvalid"}},
		}},
	}
	for name, sel := range cases {
		t.Run(name, func(t *testing.T) {
			e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 8}}}
			e.Spec.WorkloadAwareScheduling = &WorkloadAwareSchedulingSpec{Enabled: true, WorkloadSelector: sel}
			if errs := e.ValidateSpec(); len(errs) == 0 {
				t.Fatalf("expected a syntax validation error for %s", name)
			}
		})
	}
}

func TestSetDefaults_WorkloadAwareReleaseDelay(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 8}}}
	e.Spec.WorkloadAwareScheduling = enabledWAS()
	e.SetDefaults()

	if e.Spec.WorkloadAwareScheduling.ReleaseDelaySeconds == nil {
		t.Fatal("expected releaseDelaySeconds to be defaulted")
	}
	if got := *e.Spec.WorkloadAwareScheduling.ReleaseDelaySeconds; got != defaultReleaseDelaySeconds {
		t.Fatalf("expected default %d, got %d", defaultReleaseDelaySeconds, got)
	}
	if got := e.WorkloadReleaseDelay().Seconds(); got != float64(defaultReleaseDelaySeconds) {
		t.Fatalf("expected WorkloadReleaseDelay %ds, got %v", defaultReleaseDelaySeconds, got)
	}
}

func TestWorkloadReleaseDelay_Disabled(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 8}}}
	if got := e.WorkloadReleaseDelay(); got != 0 {
		t.Fatalf("expected 0 delay when disabled, got %v", got)
	}
	if e.WorkloadSchedulingEnabled() {
		t.Fatal("expected WorkloadSchedulingEnabled false")
	}
}

func enabledDRA() *DynamicResourceAllocationSpec {
	return &DynamicResourceAllocationSpec{Enabled: true}
}

func TestValidate_DRARequiresWLA(t *testing.T) {
	e := &LMCacheEngine{}
	e.Spec.WorkloadAwareScheduling = enabledWAS()
	e.Spec.WorkloadAwareScheduling.Enabled = false
	e.Spec.WorkloadAwareScheduling.DynamicResourceAllocation = enabledDRA()
	errs := e.ValidateSpec()
	if len(errs) == 0 {
		t.Fatal("DRA without WLA must be rejected")
	}
}

func TestValidate_DRARejectsResourceOverrides(t *testing.T) {
	e := &LMCacheEngine{}
	e.Spec.WorkloadAwareScheduling = enabledWAS()
	e.Spec.WorkloadAwareScheduling.DynamicResourceAllocation = enabledDRA()
	e.Spec.ResourceOverrides = &corev1.ResourceRequirements{}
	errs := e.ValidateSpec()
	found := false
	for _, err := range errs {
		if err.Field == "spec.resourceOverrides" {
			found = true
		}
	}
	if !found {
		t.Fatalf("resourceOverrides + DRA must be rejected at spec.resourceOverrides, got %v", errs)
	}
}

func TestValidate_DRARejectsUnsupportedVendor(t *testing.T) {
	e := &LMCacheEngine{}
	e.Spec.WorkloadAwareScheduling = enabledWAS()
	e.Spec.WorkloadAwareScheduling.DynamicResourceAllocation = enabledDRA()
	vendor := "habana"
	e.Spec.GPUVendor = &vendor
	errs := e.ValidateSpec()
	if len(errs) == 0 {
		t.Fatal("DRA with a vendor that has no known GPU resource name must be rejected")
	}
}

func TestValidate_DRARejectsTooLongEngineName(t *testing.T) {
	e := &LMCacheEngine{}
	e.Name = string(make([]byte, 63)) // 63 chars of NUL; only length matters
	e.Spec.WorkloadAwareScheduling = enabledWAS()
	e.Spec.WorkloadAwareScheduling.DynamicResourceAllocation = enabledDRA()
	errs := e.ValidateSpec()
	if len(errs) == 0 {
		t.Fatal("an engine name leaving no room for the -<N>x tier suffix must be rejected")
	}
}

func TestValidate_DRAValidEnginePasses(t *testing.T) {
	e := &LMCacheEngine{}
	e.Name = "s1"
	e.Spec.WorkloadAwareScheduling = enabledWAS()
	e.Spec.WorkloadAwareScheduling.DynamicResourceAllocation = enabledDRA()
	e.Spec.L1.SizeGB = 10
	errs := e.ValidateSpec()
	if len(errs) != 0 {
		t.Fatalf("valid DRA engine must pass, got %v", errs)
	}
}

func TestDRAEffectiveDefaults(t *testing.T) {
	dra := &DynamicResourceAllocationSpec{Enabled: true}
	if dra.EffectiveDRAGPUsPerUnit() != 1 {
		t.Fatal("gpusPerUnit default must be 1")
	}
	if dra.EffectiveDRAMaxMultiplier() != 8 {
		t.Fatal("maxMultiplier default must be 8")
	}
	if dra.EffectiveDRATierChangeDelay().Seconds() != 300 {
		t.Fatal("tierChangeDelaySeconds default must be 300s")
	}
	g, m, d := int32(4), int32(3), int32(0)
	dra = &DynamicResourceAllocationSpec{Enabled: true, GPUsPerUnit: &g, MaxMultiplier: &m, TierChangeDelaySeconds: &d}
	if dra.EffectiveDRAGPUsPerUnit() != 4 || dra.EffectiveDRAMaxMultiplier() != 3 || dra.EffectiveDRATierChangeDelay() != 0 {
		t.Fatal("explicit DRA values must win over defaults")
	}
}
