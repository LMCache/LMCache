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
)

// --- helpers ---

func ptr[T any](v T) *T { return &v }

// --- SetDefaults tests ---

func TestSetDefaults_LogLevelNil(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 10}}}
	e.SetDefaults()
	if e.Spec.LogLevel == nil || *e.Spec.LogLevel != "INFO" {
		t.Fatalf("expected LogLevel=INFO, got %v", e.Spec.LogLevel)
	}
}

func TestSetDefaults_LogLevelPreserved(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{
		L1:       L1BackendSpec{SizeGB: 10},
		LogLevel: ptr("DEBUG"),
	}}
	e.SetDefaults()
	if *e.Spec.LogLevel != "DEBUG" {
		t.Fatalf("expected LogLevel=DEBUG, got %s", *e.Spec.LogLevel)
	}
}

func TestSetDefaults_NodeSelectorDefaultGPU(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 10}}}
	e.SetDefaults()
	if e.Spec.NodeSelector == nil {
		t.Fatal("expected default NodeSelector, got nil")
	}
	if e.Spec.NodeSelector["nvidia.com/gpu.present"] != "true" {
		t.Fatalf("expected nvidia.com/gpu.present=true, got %v", e.Spec.NodeSelector)
	}
}

func TestSetDefaults_NodeSelectorPreserved(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{
		L1:           L1BackendSpec{SizeGB: 10},
		NodeSelector: map[string]string{"custom": "label"},
	}}
	e.SetDefaults()
	if e.Spec.NodeSelector["custom"] != "label" {
		t.Fatal("expected custom node selector preserved")
	}
	if _, ok := e.Spec.NodeSelector["nvidia.com/gpu.present"]; ok {
		t.Fatal("default should not override user-provided NodeSelector")
	}
}

// --- ValidateSpec tests ---

func TestValidateSpec_ValidMinimal(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 10}}}
	errs := e.ValidateSpec()
	if len(errs) != 0 {
		t.Fatalf("expected no errors, got %v", errs)
	}
}

func TestValidateSpec_SizeGBZero(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: 0}}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if errs[0].Field != "spec.l1.sizeGB" {
		t.Fatalf("expected field spec.l1.sizeGB, got %s", errs[0].Field)
	}
}

func TestValidateSpec_SizeGBNegative(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{L1: L1BackendSpec{SizeGB: -1}}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d", len(errs))
	}
}

func TestValidateSpec_EvictionPolicyInvalid(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{
		L1:       L1BackendSpec{SizeGB: 10},
		Eviction: &EvictionSpec{Policy: ptr("FIFO")},
	}}
	errs := e.ValidateSpec()
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
}

func TestValidateSpec_EvictionPolicyLRU(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{
		L1:       L1BackendSpec{SizeGB: 10},
		Eviction: &EvictionSpec{Policy: ptr("LRU")},
	}}
	errs := e.ValidateSpec()
	if len(errs) != 0 {
		t.Fatalf("expected no errors, got %v", errs)
	}
}

func TestValidateSpec_TriggerWatermarkBounds(t *testing.T) {
	tests := []struct {
		name    string
		value   float64
		wantErr bool
	}{
		{"zero", 0.0, true},
		{"negative", -0.1, true},
		{"valid_low", 0.01, false},
		{"valid_mid", 0.5, false},
		{"valid_one", 1.0, false},
		{"above_one", 1.1, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &LMCacheEngine{Spec: LMCacheEngineSpec{
				L1:       L1BackendSpec{SizeGB: 10},
				Eviction: &EvictionSpec{TriggerWatermark: ptr(tt.value)},
			}}
			errs := e.ValidateSpec()
			if tt.wantErr && len(errs) == 0 {
				t.Fatal("expected error, got none")
			}
			if !tt.wantErr && len(errs) != 0 {
				t.Fatalf("expected no error, got %v", errs)
			}
		})
	}
}

func TestValidateSpec_EvictionRatioBounds(t *testing.T) {
	tests := []struct {
		name    string
		value   float64
		wantErr bool
	}{
		{"zero", 0.0, true},
		{"negative", -0.5, true},
		{"valid_low", 0.1, false},
		{"valid_one", 1.0, false},
		{"above_one", 1.5, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &LMCacheEngine{Spec: LMCacheEngineSpec{
				L1:       L1BackendSpec{SizeGB: 10},
				Eviction: &EvictionSpec{EvictionRatio: ptr(tt.value)},
			}}
			errs := e.ValidateSpec()
			if tt.wantErr && len(errs) == 0 {
				t.Fatal("expected error, got none")
			}
			if !tt.wantErr && len(errs) != 0 {
				t.Fatalf("expected no error, got %v", errs)
			}
		})
	}
}

func TestValidateSpec_ServerPort(t *testing.T) {
	tests := []struct {
		name    string
		port    int32
		wantErr bool
	}{
		{"below_min", 80, true},
		{"min_valid", 1024, false},
		{"max_valid", 65535, false},
		{"above_max", 65536, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &LMCacheEngine{Spec: LMCacheEngineSpec{
				L1:     L1BackendSpec{SizeGB: 10},
				Server: &ServerSpec{Port: ptr(tt.port)},
			}}
			errs := e.ValidateSpec()
			if tt.wantErr && len(errs) == 0 {
				t.Fatal("expected error, got none")
			}
			if !tt.wantErr && len(errs) != 0 {
				t.Fatalf("expected no error, got %v", errs)
			}
		})
	}
}

func TestValidateSpec_MultipleErrors(t *testing.T) {
	e := &LMCacheEngine{Spec: LMCacheEngineSpec{
		L1:     L1BackendSpec{SizeGB: 0},
		Server: &ServerSpec{Port: ptr(int32(80))},
		Eviction: &EvictionSpec{
			Policy:           ptr("FIFO"),
			TriggerWatermark: ptr(0.0),
			EvictionRatio:    ptr(0.0),
		},
	}}
	errs := e.ValidateSpec()
	// sizeGB, port, policy, watermark, ratio = 5 errors
	if len(errs) != 5 {
		t.Fatalf("expected 5 errors, got %d: %v", len(errs), errs)
	}
}
