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

import "time"

// defaultLogLevel is the log level applied when spec.logLevel is unset (it
// mirrors the kubebuilder default on both engine kinds).
const defaultLogLevel = "INFO"

// labelValueTrue is the string value of boolean-style node-selector labels
// (e.g. nvidia.com/gpu.present: "true").
const labelValueTrue = "true"

// SetDefaults applies defaults that cannot be expressed purely via kubebuilder markers.
func (e *LMCacheEngine) SetDefaults() {
	spec := &e.Spec

	// Default logLevel to INFO if unset (belt-and-suspenders with kubebuilder default).
	if spec.LogLevel == nil {
		info := defaultLogLevel
		spec.LogLevel = &info
	}

	if spec.GPUVendor == nil {
		v := GPUVendorNvidia
		spec.GPUVendor = &v
	}

	if spec.NodeSelector == nil && *spec.GPUVendor == GPUVendorNvidia {
		spec.NodeSelector = map[string]string{
			"nvidia.com/gpu.present": labelValueTrue,
		}
	}

	// Workload-aware scheduling: default the release delay so users enabling
	// the feature don't have to think about rollout churn on day one. The
	// kubebuilder schema default covers apply-time; this covers callers
	// constructing objects in memory (controller paths, tests).
	if w := spec.WorkloadAwareScheduling; w != nil && w.Enabled && w.ReleaseDelaySeconds == nil {
		d := int32(defaultReleaseDelaySeconds)
		w.ReleaseDelaySeconds = &d
	}
}

// defaultReleaseDelaySeconds keeps the engine (and its warm KV cache) across
// typical same-node rolling updates while still cleaning up promptly after a
// genuine workload departure.
const defaultReleaseDelaySeconds = 300

// WorkloadReleaseDelay returns the effective workload-aware release delay
// for an enabled engine, zero when the feature is disabled. A nil
// ReleaseDelaySeconds falls back to defaultReleaseDelaySeconds rather than
// silently collapsing to 0: API-created objects get the CRD default, but an
// in-memory path that skipped SetDefaults must not land on the
// "always rebuild cold" behavior by accident.
func (e *LMCacheEngine) WorkloadReleaseDelay() time.Duration {
	if w := e.Spec.WorkloadAwareScheduling; w != nil && w.Enabled {
		if w.ReleaseDelaySeconds == nil {
			return time.Duration(defaultReleaseDelaySeconds) * time.Second
		}
		return time.Duration(*w.ReleaseDelaySeconds) * time.Second
	}
	return 0
}

// DRA defaults mirror the kubebuilder markers (1 / 8 / 300) for in-memory
// objects that skipped SetDefaults or CRD-defaulting.
const (
	defaultDRAGPUsPerUnit         = 1
	defaultDRAMaxMultiplier       = 8
	defaultDRATierChangeDelaySecs = 300
)

// DynamicResourceAllocationEnabled reports whether dynamic resource
// allocation (per-node tiered DaemonSets) is on. Implies
// WorkloadSchedulingEnabled: validation rejects the other combination.
func (e *LMCacheEngine) DynamicResourceAllocationEnabled() bool {
	if w := e.Spec.WorkloadAwareScheduling; w != nil && w.Enabled {
		return w.DynamicResourceAllocation != nil && w.DynamicResourceAllocation.Enabled
	}
	return false
}

// EffectiveDRAGPUsPerUnit returns gpusPerUnit with its default applied.
func (s *DynamicResourceAllocationSpec) EffectiveDRAGPUsPerUnit() int32 {
	if s.GPUsPerUnit == nil {
		return defaultDRAGPUsPerUnit
	}
	return *s.GPUsPerUnit
}

// EffectiveDRAMaxMultiplier returns maxMultiplier with its default applied.
func (s *DynamicResourceAllocationSpec) EffectiveDRAMaxMultiplier() int32 {
	if s.MaxMultiplier == nil {
		return defaultDRAMaxMultiplier
	}
	return *s.MaxMultiplier
}

// EffectiveDRATierChangeDelay returns tierChangeDelaySeconds as a duration,
// with its default applied.
func (s *DynamicResourceAllocationSpec) EffectiveDRATierChangeDelay() time.Duration {
	if s.TierChangeDelaySeconds == nil {
		return time.Duration(defaultDRATierChangeDelaySecs) * time.Second
	}
	return time.Duration(*s.TierChangeDelaySeconds) * time.Second
}

// WorkloadSchedulingEnabled reports whether workload-aware scheduling is on.
func (e *LMCacheEngine) WorkloadSchedulingEnabled() bool {
	w := e.Spec.WorkloadAwareScheduling
	return w != nil && w.Enabled
}

// IsolatedIPCEnabled resolves spec.isolatedIPC: an explicit value wins;
// unset means auto — enabled for gpuVendor "nvidia" (the default vendor),
// disabled for "amd", whose ROCm stack has no isolated-IPC backends.
// Callers must treat true as overriding spec.hostIPC and the /dev/shm
// sharing (see the isolatedIPC field documentation).
func (s *LMCacheEngineSpec) IsolatedIPCEnabled() bool {
	if s.IsolatedIPC != nil {
		return *s.IsolatedIPC
	}
	return s.GPUVendor == nil || *s.GPUVendor == GPUVendorNvidia
}
