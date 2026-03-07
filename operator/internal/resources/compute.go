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
	"encoding/json"
	"fmt"
	"math"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// ComputeResources returns resource requirements, either from overrides or auto-computed from L1 size.
func ComputeResources(spec *lmcachev1alpha1.LMCacheEngineSpec) corev1.ResourceRequirements {
	if spec.ResourceOverrides != nil {
		return *spec.ResourceOverrides
	}

	// memoryRequest = ceil(l1.sizeGB + 5) Gi
	memReqGi := int64(math.Ceil(spec.L1.SizeGB + 5))
	// memoryLimit = ceil(memoryRequest * 1.5) Gi
	memLimGi := int64(math.Ceil(float64(memReqGi) * 1.5))

	return corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("4"),
			corev1.ResourceMemory: resource.MustParse(fmt.Sprintf("%dGi", memReqGi)),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceMemory: resource.MustParse(fmt.Sprintf("%dGi", memLimGi)),
		},
	}
}

// BuildContainerArgs maps CRD spec fields to server.py CLI flags.
func BuildContainerArgs(spec *lmcachev1alpha1.LMCacheEngineSpec) []string {
	args := []string{
		"--host", "0.0.0.0",
		"--port", fmt.Sprintf("%d", derefInt32(getServerPort(spec), 5555)),
		"--l1-size-gb", fmt.Sprintf("%.1f", spec.L1.SizeGB),
		"--chunk-size", fmt.Sprintf("%d", getChunkSize(spec)),
		"--max-workers", fmt.Sprintf("%d", getMaxWorkers(spec)),
		"--hash-algorithm", getHashAlgorithm(spec),
	}

	// Eviction args
	evPolicy := "LRU"
	evTrigger := 0.8
	evRatio := 0.2
	if spec.Eviction != nil {
		evPolicy = derefString(spec.Eviction.Policy, "LRU")
		evTrigger = derefFloat64(spec.Eviction.TriggerWatermark, 0.8)
		evRatio = derefFloat64(spec.Eviction.EvictionRatio, 0.2)
	}
	args = append(args,
		"--eviction-policy", evPolicy,
		"--eviction-trigger-watermark", fmt.Sprintf("%.2f", evTrigger),
		"--eviction-ratio", fmt.Sprintf("%.2f", evRatio),
	)

	// Prometheus args
	promEnabled := true
	if spec.Prometheus != nil {
		promEnabled = derefBool(spec.Prometheus.Enabled, true)
	}
	if !promEnabled {
		args = append(args, "--disable-prometheus")
	} else {
		promPort := int32(9090)
		if spec.Prometheus != nil {
			promPort = derefInt32(spec.Prometheus.Port, 9090)
		}
		args = append(args, "--prometheus-port", fmt.Sprintf("%d", promPort))
	}

	// L2 backends
	for _, backend := range spec.L2Backends {
		l2JSON := mergeL2BackendToJSON(backend)
		args = append(args, "--l2-adapter", l2JSON)
	}

	// User-supplied extra args (appended last so they can override defaults)
	args = append(args, spec.ExtraArgs...)

	return args
}

// mergeL2BackendToJSON flattens {type, config} into the CLI-expected flat JSON.
func mergeL2BackendToJSON(backend lmcachev1alpha1.L2BackendSpec) string {
	flat := make(map[string]any)
	flat["type"] = backend.Type
	for k, v := range backend.Config {
		var parsed any
		if err := json.Unmarshal(v.Raw, &parsed); err != nil {
			flat[k] = string(v.Raw)
		} else {
			flat[k] = parsed
		}
	}
	b, _ := json.Marshal(flat)
	return string(b)
}

func getServerPort(spec *lmcachev1alpha1.LMCacheEngineSpec) *int32 {
	if spec.Server != nil {
		return spec.Server.Port
	}
	return nil
}

func getChunkSize(spec *lmcachev1alpha1.LMCacheEngineSpec) int32 {
	if spec.Server != nil {
		return derefInt32(spec.Server.ChunkSize, 256)
	}
	return 256
}

func getMaxWorkers(spec *lmcachev1alpha1.LMCacheEngineSpec) int32 {
	if spec.Server != nil {
		return derefInt32(spec.Server.MaxWorkers, 1)
	}
	return 1
}

func getHashAlgorithm(spec *lmcachev1alpha1.LMCacheEngineSpec) string {
	if spec.Server != nil {
		return derefString(spec.Server.HashAlgorithm, "blake3")
	}
	return "blake3"
}
