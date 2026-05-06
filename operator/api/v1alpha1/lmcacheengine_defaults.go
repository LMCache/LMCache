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

// SetDefaults applies defaults that cannot be expressed purely via kubebuilder markers.
func (e *LMCacheEngine) SetDefaults() {
	spec := &e.Spec

	// Default logLevel to INFO if unset (belt-and-suspenders with kubebuilder default).
	if spec.LogLevel == nil {
		info := "INFO"
		spec.LogLevel = &info
	}

	// Default gpuVendor to nvidia (belt-and-suspenders with kubebuilder default).
	if spec.GPUVendor == nil {
		v := GPUVendorNvidia
		spec.GPUVendor = &v
	}

	// Default nodeSelector only for the nvidia vendor: NVIDIA exposes a
	// universal nvidia.com/gpu.present label via the GPU Operator / device
	// plugin. AMD has no universal equivalent (varies per platform — AMD GPU
	// Operator uses feature.node.kubernetes.io/amd-gpu, managed platforms use
	// vendor-specific labels), so AMD users supply nodeSelector explicitly.
	if spec.NodeSelector == nil && *spec.GPUVendor == GPUVendorNvidia {
		spec.NodeSelector = map[string]string{
			"nvidia.com/gpu.present": "true",
		}
	}
}
