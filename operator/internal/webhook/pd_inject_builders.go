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

package webhook

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

const (
	nixlSideChannelHostEnv     = "VLLM_NIXL_SIDE_CHANNEL_HOST"
	nixlSideChannelPortEnv     = "VLLM_NIXL_SIDE_CHANNEL_PORT"
	nixlDefaultSideChannelPort = int32(5558)
)

// BuildPDEnv injects the NIXL side-channel env vars required for PD
// disaggregation into the target vLLM container's env list.
//
//   - VLLM_NIXL_SIDE_CHANNEL_HOST is populated via the Kubernetes downward API
//     (status.podIP). The NIXL agent listens in the pod's own network namespace,
//     so the pod IP is the correct address for peers to connect to. No hostPort
//     or hostNetwork is required as long as pod IPs are routable between nodes
//     (standard CNI behaviour).
//   - VLLM_NIXL_SIDE_CHANNEL_PORT is derived from pd.NixlSideChannelPort
//     (default 5558).
//
// Existing values (set by the user) are respected and left untouched.
// Returns a new env list; the input is not mutated.
func BuildPDEnv(existing []corev1.EnvVar, pd *lmcachev1alpha1.PDSpec) []corev1.EnvVar {
	hasHost := false
	hasPort := false
	for _, e := range existing {
		switch e.Name {
		case nixlSideChannelHostEnv:
			hasHost = true
		case nixlSideChannelPortEnv:
			hasPort = true
		}
	}

	portVal := nixlDefaultSideChannelPort
	if pd.NixlSideChannelPort != nil {
		portVal = *pd.NixlSideChannelPort
	}

	out := make([]corev1.EnvVar, 0, len(existing)+2)
	out = append(out, existing...)

	if !hasHost {
		out = append(out, corev1.EnvVar{
			Name: nixlSideChannelHostEnv,
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "status.podIP",
				},
			},
		})
	}
	if !hasPort {
		out = append(out, corev1.EnvVar{
			Name:  nixlSideChannelPortEnv,
			Value: fmt.Sprintf("%d", portVal),
		})
	}
	return out
}
