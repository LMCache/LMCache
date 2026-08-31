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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

func pdSpec(port int32) *lmcachev1alpha1.PDSpec {
	return &lmcachev1alpha1.PDSpec{
		NixlSideChannelPort: &port,
	}
}

var _ = Describe("BuildPDEnv", func() {
	It("injects both env vars into an empty list", func() {
		out := BuildPDEnv(nil, pdSpec(5558))

		Expect(out).To(HaveLen(2))

		var host, port *corev1.EnvVar
		for i := range out {
			switch out[i].Name {
			case nixlSideChannelHostEnv:
				host = &out[i]
			case nixlSideChannelPortEnv:
				port = &out[i]
			}
		}

		Expect(host).NotTo(BeNil(), "VLLM_NIXL_SIDE_CHANNEL_HOST must be present")
		Expect(host.ValueFrom).NotTo(BeNil())
		Expect(host.ValueFrom.FieldRef).NotTo(BeNil())
		Expect(host.ValueFrom.FieldRef.FieldPath).To(Equal("status.podIP"),
			"host env var must use pod IP; NIXL listens in the pod network namespace")

		Expect(port).NotTo(BeNil(), "VLLM_NIXL_SIDE_CHANNEL_PORT must be present")
		Expect(port.Value).To(Equal("5558"))
	})

	It("uses the default port when NixlSideChannelPort is nil", func() {
		pd := &lmcachev1alpha1.PDSpec{}
		out := BuildPDEnv(nil, pd)

		var portVar *corev1.EnvVar
		for i := range out {
			if out[i].Name == nixlSideChannelPortEnv {
				portVar = &out[i]
			}
		}
		Expect(portVar).NotTo(BeNil())
		Expect(portVar.Value).To(Equal("5558"), "default port must be 5558")
	})

	It("does not overwrite a pre-existing VLLM_NIXL_SIDE_CHANNEL_HOST", func() {
		existing := []corev1.EnvVar{
			{Name: nixlSideChannelHostEnv, Value: "192.168.1.1"},
		}
		out := BuildPDEnv(existing, pdSpec(5558))

		var count int
		for _, e := range out {
			if e.Name == nixlSideChannelHostEnv {
				count++
				Expect(e.Value).To(Equal("192.168.1.1"), "user-set host must be preserved")
			}
		}
		Expect(count).To(Equal(1), "host env var must not be duplicated")
	})

	It("does not overwrite a pre-existing VLLM_NIXL_SIDE_CHANNEL_PORT", func() {
		existing := []corev1.EnvVar{
			{Name: nixlSideChannelPortEnv, Value: "9999"},
		}
		out := BuildPDEnv(existing, pdSpec(5558))

		var count int
		for _, e := range out {
			if e.Name == nixlSideChannelPortEnv {
				count++
				Expect(e.Value).To(Equal("9999"), "user-set port must be preserved")
			}
		}
		Expect(count).To(Equal(1), "port env var must not be duplicated")
	})

	It("preserves existing env vars and appends only missing ones", func() {
		existing := []corev1.EnvVar{
			{Name: "OTHER_VAR", Value: "other"},
		}
		out := BuildPDEnv(existing, pdSpec(5557))

		Expect(out).To(HaveLen(3))
		Expect(out[0].Name).To(Equal("OTHER_VAR"), "existing vars must come first")
	})

	It("does not mutate the input slice", func() {
		existing := []corev1.EnvVar{
			{Name: "X", Value: "1"},
		}
		original := make([]corev1.EnvVar, len(existing))
		copy(original, existing)

		BuildPDEnv(existing, pdSpec(5558))

		Expect(existing).To(Equal(original), "input slice must not be mutated")
	})
})
