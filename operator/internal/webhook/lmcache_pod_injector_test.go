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
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

const (
	// testContainerVLLM is the default target container name in these specs.
	testContainerVLLM = "vllm"
	// testUnknownContainer names a container that does not exist on the pod.
	testUnknownContainer = "does-not-exist"
)

// newLMCacheInjector returns an LMCachePodInjector backed by a fake client,
// seeded with the test engine's connection ConfigMap when seedConn is true. The
// LMCacheEngine CR itself is not seeded — the injector reads only the ConfigMap.
func newLMCacheInjector(seedConn bool) *LMCachePodInjector {
	scheme := newTestScheme()
	builder := fake.NewClientBuilder().WithScheme(scheme)
	if seedConn {
		engine := &lmcachev1alpha1.LMCacheEngine{
			ObjectMeta: metav1.ObjectMeta{Name: testEngineName, Namespace: testNamespace},
		}
		builder = builder.WithRuntimeObjects(runtime.Object(resources.BuildConnectionConfigMap(engine)))
	}
	return &LMCachePodInjector{
		Client:  builder.Build(),
		Decoder: admission.NewDecoder(scheme),
	}
}

// lmcachePod returns a minimal args-only vLLM pod (no command override) opted in
// to LMCache injection via annotation. mutate may further customize it.
func lmcachePod(mutate func(*corev1.Pod)) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testPodName,
			Namespace: testNamespace,
			Annotations: map[string]string{
				LMCacheAnnotationEngine: testEngineName,
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  testContainerVLLM,
					Image: "vllm/vllm-openai:latest",
					Args:  []string{"--model", "Qwen/Qwen2.5-0.5B"},
				},
			},
		},
	}
	if mutate != nil {
		mutate(pod)
	}
	return pod
}

var _ = Describe("LMCachePodInjector", func() {
	ctx := context.Background()

	Describe("full injection", func() {
		It("injects hostIPC, --kv-transfer-config, and PYTHONHASHSEED for an opted-in pod", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(nil)

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			By("hostIPC")
			Expect(out.Spec.HostIPC).To(BeTrue())

			c := findContainer(out, testContainerVLLM)
			Expect(c).NotTo(BeNil())

			By("--kv-transfer-config carries LMCacheMPConnector + tcp:// host")
			kv := argsFlagValue(c.Args, lmcFlagKVTransferConfig)
			Expect(kv).NotTo(BeEmpty())
			Expect(kv).To(ContainSubstring("LMCacheMPConnector"))
			Expect(kv).To(ContainSubstring(testSvcHost))

			By("the original --model arg is preserved")
			Expect(argsFlagValue(c.Args, "--model")).To(Equal("Qwen/Qwen2.5-0.5B"))

			By("PYTHONHASHSEED set to 0")
			Expect(envValue(c, pythonHashSeedEnvName)).To(Equal(pythonHashSeedValue))

			By("idempotency annotation stamped, no skip reason")
			Expect(out.Annotations[LMCacheAnnotationInjected]).To(Equal(valueTrue))
			Expect(out.Annotations).NotTo(HaveKey(LMCacheAnnotationSkipReason))
		})

		It("respects a user-set PYTHONHASHSEED", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Env = []corev1.EnvVar{
					{Name: pythonHashSeedEnvName, Value: "42"},
				}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)
			c := findContainer(out, testContainerVLLM)

			Expect(envValue(c, pythonHashSeedEnvName)).To(Equal("42"))
		})
	})

	Describe("gating", func() {
		It("allows a pod with no engine annotation unchanged", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				delete(p.Annotations, LMCacheAnnotationEngine)
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).To(BeEmpty())
		})

		It("allows an already-injected pod as a no-op", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Annotations[LMCacheAnnotationInjected] = valueTrue
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).To(BeEmpty())
		})

		It("skips + stamps engine-not-found when the connection ConfigMap is absent", func() {
			injector := newLMCacheInjector(false) // no connection ConfigMap seeded
			pod := lmcachePod(nil)

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Annotations[LMCacheAnnotationSkipReason]).To(Equal(SkipReasonEngineNotFound))
			Expect(out.Annotations).NotTo(HaveKey(LMCacheAnnotationInjected))
			Expect(out.Spec.HostIPC).To(BeFalse())
		})

		It("skips + stamps command-override when the target container overrides command", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exec vllm serve"}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Annotations[LMCacheAnnotationSkipReason]).To(Equal(SkipReasonCommandOverride))
			Expect(out.Annotations).NotTo(HaveKey(LMCacheAnnotationInjected))
			Expect(out.Spec.HostIPC).To(BeFalse())
		})

		It("skips + stamps target-container-not-found for an unknown container name", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Annotations[LMCacheAnnotationContainer] = testUnknownContainer
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			Expect(out.Annotations[LMCacheAnnotationSkipReason]).To(Equal(SkipReasonTargetContainerNotFound))
			Expect(out.Annotations).NotTo(HaveKey(LMCacheAnnotationInjected))
			Expect(out.Spec.HostIPC).To(BeFalse())
		})
	})

	Describe("user-supplied --kv-transfer-config", func() {
		It("skips + stamps but still applies hostIPC + PYTHONHASHSEED", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Spec.Containers[0].Args = []string{"--kv-transfer-config", `{"kv_connector":"Other"}`}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)
			c := findContainer(out, testContainerVLLM)

			By("the user's kv-transfer-config JSON is untouched")
			Expect(argsFlagValue(c.Args, lmcFlagKVTransferConfig)).To(Equal(`{"kv_connector":"Other"}`))
			Expect(argsFlagValue(c.Args, lmcFlagKVTransferConfig)).NotTo(ContainSubstring("LMCacheMPConnector"))

			By("the skip reason is stamped but the rest of the injection still applies")
			Expect(out.Annotations[LMCacheAnnotationSkipReason]).To(Equal(SkipReasonKVTransferConfigPresent))
			Expect(out.Annotations[LMCacheAnnotationInjected]).To(Equal(valueTrue))
			Expect(out.Spec.HostIPC).To(BeTrue())
			Expect(envValue(c, pythonHashSeedEnvName)).To(Equal(pythonHashSeedValue))
		})
	})

	Describe("target container resolution", func() {
		It("injects into the annotation-named non-first container", func() {
			injector := newLMCacheInjector(true)
			pod := lmcachePod(func(p *corev1.Pod) {
				p.Annotations[LMCacheAnnotationContainer] = testContainerVLLM
				p.Spec.Containers = []corev1.Container{
					{Name: "sidecar", Image: "busybox", Args: []string{"sleep"}},
					{Name: testContainerVLLM, Image: "vllm/vllm-openai:latest", Args: []string{"--model", "m"}},
				}
			})

			resp := injector.Handle(ctx, makeRequest(pod))
			out := applyResponse(pod, resp)

			By("the vLLM container is mutated")
			vllm := findContainer(out, testContainerVLLM)
			Expect(argsFlagValue(vllm.Args, lmcFlagKVTransferConfig)).To(ContainSubstring("LMCacheMPConnector"))
			Expect(envValue(vllm, pythonHashSeedEnvName)).To(Equal(pythonHashSeedValue))

			By("the sidecar container is untouched")
			sidecar := findContainer(out, "sidecar")
			Expect(sidecar.Args).To(Equal([]string{"sleep"}))
			Expect(envValue(sidecar, pythonHashSeedEnvName)).To(BeEmpty())
		})
	})
})
