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
	"encoding/json"
	"net/http"

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

// buildPDEngine returns a minimal LMCacheEngine with spec.pd set.
func buildPDEngine() *lmcachev1alpha1.LMCacheEngine {
	return &lmcachev1alpha1.LMCacheEngine{
		ObjectMeta: metav1.ObjectMeta{Name: testEngineName, Namespace: testNamespace},
		Spec: lmcachev1alpha1.LMCacheEngineSpec{
			L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10},
			PD: &lmcachev1alpha1.PDSpec{},
		},
	}
}

// minimalPod returns a pod with one args-only vLLM container.
func minimalPod(namespace, name string, annotations map[string]string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: annotations,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "vllm", Image: "vllm/vllm-openai:latest", Args: []string{"--model", "test"}},
			},
		},
	}
}

// clientWithCM returns a fake client pre-seeded with the given ConfigMap.
func clientWithCM(cm *corev1.ConfigMap) *fake.ClientBuilder {
	return fake.NewClientBuilder().WithScheme(newTestScheme()).WithRuntimeObjects(cm)
}

var _ = Describe("prepareInjection", func() {
	var (
		ctx    context.Context
		keys   injectionKeys
		pod    *corev1.Pod
		engine *lmcachev1alpha1.LMCacheEngine
	)

	BeforeEach(func() {
		ctx = context.Background()
		keys = lmCacheKeys
		engine = &lmcachev1alpha1.LMCacheEngine{
			ObjectMeta: metav1.ObjectMeta{Name: testEngineName, Namespace: testNamespace},
			Spec:       lmcachev1alpha1.LMCacheEngineSpec{L1: lmcachev1alpha1.L1BackendSpec{SizeGB: 10}},
		}
		pod = minimalPod(testNamespace, testPodName, map[string]string{
			LMCacheAnnotationEngine: testEngineName,
		})
	})

	It("returns EngineNotFound skip when ConfigMap is absent", func() {
		c := fake.NewClientBuilder().WithScheme(newTestScheme()).Build()
		req := buildAdmissionRequest(pod)

		kvJSON, _, _, ok := prepareInjection(ctx, c, req, pod, keys, testEngineName, testNamespace, nil, "")

		Expect(ok).To(BeFalse())
		Expect(kvJSON).To(BeEmpty())
		Expect(pod.Annotations[keys.skipReason]).To(Equal(SkipReasonEngineNotFound))
	})

	It("selects kv-transfer-config.json for empty pdRole (non-PD)", func() {
		cm := resources.BuildConnectionConfigMap(engine)
		c := clientWithCM(cm).Build()
		req := buildAdmissionRequest(pod)

		kvJSON, idx, _, ok := prepareInjection(ctx, c, req, pod, keys, testEngineName, testNamespace, nil, "")

		Expect(ok).To(BeTrue())
		Expect(idx).To(Equal(0))
		Expect(kvJSON).To(ContainSubstring("LMCacheMPConnector"))
	})

	It("selects kv-transfer-config-prefiller.json for pdRole=prefiller", func() {
		pdEngine := buildPDEngine()
		cm := resources.BuildConnectionConfigMap(pdEngine)
		c := clientWithCM(cm).Build()
		req := buildAdmissionRequest(pod)

		kvJSON, _, _, ok := prepareInjection(ctx, c, req, pod, keys, testEngineName, testNamespace, nil, lmcachev1alpha1.PDRolePrefiller)

		Expect(ok).To(BeTrue())
		Expect(kvJSON).To(ContainSubstring("kv_producer"))
	})

	It("selects kv-transfer-config-decoder.json for pdRole=decoder", func() {
		pdEngine := buildPDEngine()
		cm := resources.BuildConnectionConfigMap(pdEngine)
		c := clientWithCM(cm).Build()
		req := buildAdmissionRequest(pod)

		kvJSON, _, _, ok := prepareInjection(ctx, c, req, pod, keys, testEngineName, testNamespace, nil, lmcachev1alpha1.PDRoleDecoder)

		Expect(ok).To(BeTrue())
		Expect(kvJSON).To(ContainSubstring("kv_consumer"))
	})

	It("returns TargetContainerNotFound skip when annotation names a missing container", func() {
		pod.Annotations[keys.container] = testUnknownContainer
		cm := resources.BuildConnectionConfigMap(engine)
		c := clientWithCM(cm).Build()
		req := buildAdmissionRequest(pod)

		_, _, _, ok := prepareInjection(ctx, c, req, pod, keys, testEngineName, testNamespace, nil, "")

		Expect(ok).To(BeFalse())
		Expect(pod.Annotations[keys.skipReason]).To(Equal(SkipReasonTargetContainerNotFound))
	})

	It("returns CommandOverride skip when target container sets Command", func() {
		pod.Spec.Containers[0].Command = []string{"/bin/sh", "-c"}
		cm := resources.BuildConnectionConfigMap(engine)
		c := clientWithCM(cm).Build()
		req := buildAdmissionRequest(pod)

		_, _, _, ok := prepareInjection(ctx, c, req, pod, keys, testEngineName, testNamespace, nil, "")

		Expect(ok).To(BeFalse())
		Expect(pod.Annotations[keys.skipReason]).To(Equal(SkipReasonCommandOverride))
	})

	It("returns error response when ConfigMap Get fails with non-404", func() {
		// Use a client whose scheme does not know ConfigMaps → triggers a scheme
		// error which the fake client surfaces as an internal error.
		emptyScheme := runtime.NewScheme()
		c := fake.NewClientBuilder().WithScheme(emptyScheme).Build()
		req := buildAdmissionRequest(pod)

		_, _, resp, ok := prepareInjection(ctx, c, req, pod, keys, testEngineName, testNamespace, nil, "")

		Expect(ok).To(BeFalse())
		Expect(resp.Result).NotTo(BeNil())
		Expect(resp.Result.Code).To(Equal(int32(http.StatusInternalServerError)))
	})
})

// buildAdmissionRequest encodes pod into a minimal admission.Request.
func buildAdmissionRequest(pod *corev1.Pod) admission.Request {
	raw, _ := json.Marshal(pod)
	req := admission.Request{}
	req.Object.Raw = raw
	req.Namespace = pod.Namespace
	return req
}
