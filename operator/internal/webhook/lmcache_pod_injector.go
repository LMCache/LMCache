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

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// LMCache annotation keys the webhook reads and stamps. They mirror the
// CacheBlend keys (cacheblend_pod_injector.go) with an lmcache- discriminator so
// the two injectors never cross-fire on the same pod.
const (
	// LMCacheAnnotationEngine binds a pod to an LMCacheEngine in the same
	// namespace. Its presence is the opt-in signal; its value is the engine name.
	LMCacheAnnotationEngine = "lmcache.ai/lmcache-engine"

	// LMCacheAnnotationContainer optionally names the target vLLM container;
	// empty or absent selects the first container.
	LMCacheAnnotationContainer = "lmcache.ai/lmcache-container"

	// LMCacheAnnotationInjected is the idempotency guard stamped after a
	// successful injection; a re-admitted pod carrying it is allowed unchanged.
	LMCacheAnnotationInjected = "lmcache.ai/lmcache-injected"

	// LMCacheAnnotationSkipReason records why injection was skipped (fail-open).
	LMCacheAnnotationSkipReason = "lmcache.ai/lmcache-skip-reason"
)

// LMCacheLabelInject is the opt-in label the MutatingWebhookConfiguration's
// objectSelector matches (mutating_webhook_selectors_patch.yaml). It gates which
// pods reach the webhook; the handler itself gates on LMCacheAnnotationEngine.
const LMCacheLabelInject = "lmcache.ai/lmcache-inject"

// lmCacheKeys is the LMCache injector's annotation key set, consumed by the
// shared gate / skip / stamp helpers.
var lmCacheKeys = injectionKeys{
	engine:     LMCacheAnnotationEngine,
	container:  LMCacheAnnotationContainer,
	injected:   LMCacheAnnotationInjected,
	skipReason: LMCacheAnnotationSkipReason,
}

// +kubebuilder:webhook:path=/mutate-lmcache--v1-pod,mutating=true,failurePolicy=ignore,sideEffects=None,groups="",resources=pods,verbs=create,versions=v1,name=mlmcachepod.lmcache.ai,admissionReviewVersions=v1,reinvocationPolicy=Never

// LMCachePodInjector is the mutating admission handler that wires an opted-in
// vLLM pod to an LMCacheEngine so the user no longer has to hand-write the
// --kv-transfer-config flag, hostIPC, and PYTHONHASHSEED. It is gated by the
// engine's connection ConfigMap: it mutates a pod only when the pod's
// lmcache.ai/lmcache-engine annotation names an engine whose <engine>-connection
// ConfigMap exists. It fails open (failurePolicy: Ignore) and is idempotent.
//
// Unlike the CacheBlend injector it does not read the engine CR: the entire
// connector config lives in the connection ConfigMap, and LMCacheEngine has no
// injection sub-spec, so the ConfigMap is the only thing it needs.
type LMCachePodInjector struct {
	// Client reads the engine's connection ConfigMap. It uses the shared manager
	// ServiceAccount, whose RBAC already grants configmaps get.
	Client client.Client

	// Decoder decodes the admission request's raw pod object.
	Decoder admission.Decoder
}

// Handle implements admission.Handler. It applies the LMCache mutations (hostIPC,
// --kv-transfer-config, PYTHONHASHSEED) to an opted-in pod whose named
// LMCacheEngine connection ConfigMap exists, then returns a JSON patch. It
// short-circuits to an unchanged Allowed response for non-opted-in or
// already-injected pods, and stamps a skip-reason annotation (still Allowed,
// fail-open) when it declines to mutate (engine missing, command override,
// target container missing, or user-supplied --kv-transfer-config).
func (p *LMCachePodInjector) Handle(ctx context.Context, req admission.Request) admission.Response {
	log := ctrl.LoggerFrom(ctx)

	pod, engineName, namespace, resp, handled := lmCacheKeys.gate(p.Decoder, req)
	if handled {
		return resp
	}

	// Read the connection ConfigMap, resolve the target container, and apply the
	// command-override gate (shared with the CacheBlend injector). No engine CR
	// lookup: the connector config lives entirely in the ConfigMap.
	kvTransferConfigJSON, containerIdx, resp, ok := prepareInjection(
		ctx, p.Client, req, pod, lmCacheKeys, engineName, namespace, nil)
	if !ok {
		return resp
	}
	target := &pod.Spec.Containers[containerIdx]

	// user --kv-transfer-config gate: do not clobber the user's structured JSON.
	userHasKVTransferConfig := argsHasFlag(target.Args, lmcFlagKVTransferConfig)

	// M0: pod hostIPC for CUDA IPC with the node-local engine.
	pod.Spec.HostIPC = true

	// Args: inject --kv-transfer-config unless the user already supplied one.
	kvForArgs := kvTransferConfigJSON
	if userHasKVTransferConfig {
		kvForArgs = ""
	}
	target.Args = BuildLMCacheArgs(target.Args, kvForArgs)

	// Env: deterministic prefix hashing (set-if-absent).
	target.Env = BuildLMCacheEnv(target.Env)

	log.Info("Injected LMCache connection", "engine", engineName, "container", target.Name)
	return lmCacheKeys.stampInjected(req, pod, userHasKVTransferConfig)
}
