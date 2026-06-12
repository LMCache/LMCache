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
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

// Annotation keys the webhook reads and stamps (design §8).
const (
	// AnnotationEngine binds a pod to a CacheBlendEngine in the same namespace.
	// Its presence is the opt-in signal; its value is the engine name.
	AnnotationEngine = "lmcache.ai/cacheblend-engine"

	// AnnotationContainer optionally names the target vLLM container; empty or
	// absent selects the first container.
	AnnotationContainer = "lmcache.ai/cacheblend-container"

	// AnnotationImagePullSecrets optionally overrides the engine's
	// injection.imagePullSecrets with a comma-separated list of Secret names
	// appended to the pod's spec.imagePullSecrets for the private payload image.
	AnnotationImagePullSecrets = "lmcache.ai/cacheblend-image-pull-secrets"

	// AnnotationInjected is the idempotency guard stamped after a successful
	// injection; a re-admitted pod carrying it is allowed unchanged.
	AnnotationInjected = "lmcache.ai/cacheblend-injected"

	// AnnotationSkipReason records why injection was skipped (fail-open).
	AnnotationSkipReason = "lmcache.ai/cacheblend-skip-reason"
)

// valueTrue is the boolean-true string stamped on AnnotationInjected and used as
// the opt-in value of the lmcache.ai/cacheblend-inject label.
const valueTrue = "true"

// Skip-reason values stamped on AnnotationSkipReason (design §8).
const (
	// SkipReasonEngineNotFound is stamped when the named engine's connection
	// ConfigMap (or the engine CR) does not exist (fail-open).
	SkipReasonEngineNotFound = "engine-not-found"

	// SkipReasonCommandOverride is stamped when the target container overrides
	// command, so appended args may never reach `vllm serve`.
	SkipReasonCommandOverride = "command-override"

	// SkipReasonKVTransferConfigPresent is stamped when the user already supplies
	// --kv-transfer-config; the webhook does not clobber their structured JSON.
	SkipReasonKVTransferConfigPresent = "kv-transfer-config-present"

	// SkipReasonPayloadImageUnset is stamped when the engine's
	// injection.payloadImage resolves to an empty reference (no repository). The
	// webhook skips rather than inject an init container with an empty image,
	// which the API server would reject. CRD validation normally prevents this.
	SkipReasonPayloadImageUnset = "payload-image-unset"

	// SkipReasonTargetContainerNotFound is stamped when the requested target
	// container (injection.targetContainer or the cacheblend-container
	// annotation) names a container that does not exist on the pod, so there is
	// nothing to inject into.
	SkipReasonTargetContainerNotFound = "target-container-not-found"
)

// kvTransferConfigDataKey is the key within the <engine>-connection ConfigMap's
// Data map that holds the CBKVConnector kv-transfer-config JSON. It must match
// the key written by resources.buildConnectionConfigMapCore.
const kvTransferConfigDataKey = "kv-transfer-config.json"

// +kubebuilder:webhook:path=/mutate--v1-pod,mutating=true,failurePolicy=ignore,sideEffects=None,groups="",resources=pods,verbs=create,versions=v1,name=mpod.lmcache.ai,admissionReviewVersions=v1,reinvocationPolicy=Never

// PodInjector is the mutating admission handler that injects the
// lmcache-cacheblend vLLM plugin into opted-in pods (design §7). It is gated by
// the CacheBlendEngine CR: it mutates a pod only when the pod's
// lmcache.ai/cacheblend-engine annotation names an engine whose connection
// ConfigMap exists. It fails open (failurePolicy: Ignore) and is idempotent.
type PodInjector struct {
	// Client reads the named CacheBlendEngine and its connection ConfigMap. It
	// uses the shared manager ServiceAccount, whose RBAC already grants
	// cacheblendengines get and configmaps get (design §7 RBAC note).
	Client client.Client

	// Decoder decodes the admission request's raw pod object.
	Decoder admission.Decoder
}

// Handle implements admission.Handler. It applies mutations M0–M7 to an opted-in
// pod whose named CacheBlendEngine connection ConfigMap exists, then returns a
// JSON patch via admission.PatchResponseFromRaw. It short-circuits to an
// unchanged Allowed response for non-opted-in or already-injected pods, and
// stamps a skip-reason annotation (still Allowed, fail-open) when it declines to
// mutate (engine missing, command override, or user-supplied
// --kv-transfer-config).
func (p *PodInjector) Handle(ctx context.Context, req admission.Request) admission.Response {
	log := ctrl.LoggerFrom(ctx)

	pod := &corev1.Pod{}
	if err := p.Decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	// (1) Idempotency short-circuit: a pod already carrying the injected guard is
	// allowed unchanged on re-admission.
	if pod.Annotations[AnnotationInjected] == valueTrue {
		return admission.Allowed("already injected")
	}

	// (2) Opt-in gate: no engine annotation means this pod did not opt in.
	engineName := strings.TrimSpace(pod.Annotations[AnnotationEngine])
	if engineName == "" {
		return admission.Allowed("not opted in")
	}

	// The webhook config carries no defaulting for the pod object, so apply the
	// engine annotation value as the lookup namespace using the admission
	// request's namespace (the pod's namespace; pod.Namespace may be empty on
	// CREATE before the API server stamps it).
	namespace := req.Namespace
	if namespace == "" {
		namespace = pod.Namespace
	}

	// (3a) Resolve the engine CR for its injection defaults.
	engine := &lmcachev1alpha1.CacheBlendEngine{}
	if err := p.Client.Get(ctx, types.NamespacedName{Name: engineName, Namespace: namespace}, engine); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Skipped CacheBlend injection: engine not found",
				"engine", engineName, "namespace", namespace)
			return p.skip(req, pod, SkipReasonEngineNotFound)
		}
		return admission.Errored(http.StatusInternalServerError, err)
	}
	engine.SetDefaults()

	// (3b) Read the engine's connection ConfigMap (existence gate, no readiness
	// check — design §7/§9.9). Absent means the engine is not provisioned yet.
	connCM := &corev1.ConfigMap{}
	connName := resources.ConnectionConfigMapName(engineName)
	if err := p.Client.Get(ctx, types.NamespacedName{Name: connName, Namespace: namespace}, connCM); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Skipped CacheBlend injection: connection ConfigMap not found",
				"configMap", connName, "namespace", namespace)
			return p.skip(req, pod, SkipReasonEngineNotFound)
		}
		return admission.Errored(http.StatusInternalServerError, err)
	}
	kvTransferConfigJSON := connCM.Data[kvTransferConfigDataKey]

	// (6) Resolve the target container (annotation or first). A requested
	// container name that does not exist (or a pod with no containers) is a
	// misconfiguration: skip + stamp rather than silently allowing it through.
	containerIdx, ok := resolveTargetContainer(pod, engine.Spec.Injection.TargetContainer,
		pod.Annotations[AnnotationContainer])
	if !ok {
		log.Info("Skipped CacheBlend injection: target container not found",
			"engine", engineName,
			"annotationContainer", pod.Annotations[AnnotationContainer],
			"specTargetContainer", deref(engine.Spec.Injection.TargetContainer))
		return p.skip(req, pod, SkipReasonTargetContainerNotFound)
	}
	target := &pod.Spec.Containers[containerIdx]

	// (4) command-override gate: a wrapper command means appended args may never
	// reach `vllm serve`, so skip + stamp (design §8).
	if len(target.Command) > 0 {
		log.Info("Skipped CacheBlend injection: target container overrides command",
			"engine", engineName, "container", target.Name)
		return p.skip(req, pod, SkipReasonCommandOverride)
	}

	// (5) user --kv-transfer-config gate: skip that flag (do not clobber the
	// user's structured JSON) but still apply the rest of the mutation, stamping
	// the skip reason for diagnostics (design §9.2).
	userHasKVTransferConfig := argsHasFlag(target.Args, cbFlagKVTransferConfig)

	original := req.Object.Raw

	// --- Apply mutations M0–M7 ---

	// M0: pod hostIPC for CUDA IPC with the node-local engine.
	pod.Spec.HostIPC = true

	// M1: shared emptyDir volume.
	pod.Spec.Volumes = appendVolumeIfAbsent(pod.Spec.Volumes, BuildCBPluginVolume())

	// M2: payload init container (payloadImage is an ImageSpec: repo/tag/policy).
	// Fail open if the image resolves to empty (no repository) rather than inject
	// an init container with an empty image, which the API server would reject.
	payloadRef, payloadPullPolicy := resolvePayloadImage(engine.Spec.Injection.PayloadImage)
	if payloadRef == "" {
		log.Info("Skipped CacheBlend injection: payload image repository is unset",
			"engine", engineName)
		return p.skip(req, pod, SkipReasonPayloadImageUnset)
	}
	pod.Spec.InitContainers = appendInitContainerIfAbsent(pod.Spec.InitContainers,
		BuildCBInitContainer(payloadRef, payloadPullPolicy))

	// M3: read-only mount on the target container.
	target.VolumeMounts = appendVolumeMountIfAbsent(target.VolumeMounts, BuildCBVolumeMount())

	// M4: PYTHONPATH on the target container.
	target.Env = BuildCBPodEnv(target.Env)

	// M5: required vLLM args. Pass "" for the kv-transfer-config JSON when the
	// user already supplies one so BuildCBArgs leaves their value untouched.
	kvForArgs := kvTransferConfigJSON
	if userHasKVTransferConfig {
		kvForArgs = ""
	}
	cudagraph := deref(engine.Spec.Injection.Cudagraph)
	target.Args = BuildCBArgs(target.Args, kvForArgs, cudagraph)

	// M7: append injection pull secrets (annotation override wins) to the pod's
	// imagePullSecrets, deduped (private payload image).
	injectedSecrets := resolveInjectedPullSecrets(engine.Spec.Injection.ImagePullSecrets,
		pod.Annotations[AnnotationImagePullSecrets])
	pod.Spec.ImagePullSecrets = MergeImagePullSecrets(pod.Spec.ImagePullSecrets, injectedSecrets)

	// M6 + skip-reason stamping: idempotency guard, plus the diagnostic reason if
	// we skipped the --kv-transfer-config flag.
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[AnnotationInjected] = valueTrue
	if userHasKVTransferConfig {
		pod.Annotations[AnnotationSkipReason] = SkipReasonKVTransferConfigPresent
	}

	marshaled, err := json.Marshal(pod)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, err)
	}
	log.Info("Injected CacheBlend plugin", "engine", engineName, "container", target.Name)
	return admission.PatchResponseFromRaw(original, marshaled)
}

// skip stamps the given skip reason on the pod (without injecting), marshals it,
// and returns an Allowed patch response. The pod is still admitted (fail-open).
func (p *PodInjector) skip(req admission.Request, pod *corev1.Pod, reason string) admission.Response {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[AnnotationSkipReason] = reason
	marshaled, err := json.Marshal(pod)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, err)
	}
	return admission.PatchResponseFromRaw(req.Object.Raw, marshaled)
}

// resolveTargetContainer returns the index of the container to inject into and
// whether one was found. The per-pod annotation override (annotationName) takes
// precedence over the engine's injection.targetContainer default; an empty
// selection falls back to the first container. A non-empty name that matches no
// container yields ok=false.
//
// Parameters:
//   - pod: the decoded pod.
//   - specDefault: the engine's injection.targetContainer (nil/"" = first).
//   - annotationName: the per-pod cacheblend-container annotation value.
func resolveTargetContainer(
	pod *corev1.Pod,
	specDefault *string,
	annotationName string,
) (int, bool) {
	if len(pod.Spec.Containers) == 0 {
		return 0, false
	}

	name := strings.TrimSpace(annotationName)
	if name == "" && specDefault != nil {
		name = strings.TrimSpace(*specDefault)
	}
	if name == "" {
		return 0, true
	}
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == name {
			return i, true
		}
	}
	return 0, false
}

// resolveInjectedPullSecrets returns the pull-secret references to inject: the
// per-pod annotation override (a comma-separated list of Secret names) when
// present, otherwise the engine's injection.imagePullSecrets.
//
// Parameters:
//   - specSecrets: the engine's injection.imagePullSecrets.
//   - annotationCSV: the cacheblend-image-pull-secrets annotation value.
func resolveInjectedPullSecrets(
	specSecrets []corev1.LocalObjectReference,
	annotationCSV string,
) []corev1.LocalObjectReference {
	csv := strings.TrimSpace(annotationCSV)
	if csv == "" {
		return specSecrets
	}
	out := make([]corev1.LocalObjectReference, 0)
	for part := range strings.SplitSeq(csv, ",") {
		name := strings.TrimSpace(part)
		if name == "" {
			continue
		}
		out = append(out, corev1.LocalObjectReference{Name: name})
	}
	return out
}

// argsHasFlag reports whether args already carries the given flag in either the
// two-token "--flag value" form or the single-token "--flag=value" form.
func argsHasFlag(args []string, flag string) bool {
	eqPrefix := flag + "="
	for _, a := range args {
		if a == flag {
			return true
		}
		if strings.HasPrefix(a, eqPrefix) {
			return true
		}
	}
	return false
}

// appendVolumeIfAbsent appends v to volumes unless a volume of the same name is
// already present (idempotency within a single Handle call). Returns the slice.
func appendVolumeIfAbsent(volumes []corev1.Volume, v corev1.Volume) []corev1.Volume {
	for i := range volumes {
		if volumes[i].Name == v.Name {
			return volumes
		}
	}
	return append(volumes, v)
}

// appendInitContainerIfAbsent appends c to initContainers unless one of the same
// name is already present. Returns the slice.
func appendInitContainerIfAbsent(
	initContainers []corev1.Container,
	c corev1.Container,
) []corev1.Container {
	for i := range initContainers {
		if initContainers[i].Name == c.Name {
			return initContainers
		}
	}
	return append(initContainers, c)
}

// appendVolumeMountIfAbsent appends m to mounts unless one of the same name is
// already present. Returns the slice.
func appendVolumeMountIfAbsent(
	mounts []corev1.VolumeMount,
	m corev1.VolumeMount,
) []corev1.VolumeMount {
	for i := range mounts {
		if mounts[i].Name == m.Name {
			return mounts
		}
	}
	return append(mounts, m)
}

// deref returns the value pointed to by s, or "" if s is nil.
func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// resolvePayloadImage builds the "<repository>:<tag>" reference and pull policy
// for the payload init container from the engine's injection.payloadImage. Tag
// and pull policy fall back to "latest" / IfNotPresent when unset; repository is
// taken as-is (it has no sensible cluster-wide default — see InjectionSpec docs).
func resolvePayloadImage(img *lmcachev1alpha1.ImageSpec) (string, corev1.PullPolicy) {
	if img == nil || deref(img.Repository) == "" {
		return "", corev1.PullIfNotPresent
	}
	tag := deref(img.Tag)
	if tag == "" {
		tag = "latest"
	}
	policy := corev1.PullPolicy(deref(img.PullPolicy))
	if policy == "" {
		policy = corev1.PullIfNotPresent
	}
	return deref(img.Repository) + ":" + tag, policy
}
