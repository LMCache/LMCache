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
	corev1 "k8s.io/api/core/v1"
)

// LMCache vLLM flag / env names injected into an opted-in pod. Unlike the
// CacheBlend injector, the standard LMCacheEngine path needs no plugin staging,
// init container, or fixed flag battery: a vLLM only needs the connector JSON
// (--kv-transfer-config) and deterministic prefix hashing (PYTHONHASHSEED).
const (
	// lmcFlagKVTransferConfig is the vLLM flag carrying the connector JSON. The
	// value is the LMCacheMPConnector config read verbatim from the engine's
	// <engine>-connection ConfigMap.
	lmcFlagKVTransferConfig = "--kv-transfer-config"

	// pythonHashSeedEnvName pins Python's hash seed so prefix-chunk hashes are
	// identical across the vLLM process and the LMCache server; without it a
	// repeated prompt would hash differently and miss the cache.
	pythonHashSeedEnvName = "PYTHONHASHSEED"

	// pythonHashSeedValue is the deterministic seed value injected when the user
	// has not already set PYTHONHASHSEED.
	pythonHashSeedValue = "0"
)

// BuildLMCacheArgs returns the target vLLM container's args with the LMCache
// --kv-transfer-config flag applied (append-or-replace via applyArg, shared with
// the CacheBlend builders). The flag is injected only when kvTransferConfigJSON
// is non-empty; the handler passes "" when the user already supplied their own
// --kv-transfer-config so this builder leaves the user's value untouched.
//
// Parameters:
//   - existingArgs: the target container's current args (may be nil).
//   - kvTransferConfigJSON: the LMCacheMPConnector JSON from the engine's
//     connection ConfigMap, or "" to skip injecting/replacing the flag.
//
// Returns a new args slice; the input is not mutated.
func BuildLMCacheArgs(existingArgs []string, kvTransferConfigJSON string) []string {
	args := make([]string, len(existingArgs))
	copy(args, existingArgs)

	if kvTransferConfigJSON != "" {
		args = applyArg(args, lmcFlagKVTransferConfig, kvTransferConfigJSON)
	}
	return args
}

// BuildLMCacheEnv returns the target vLLM container's env with PYTHONHASHSEED set
// to "0" when it is absent. It is set on the container, never the pod, so every
// spawned worker inherits the deterministic seed. A user-supplied PYTHONHASHSEED
// (any value, including a valueFrom) is respected and left untouched — the user
// has opted into their own hashing scheme.
//
// Parameters:
//   - existing: the target container's current env list (may be nil).
//
// Returns the env list to assign; the input is not mutated.
func BuildLMCacheEnv(existing []corev1.EnvVar) []corev1.EnvVar {
	for i := range existing {
		if existing[i].Name == pythonHashSeedEnvName {
			return existing
		}
	}
	out := make([]corev1.EnvVar, 0, len(existing)+1)
	out = append(out, existing...)
	out = append(out, corev1.EnvVar{Name: pythonHashSeedEnvName, Value: pythonHashSeedValue})
	return out
}
