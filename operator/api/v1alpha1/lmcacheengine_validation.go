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

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateSpec validates the LMCacheEngineSpec and returns any validation errors.
func (e *LMCacheEngine) ValidateSpec() field.ErrorList {
	var errs field.ErrorList
	spec := &e.Spec
	l1Path := field.NewPath("spec", "l1")

	// l1.sizeGB must be > 0
	if spec.L1.SizeGB <= 0 {
		errs = append(errs, field.Invalid(l1Path.Child("sizeGB"), spec.L1.SizeGB, "must be greater than 0"))
	}

	// Eviction validation
	if spec.Eviction != nil {
		evPath := field.NewPath("spec", "eviction")

		if spec.Eviction.Policy != nil && *spec.Eviction.Policy != "LRU" {
			errs = append(errs, field.NotSupported(evPath.Child("policy"), *spec.Eviction.Policy, []string{"LRU"}))
		}

		if spec.Eviction.TriggerWatermark != nil {
			tw := *spec.Eviction.TriggerWatermark
			if tw <= 0.0 || tw > 1.0 {
				errs = append(errs, field.Invalid(evPath.Child("triggerWatermark"), tw, "must be in (0.0, 1.0]"))
			}
		}

		if spec.Eviction.EvictionRatio != nil {
			er := *spec.Eviction.EvictionRatio
			if er <= 0.0 || er > 1.0 {
				errs = append(errs, field.Invalid(evPath.Child("evictionRatio"), er, "must be in (0.0, 1.0]"))
			}
		}
	}

	// Server port validation
	if spec.Server != nil && spec.Server.Port != nil {
		port := *spec.Server.Port
		if port < 1024 || port > 65535 {
			errs = append(errs, field.Invalid(field.NewPath("spec", "server", "port"), port, "must be in [1024, 65535]"))
		}
	}

	// L2 backend validation
	if spec.L2Backend != nil {
		l2Path := field.NewPath("spec", "l2Backend")
		b := spec.L2Backend

		setCount := 0
		if b.RESP != nil {
			setCount++
		}
		if b.Raw != nil {
			setCount++
		}
		// For now we only support one kind at each LMCache server. LMCache
		// MP mode is designed to support multiple ones at the same time
		// but tests and performance validation is needed before we ship
		// it into operator.
		if setCount == 0 {
			errs = append(errs, field.Required(l2Path, "exactly one of resp or raw must be set"))
		} else if setCount > 1 {
			errs = append(errs, field.Invalid(l2Path, "", "exactly one of resp or raw must be set, got multiple"))
		}

		if b.RESP != nil {
			respPath := l2Path.Child("resp")
			if b.RESP.Host == "" {
				errs = append(errs, field.Required(respPath.Child("host"), "must be a non-empty string"))
			}
			if b.RESP.Port < 1 || b.RESP.Port > 65535 {
				errs = append(errs, field.Invalid(respPath.Child("port"), b.RESP.Port, "must be in [1, 65535]"))
			}
			if b.RESP.AuthSecretRef != nil && b.RESP.AuthSecretRef.Name == "" {
				errs = append(errs, field.Required(respPath.Child("authSecretRef", "name"), "must be non-empty"))
			}
		}

		if b.Raw != nil {
			rawPath := l2Path.Child("raw")
			if b.Raw.Type == "" {
				errs = append(errs, field.Required(rawPath.Child("type"), "must be a non-empty string"))
			}
		}
	}

	return errs
}
