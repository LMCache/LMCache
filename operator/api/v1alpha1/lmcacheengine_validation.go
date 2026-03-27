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

	return errs
}
