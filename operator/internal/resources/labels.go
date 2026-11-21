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

package resources

import (
	"fmt"
	"hash/fnv"
	"maps"
	"strconv"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// SelectorLabels returns the immutable label subset used for pod selectors.
func SelectorLabels(name string) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":       "lmcache",
		"app.kubernetes.io/instance":   name,
		"app.kubernetes.io/managed-by": "lmcache-operator",
	}
}

// StandardLabels returns the full label set for all owned resources.
func StandardLabels(name string) map[string]string {
	labels := SelectorLabels(name)
	labels["app.kubernetes.io/component"] = "cache-engine"
	return labels
}

// MergeLabels merges multiple label maps. Later maps take precedence.
func MergeLabels(labelMaps ...map[string]string) map[string]string {
	result := make(map[string]string)
	for _, m := range labelMaps {
		maps.Copy(result, m)
	}
	return result
}

// WorkloadLabelPrefix is the node-label prefix the operator manages for
// workload-aware scheduling. The suffix is the hexadecimal encoding of an
// fnv64a hash of the engine's "namespace/name": keys stay inside the
// 63-character label budget regardless of CR name length, and the mapping
// is stable for the lifetime of the CR. Exported for the controller's node
// watch predicate, which acts on any label carrying this prefix.
const WorkloadLabelPrefix = "lmcache.ai/wlas-"

// WorkloadLabelValueBase is the label value stamped on every node when
// dynamic resource allocation is off: the tier multiplier "1". Keeping it
// numeric (rather than "true") means enabling DRA later only changes the
// value, never the semantics, of the existing placement label.
const WorkloadLabelValueBase = "1"

// TierLabelValue returns the managed node-label value for a tier
// multiplier: its decimal string ("1", "2", ...).
func TierLabelValue(multiplier int32) string {
	return strconv.Itoa(int(multiplier))
}

// EffectiveNodeSelector returns the DaemonSet nodeSelector for an engine:
// spec.nodeSelector alone, or — when workload-aware scheduling is enabled —
// merged with the managed placement label the operator maintains on nodes
// hosting matching workload pods. The extra key is what stops the DS
// controller from creating engine pods on nodes without the workload.
// With dynamic resource allocation the label value on a node is that
// node's tier multiplier; the base (1x) DaemonSet selects the base value.
func EffectiveNodeSelector(namespace, name string, spec *lmcachev1alpha1.LMCacheEngineSpec) map[string]string {
	if w := spec.WorkloadAwareScheduling; w != nil && w.Enabled {
		key := WorkloadPlacementLabel(namespace, name)
		// Tier 1 is the base label value whether or not DRA is on; a
		// node labeled with a higher tier simply does not match the
		// base DS.
		return MergeLabels(spec.NodeSelector, map[string]string{key: WorkloadLabelValueBase})
	}
	return spec.NodeSelector
}

// TierNodeSelector returns the nodeSelector for a tier DaemonSet of the
// given multiplier (N >= 2, DRA enabled): spec.nodeSelector plus the
// placement label pinned to the tier value.
func TierNodeSelector(namespace, name string, spec *lmcachev1alpha1.LMCacheEngineSpec, multiplier int32) map[string]string {
	key := WorkloadPlacementLabel(namespace, name)
	return MergeLabels(spec.NodeSelector, map[string]string{key: TierLabelValue(multiplier)})
}

// WorkloadPlacementLabel returns the node label key an LMCacheEngine
// manages on nodes hosting its matching workload pods. The key embeds an
// 16-hex fnv64a hash of "namespace/name" (the full uint64 as %x):
// deterministic, collision-safe for practical fleet sizes, and still
// well inside the 63-char key budget. The label's value is the node's
// tier multiplier — WorkloadLabelValueBase ("1") unless dynamic resource
// allocation is enabled.
func WorkloadPlacementLabel(namespace, name string) string {
	h := fnv.New64a()
	_, _ = h.Write([]byte(namespace + "/" + name))
	return fmt.Sprintf("%s%x", WorkloadLabelPrefix, h.Sum64())
}
