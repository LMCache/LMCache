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
	"encoding/json"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// LookupServiceName returns the name of the node-local lookup Service for discovery.
func LookupServiceName(engineName string) string {
	return engineName
}

// BuildConnectionConfigMap creates the <name>-connection ConfigMap with kv-transfer-config JSON.
func BuildConnectionConfigMap(engine *lmcachev1alpha1.LMCacheEngine) *corev1.ConfigMap {
	port := derefInt32(getServerPort(&engine.Spec), 5555)
	svcHost := fmt.Sprintf("%s.%s.svc.cluster.local", LookupServiceName(engine.Name), engine.Namespace)

	config := map[string]any{
		"kv_connector": "LMCacheMPConnector",
		"kv_role":      "kv_both",
		"kv_connector_extra_config": map[string]any{
			"lmcache.mp.host": fmt.Sprintf("tcp://%s", svcHost),
			"lmcache.mp.port": fmt.Sprintf("%d", port),
		},
	}

	configJSON, _ := json.MarshalIndent(config, "", "  ")

	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-connection", engine.Name),
			Namespace: engine.Namespace,
			Labels:    StandardLabels(engine.Name),
		},
		Data: map[string]string{
			"kv-transfer-config.json": string(configJSON),
		},
	}
}
