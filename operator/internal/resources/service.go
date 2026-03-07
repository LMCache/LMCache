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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// BuildLookupService creates a ClusterIP Service with internalTrafficPolicy=Local
// for node-local service discovery. vLLM pods connect to this service and kube-proxy
// routes traffic only to the LMCache pod on the same node.
func BuildLookupService(engine *lmcachev1alpha1.LMCacheEngine) *corev1.Service {
	serverPort := derefInt32(getServerPort(&engine.Spec), 5555)
	localPolicy := corev1.ServiceInternalTrafficPolicyLocal

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      LookupServiceName(engine.Name),
			Namespace: engine.Namespace,
			Labels:    StandardLabels(engine.Name),
		},
		Spec: corev1.ServiceSpec{
			Selector:              SelectorLabels(engine.Name),
			InternalTrafficPolicy: &localPolicy,
			Ports: []corev1.ServicePort{
				{
					Name:     "server",
					Port:     serverPort,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
	}
}

// BuildMetricsService creates a headless Service for Prometheus scraping.
func BuildMetricsService(engine *lmcachev1alpha1.LMCacheEngine) *corev1.Service {
	promPort := int32(9090)
	if engine.Spec.Prometheus != nil {
		promPort = derefInt32(engine.Spec.Prometheus.Port, 9090)
	}

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-metrics", engine.Name),
			Namespace: engine.Namespace,
			Labels:    StandardLabels(engine.Name),
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: corev1.ClusterIPNone,
			Selector:  SelectorLabels(engine.Name),
			Ports: []corev1.ServicePort{
				{
					Name:     "metrics",
					Port:     promPort,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
	}
}
