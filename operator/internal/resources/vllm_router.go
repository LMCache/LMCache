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
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

const (
	// routerDefaultImageRepo is the default image for the vllm-router container.
	// vllm/vllm-router is the dedicated router image (~85 MB); it ships the
	// vllm-router binary without the full vLLM/lmcache stack.
	routerDefaultImageRepo = "vllm/vllm-router"
	routerDefaultImageTag  = "nightly"
)

// RouterSelectorLabels returns the immutable label subset used for the router
// pod selector. It uses a distinct component value from cache-engine labels so
// a router and an engine with the same name can coexist without selector collision.
func RouterSelectorLabels(name string) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":       "lmcache-router",
		"app.kubernetes.io/instance":   name,
		"app.kubernetes.io/managed-by": "lmcache-operator",
	}
}

// RouterStandardLabels returns the full label set for router-owned resources.
func RouterStandardLabels(name string) map[string]string {
	labels := RouterSelectorLabels(name)
	labels["app.kubernetes.io/component"] = "router"
	return labels
}

// RouterServiceName returns the name of the frontend Service for the given router.
func RouterServiceName(routerName string) string {
	return routerName
}

// resolveRouterImageRef builds the "<repository>:<tag>" string from the
// router's optional image override. Falls back to routerDefaultImageRepo:latest.
func resolveRouterImageRef(img *lmcachev1alpha1.ImageSpec) (string, corev1.PullPolicy) {
	repo := routerDefaultImageRepo
	tag := routerDefaultImageTag
	policy := corev1.PullIfNotPresent

	if img != nil {
		if img.Repository != nil && *img.Repository != "" {
			repo = *img.Repository
		}
		if img.Tag != nil && *img.Tag != "" {
			tag = *img.Tag
		}
		if img.PullPolicy != nil && *img.PullPolicy != "" {
			policy = corev1.PullPolicy(*img.PullPolicy)
		}
	}
	return repo + ":" + tag, policy
}

// endpointURL builds the in-cluster HTTP URL for a RouterEndpointSpec.
func endpointURL(spec lmcachev1alpha1.RouterEndpointSpec, namespace string) string {
	port := derefInt32(spec.Port, 8000)
	return fmt.Sprintf("http://%s.%s.svc.cluster.local:%d", spec.ServiceName, namespace, port)
}

// BuildRouterDeployment creates the Deployment that runs the vllm-router binary
// for the given VllmRouter CR.
func BuildRouterDeployment(router *lmcachev1alpha1.VllmRouter) *appsv1.Deployment {
	spec := &router.Spec

	imageRef, pullPolicy := resolveRouterImageRef(spec.Image)
	replicas := derefInt32(spec.Replicas, 1)
	routerPort := derefInt32(spec.Port, 30000)
	policy := derefString(spec.Policy, "round_robin")
	parallelSize := derefInt32(spec.IntraNodeDataParallelSize, 1)

	selectorLabels := RouterSelectorLabels(router.Name)
	podLabels := RouterStandardLabels(router.Name)

	// Build the vllm-router invocation as a shell command so that the binary
	// is found via PATH even when it lives in /opt/venv/bin (not in the
	// default $PATH in some base images). Mirrors the pattern used by vllm.yaml
	// in the tensormesh-operator examples (exec python3 -m ...).
	flags := []string{
		"--policy", policy,
		"--vllm-pd-disaggregation",
		"--prefill", endpointURL(spec.Prefill, router.Namespace),
		"--decode", endpointURL(spec.Decode, router.Namespace),
		"--host", "0.0.0.0",
		"--port", fmt.Sprintf("%d", routerPort),
		"--intra-node-data-parallel-size", fmt.Sprintf("%d", parallelSize),
	}
	shellCmd := "exec vllm-router " + strings.Join(flags, " ")

	container := corev1.Container{
		Name:            "router",
		Image:           imageRef,
		ImagePullPolicy: pullPolicy,
		Command:         []string{"/bin/sh", "-c"},
		Args:            []string{shellCmd},
		Ports: []corev1.ContainerPort{
			{Name: "http", ContainerPort: routerPort, Protocol: corev1.ProtocolTCP},
		},
		Env: spec.Env,
		ReadinessProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: "/health",
					Port: intstr.FromInt32(routerPort),
				},
			},
			InitialDelaySeconds: 5,
			PeriodSeconds:       10,
		},
	}

	podSpec := corev1.PodSpec{
		Containers:        []corev1.Container{container},
		NodeSelector:      spec.NodeSelector,
		Tolerations:       spec.Tolerations,
		PriorityClassName: spec.PriorityClassName,
	}

	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      router.Name,
			Namespace: router.Namespace,
			Labels:    RouterStandardLabels(router.Name),
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: selectorLabels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabels,
				},
				Spec: podSpec,
			},
		},
	}
}

// BuildRouterService creates the frontend ClusterIP (or user-chosen type) Service
// for the VllmRouter Deployment.
func BuildRouterService(router *lmcachev1alpha1.VllmRouter) *corev1.Service {
	routerPort := derefInt32(router.Spec.Port, 30000)

	svcType := corev1.ServiceTypeClusterIP
	if router.Spec.ServiceType != nil && *router.Spec.ServiceType != "" {
		svcType = *router.Spec.ServiceType
	}

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      RouterServiceName(router.Name),
			Namespace: router.Namespace,
			Labels:    RouterStandardLabels(router.Name),
		},
		Spec: corev1.ServiceSpec{
			Type:     svcType,
			Selector: RouterSelectorLabels(router.Name),
			Ports: []corev1.ServicePort{
				{
					Name:       "http",
					Port:       routerPort,
					TargetPort: intstr.FromInt32(routerPort),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}
}
