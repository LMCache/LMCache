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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// RouterEndpointSpec identifies a vLLM Service (not the LMCache engine Service)
// that the router should forward requests to.
type RouterEndpointSpec struct {
	// serviceName is the name of the vLLM Kubernetes Service in the same namespace.
	ServiceName string `json:"serviceName"`

	// port is the HTTP port the vLLM Service exposes.
	// +optional
	// +kubebuilder:default=8000
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	Port *int32 `json:"port,omitempty"`
}

// VllmRouterSpec defines the desired state of VllmRouter.
type VllmRouterSpec struct {
	// prefill identifies the vLLM Service that runs the prefiller (kv_producer) role.
	Prefill RouterEndpointSpec `json:"prefill"`

	// decode identifies the vLLM Service that runs the decoder (kv_consumer) role.
	Decode RouterEndpointSpec `json:"decode"`

	// port is the port the router listens on.
	// +optional
	// +kubebuilder:default=30000
	// +kubebuilder:validation:Minimum=1024
	// +kubebuilder:validation:Maximum=65535
	Port *int32 `json:"port,omitempty"`

	// policy is the vllm-router routing policy.
	// +optional
	// +kubebuilder:default="round_robin"
	// +kubebuilder:validation:Enum=round_robin;session_based
	Policy *string `json:"policy,omitempty"`

	// replicas is the number of router replicas.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	Replicas *int32 `json:"replicas,omitempty"`

	// image configures the router container image.
	// Defaults to the same image as LMCacheEngine (lmcache/vllm-openai:latest).
	// +optional
	Image *ImageSpec `json:"image,omitempty"`

	// intraNodeDataParallelSize must match the tensor-parallel-size of the
	// vLLM instances the router talks to.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	IntraNodeDataParallelSize *int32 `json:"intraNodeDataParallelSize,omitempty"`

	// serviceType is the Kubernetes Service type for the router's frontend Service.
	// +optional
	// +kubebuilder:default="ClusterIP"
	ServiceType *corev1.ServiceType `json:"serviceType,omitempty"`

	// nodeSelector constrains which nodes the router pod may land on.
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// tolerations for the router pod.
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// env defines additional environment variables for the router container.
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// priorityClassName is the priority class for the router pod.
	// +optional
	PriorityClassName string `json:"priorityClassName,omitempty"`
}

// VllmRouterStatus defines the observed state of VllmRouter.
type VllmRouterStatus struct {
	// phase is the overall phase of the router.
	// +optional
	Phase string `json:"phase,omitempty"`

	// observedGeneration is the most recent generation observed.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// readyReplicas is the number of ready router replicas.
	// +optional
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// conditions represent the current state of the VllmRouter resource.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=vmr
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyReplicas`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// VllmRouter manages a vllm-router Deployment that routes requests between
// a prefiller vLLM pool and a decoder vLLM pool for PD disaggregation.
type VllmRouter struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is standard object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec defines the desired state of VllmRouter.
	// +required
	Spec VllmRouterSpec `json:"spec"`

	// status defines the observed state of VllmRouter.
	// +optional
	Status VllmRouterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// VllmRouterList contains a list of VllmRouter.
type VllmRouterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []VllmRouter `json:"items"`
}

func init() {
	SchemeBuilder.Register(&VllmRouter{}, &VllmRouterList{})
}
