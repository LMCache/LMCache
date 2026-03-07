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
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Phase constants for LMCacheEngine status.
const (
	PhasePending  = "Pending"
	PhaseRunning  = "Running"
	PhaseDegraded = "Degraded"
	PhaseFailed   = "Failed"
)

// Condition type constants.
const (
	ConditionAvailable         = "Available"
	ConditionAllInstancesReady = "AllInstancesReady"
	ConditionConfigValid       = "ConfigValid"
)

// ImageSpec defines the container image to use.
type ImageSpec struct {
	// repository is the container image repository.
	// +optional
	// +kubebuilder:default="lmcache/vllm-openai"
	Repository *string `json:"repository,omitempty"`

	// tag is the container image tag.
	// +optional
	// +kubebuilder:default="latest"
	Tag *string `json:"tag,omitempty"`

	// pullPolicy is the image pull policy.
	// +optional
	// +kubebuilder:default="IfNotPresent"
	// +kubebuilder:validation:Enum=Always;Never;IfNotPresent
	PullPolicy *string `json:"pullPolicy,omitempty"`
}

// ServerSpec defines server configuration mapping to server.py argparse.
type ServerSpec struct {
	// port is the server listening port.
	// +optional
	// +kubebuilder:default=5555
	// +kubebuilder:validation:Minimum=1024
	// +kubebuilder:validation:Maximum=65535
	Port *int32 `json:"port,omitempty"`

	// chunkSize is the token chunk size.
	// +optional
	// +kubebuilder:default=256
	ChunkSize *int32 `json:"chunkSize,omitempty"`

	// maxWorkers is the number of worker threads.
	// +optional
	// +kubebuilder:default=1
	MaxWorkers *int32 `json:"maxWorkers,omitempty"`

	// hashAlgorithm is the hash algorithm used for token hashing.
	// +optional
	// +kubebuilder:default="blake3"
	// +kubebuilder:validation:Enum=builtin;sha256_cbor;blake3
	HashAlgorithm *string `json:"hashAlgorithm,omitempty"`
}

// L1BackendSpec defines the L1 memory cache configuration.
type L1BackendSpec struct {
	// sizeGB is the L1 cache size in gigabytes. Required, must be > 0.
	SizeGB float64 `json:"sizeGB"`
}

// EvictionSpec defines the cache eviction configuration.
type EvictionSpec struct {
	// policy is the eviction policy. Currently only LRU is supported.
	// +optional
	// +kubebuilder:default="LRU"
	// +kubebuilder:validation:Enum=LRU
	Policy *string `json:"policy,omitempty"`

	// triggerWatermark is the cache usage ratio that triggers eviction.
	// +optional
	// +kubebuilder:default=0.8
	TriggerWatermark *float64 `json:"triggerWatermark,omitempty"`

	// evictionRatio is the fraction of cache to evict when triggered.
	// +optional
	// +kubebuilder:default=0.2
	EvictionRatio *float64 `json:"evictionRatio,omitempty"`
}

// ServiceMonitorSpec defines Prometheus ServiceMonitor configuration.
type ServiceMonitorSpec struct {
	// enabled controls whether a ServiceMonitor CR is created.
	// +optional
	// +kubebuilder:default=false
	Enabled *bool `json:"enabled,omitempty"`

	// interval is the Prometheus scrape interval.
	// +optional
	// +kubebuilder:default="30s"
	Interval *string `json:"interval,omitempty"`

	// labels are additional labels added to the ServiceMonitor.
	// +optional
	Labels map[string]string `json:"labels,omitempty"`
}

// PrometheusSpec defines Prometheus monitoring configuration.
type PrometheusSpec struct {
	// enabled controls whether Prometheus metrics are exposed.
	// +optional
	// +kubebuilder:default=true
	Enabled *bool `json:"enabled,omitempty"`

	// port is the Prometheus metrics port.
	// +optional
	// +kubebuilder:default=9090
	Port *int32 `json:"port,omitempty"`

	// serviceMonitor configures the Prometheus ServiceMonitor.
	// +optional
	ServiceMonitor *ServiceMonitorSpec `json:"serviceMonitor,omitempty"`
}

// L2BackendSpec defines an L2 storage backend.
type L2BackendSpec struct {
	// type is the adapter type name (mock, disk, redis, s3, p2p).
	Type string `json:"type"`

	// config is type-specific configuration as a free-form map.
	// +optional
	Config map[string]apiextensionsv1.JSON `json:"config,omitempty"`
}

// LMCacheEngineSpec defines the desired state of LMCacheEngine.
type LMCacheEngineSpec struct {
	// image defines the container image to use.
	// +optional
	Image *ImageSpec `json:"image,omitempty"`

	// imagePullSecrets is a list of references to secrets for pulling the image.
	// +optional
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`

	// server defines server configuration.
	// +optional
	Server *ServerSpec `json:"server,omitempty"`

	// l1 defines the L1 memory cache configuration.
	L1 L1BackendSpec `json:"l1"`

	// eviction defines the cache eviction configuration.
	// +optional
	Eviction *EvictionSpec `json:"eviction,omitempty"`

	// prometheus defines Prometheus monitoring configuration.
	// +optional
	Prometheus *PrometheusSpec `json:"prometheus,omitempty"`

	// l2Backends defines L2 storage backends.
	// +optional
	L2Backends []L2BackendSpec `json:"l2Backends,omitempty"`

	// resourceOverrides allows overriding auto-computed resource requirements.
	// +optional
	ResourceOverrides *corev1.ResourceRequirements `json:"resourceOverrides,omitempty"`

	// logLevel is the log level for the LMCache server.
	// +optional
	// +kubebuilder:default="INFO"
	// +kubebuilder:validation:Enum=DEBUG;INFO;WARNING;ERROR
	LogLevel *string `json:"logLevel,omitempty"`

	// nodeSelector determines which nodes get an LMCache instance.
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// affinity defines pod scheduling affinity rules.
	// +optional
	Affinity *corev1.Affinity `json:"affinity,omitempty"`

	// tolerations defines pod tolerations.
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// env defines additional environment variables.
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// volumes defines additional volumes.
	// +optional
	Volumes []corev1.Volume `json:"volumes,omitempty"`

	// volumeMounts defines additional volume mounts.
	// +optional
	VolumeMounts []corev1.VolumeMount `json:"volumeMounts,omitempty"`

	// podAnnotations are additional annotations added to pods.
	// +optional
	PodAnnotations map[string]string `json:"podAnnotations,omitempty"`

	// podLabels are additional labels added to pods.
	// +optional
	PodLabels map[string]string `json:"podLabels,omitempty"`

	// serviceAccountName is the name of the ServiceAccount to use.
	// +optional
	ServiceAccountName string `json:"serviceAccountName,omitempty"`

	// priorityClassName is the priority class for the pods.
	// +optional
	PriorityClassName string `json:"priorityClassName,omitempty"`

	// extraArgs are additional CLI flags appended to the server command.
	// They are appended last and can override any auto-generated flag.
	// +optional
	ExtraArgs []string `json:"extraArgs,omitempty"`
}

// EndpointStatus represents a single LMCache instance endpoint.
type EndpointStatus struct {
	// nodeName is the name of the node running this instance.
	NodeName string `json:"nodeName"`

	// hostIP is the IP address of the host.
	HostIP string `json:"hostIP"`

	// podName is the name of the pod.
	PodName string `json:"podName"`

	// port is the server port.
	Port int32 `json:"port"`

	// metricsPort is the Prometheus metrics port.
	MetricsPort int32 `json:"metricsPort"`

	// ready indicates whether this instance is ready.
	Ready bool `json:"ready"`
}

// LMCacheEngineStatus defines the observed state of LMCacheEngine.
type LMCacheEngineStatus struct {
	// phase is the overall phase of the LMCacheEngine.
	// +optional
	Phase string `json:"phase,omitempty"`

	// observedGeneration is the most recent generation observed.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// desiredInstances is the number of desired instances.
	// +optional
	DesiredInstances int32 `json:"desiredInstances,omitempty"`

	// readyInstances is the number of ready instances.
	// +optional
	ReadyInstances int32 `json:"readyInstances,omitempty"`

	// endpoints lists per-node connection info.
	// +optional
	Endpoints []EndpointStatus `json:"endpoints,omitempty"`

	// conditions represent the current state of the LMCacheEngine resource.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=lmc
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyInstances`
// +kubebuilder:printcolumn:name="Desired",type=integer,JSONPath=`.status.desiredInstances`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// LMCacheEngine is the Schema for the lmcacheengines API.
type LMCacheEngine struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitzero"`

	// spec defines the desired state of LMCacheEngine.
	// +required
	Spec LMCacheEngineSpec `json:"spec"`

	// status defines the observed state of LMCacheEngine.
	// +optional
	Status LMCacheEngineStatus `json:"status,omitzero"`
}

// +kubebuilder:object:root=true

// LMCacheEngineList contains a list of LMCacheEngine.
type LMCacheEngineList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitzero"`
	Items           []LMCacheEngine `json:"items"`
}

func init() {
	SchemeBuilder.Register(&LMCacheEngine{}, &LMCacheEngineList{})
}
