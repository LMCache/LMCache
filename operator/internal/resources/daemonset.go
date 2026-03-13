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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// BuildDaemonSet constructs a DaemonSet for the given LMCacheEngine.
func BuildDaemonSet(engine *lmcachev1alpha1.LMCacheEngine) *appsv1.DaemonSet {
	spec := &engine.Spec
	selectorLabels := SelectorLabels(engine.Name)
	podLabels := MergeLabels(StandardLabels(engine.Name), spec.PodLabels)
	podAnnotations := spec.PodAnnotations

	serverPort := derefInt32(getServerPort(spec), 5555)
	imgRepo := "lmcache/vllm-openai"
	imgTag := "latest"
	imgPullPolicy := corev1.PullIfNotPresent
	if spec.Image != nil {
		imgRepo = derefString(spec.Image.Repository, imgRepo)
		imgTag = derefString(spec.Image.Tag, imgTag)
		switch derefString(spec.Image.PullPolicy, "IfNotPresent") {
		case "Always":
			imgPullPolicy = corev1.PullAlways
		case "Never":
			imgPullPolicy = corev1.PullNever
		default:
			imgPullPolicy = corev1.PullIfNotPresent
		}
	}

	// Build env vars
	envVars := make([]corev1.EnvVar, 0, 2+len(spec.Env))
	envVars = append(envVars,
		corev1.EnvVar{
			Name:  "LMCACHE_LOG_LEVEL",
			Value: derefString(spec.LogLevel, "INFO"),
		},
		corev1.EnvVar{
			// Expose all GPUs without consuming device plugin resources.
			// LMCache needs GPU visibility for CUDA IPC, not compute ownership.
			Name:  "NVIDIA_VISIBLE_DEVICES",
			Value: "all",
		},
	)
	envVars = append(envVars, spec.Env...)

	// No emptyDir /dev/shm mount — hostIPC: true exposes the host's /dev/shm
	// directly. An emptyDir mount would shadow it and break CUDA IPC between
	// LMCache and vLLM pods (cudaIpcOpenMemHandle requires shared /dev/shm).
	volumes := append([]corev1.Volume{}, spec.Volumes...)
	volumeMounts := append([]corev1.VolumeMount{}, spec.VolumeMounts...)

	// Build container args
	containerArgs := BuildContainerArgs(spec)

	// Probes
	tcpProbe := &corev1.TCPSocketAction{
		Port: intstr.FromInt32(serverPort),
	}

	startupProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			TCPSocket: tcpProbe,
		},
		InitialDelaySeconds: 5,
		PeriodSeconds:       5,
		FailureThreshold:    30,
	}

	livenessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			TCPSocket: tcpProbe,
		},
		PeriodSeconds: 10,
	}

	readinessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			TCPSocket: tcpProbe,
		},
		PeriodSeconds: 5,
	}

	// Container ports
	containerPorts := []corev1.ContainerPort{
		{
			Name:          "server",
			ContainerPort: serverPort,
			Protocol:      corev1.ProtocolTCP,
		},
	}

	// Add metrics port if prometheus is enabled
	promEnabled := true
	promPort := int32(9090)
	if spec.Prometheus != nil {
		promEnabled = derefBool(spec.Prometheus.Enabled, true)
		promPort = derefInt32(spec.Prometheus.Port, 9090)
	}
	if promEnabled {
		containerPorts = append(containerPorts, corev1.ContainerPort{
			Name:          "metrics",
			ContainerPort: promPort,
			Protocol:      corev1.ProtocolTCP,
		})
	}

	ds := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      engine.Name,
			Namespace: engine.Namespace,
			Labels:    StandardLabels(engine.Name),
		},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: selectorLabels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      podLabels,
					Annotations: podAnnotations,
				},
				Spec: corev1.PodSpec{
					HostIPC:            true,
					ServiceAccountName: spec.ServiceAccountName,
					PriorityClassName:  spec.PriorityClassName,
					NodeSelector:       spec.NodeSelector,
					Affinity:           spec.Affinity,
					Tolerations:        spec.Tolerations,
					ImagePullSecrets:   spec.ImagePullSecrets,
					Containers: []corev1.Container{
						{
							Name:            "lmcache",
							Image:           fmt.Sprintf("%s:%s", imgRepo, imgTag),
							ImagePullPolicy: imgPullPolicy,
							Command:         []string{"/opt/venv/bin/python3", "-m", "lmcache.v1.multiprocess.server"},
							Args:            containerArgs,
							Ports:           containerPorts,
							Env:             envVars,
							Resources:       ComputeResources(spec),
							VolumeMounts:    volumeMounts,
							StartupProbe:    startupProbe,
							LivenessProbe:   livenessProbe,
							ReadinessProbe:  readinessProbe,
						},
					},
					Volumes: volumes,
				},
			},
		},
	}

	return ds
}
