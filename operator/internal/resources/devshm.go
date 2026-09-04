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

// This file holds the host /dev/shm wiring shared by the engine DaemonSet
// builder (buildDaemonSetCore) and the pod-injection webhooks
// (webhook.applyIPCSharing): the hostPath volume/mount that gives two
// same-node pods the shared /dev/shm tmpfs cross-pod CUDA IPC needs, plus
// the guards that keep the injection away from user-supplied wiring.

package resources

import (
	corev1 "k8s.io/api/core/v1"
)

const (
	// devShmVolumeName is the pod volume name for the host /dev/shm mount used
	// for cross-pod CUDA IPC when hostIPC is disabled (the default).
	devShmVolumeName = "lmcache-dev-shm"

	// devShmPath is both the hostPath source and the in-container mount path of
	// the shared /dev/shm tmpfs.
	devShmPath = "/dev/shm"
)

// BuildDevShmVolume returns the hostPath volume exposing the host's /dev/shm
// tmpfs to the pod. Sharing this tmpfs is what cross-pod CUDA IPC actually
// needs: PyTorch's CUDA IPC handles reference a shared-memory ref-counter file
// in /dev/shm that the receiving process opens by name. Mounting the host's
// /dev/shm (rather than sharing the whole host IPC namespace via hostIPC) is
// the narrower grant. Also consumed by the injection webhooks so the engine
// and vLLM pods always share the same tmpfs.
func BuildDevShmVolume() corev1.Volume {
	hostPathType := corev1.HostPathDirectory
	return corev1.Volume{
		Name: devShmVolumeName,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: devShmPath,
				Type: &hostPathType,
			},
		},
	}
}

// BuildDevShmVolumeMount returns the container mount for BuildDevShmVolume,
// mounting the host's /dev/shm at /dev/shm.
func BuildDevShmVolumeMount() corev1.VolumeMount {
	return corev1.VolumeMount{
		Name:      devShmVolumeName,
		MountPath: devShmPath,
	}
}

// HasDevShmMount reports whether mounts already contains a mount at /dev/shm
// (any volume name). Callers use it to avoid double-mounting when the user
// supplies their own /dev/shm volume.
func HasDevShmMount(mounts []corev1.VolumeMount) bool {
	for i := range mounts {
		if mounts[i].MountPath == devShmPath {
			return true
		}
	}
	return false
}

// HasDevShmVolume reports whether volumes already contains a volume named
// "lmcache-dev-shm" (regardless of its source). Callers skip the default injection in
// that case: appending a same-named volume would make the pod spec invalid,
// and mounting a user-owned volume of unknown source at /dev/shm could
// silently shadow the host tmpfs.
func HasDevShmVolume(volumes []corev1.Volume) bool {
	for i := range volumes {
		if volumes[i].Name == devShmVolumeName {
			return true
		}
	}
	return false
}
