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
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestStartupProbeFailureThreshold(t *testing.T) {
	cases := []struct {
		name     string
		l1SizeGB float64
		want     int32
	}{
		{"tiny is floored", 10, 30},
		{"fractional is floored", 0.5, 30},
		{"at floor boundary", 150, 30}, // 150/5 = 30, not greater than the floor
		{"just above floor", 155, 31},  // 155/5 = 31
		{"300GB", 300, 60},
		{"1200GB", 1200, 240},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := startupProbeFailureThreshold(tc.l1SizeGB); got != tc.want {
				t.Fatalf("startupProbeFailureThreshold(%v) = %d, want %d",
					tc.l1SizeGB, got, tc.want)
			}
		})
	}
}

func TestBuildDaemonSet_StartupProbeScalesWithL1(t *testing.T) {
	cases := []struct {
		name              string
		l1SizeGB          float64
		wantFailThreshold int32
	}{
		{"small L1 keeps the default window", 10, 30},
		{"large L1 gets a proportional window", 1200, 240},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			engine := minimalEngine()
			engine.Spec.L1.SizeGB = tc.l1SizeGB

			ds := BuildDaemonSet(engine)
			containers := ds.Spec.Template.Spec.Containers
			if len(containers) == 0 {
				t.Fatalf("expected at least one container in the DaemonSet")
			}
			probe := containers[0].StartupProbe
			if probe == nil {
				t.Fatalf("expected a startup probe on the engine container")
			}
			if probe.FailureThreshold != tc.wantFailThreshold {
				t.Fatalf("StartupProbe.FailureThreshold = %d, want %d",
					probe.FailureThreshold, tc.wantFailThreshold)
			}
			// Only the threshold scales; the period and initial delay are unchanged.
			if probe.PeriodSeconds != 5 {
				t.Fatalf("StartupProbe.PeriodSeconds = %d, want 5", probe.PeriodSeconds)
			}
			if probe.InitialDelaySeconds != 5 {
				t.Fatalf("StartupProbe.InitialDelaySeconds = %d, want 5",
					probe.InitialDelaySeconds)
			}
		})
	}
}

func TestBuildDaemonSet_SecurityContextDefaultUnchanged(t *testing.T) {
	// No spec.securityContext: the generated context must stay exactly the
	// pre-field output — only the privileged bit, defaulting to false — and
	// no pod-level security context.
	ds := BuildDaemonSet(minimalEngine())
	sc := ds.Spec.Template.Spec.Containers[0].SecurityContext
	privileged := false
	want := &corev1.SecurityContext{Privileged: &privileged}
	if sc == nil || *sc != *want {
		t.Fatalf("default securityContext must stay {privileged: false}, got %+v", sc)
	}
	if ds.Spec.Template.Spec.SecurityContext != nil {
		t.Fatalf("no podSecurityContext must be emitted when unset, got %+v",
			ds.Spec.Template.Spec.SecurityContext)
	}
}

func TestBuildDaemonSet_SecurityContextPassthrough(t *testing.T) {
	// The official images run as the non-root lmcache user (uid 1000);
	// runAsUser: 0 (e.g. to reuse a root-populated mount) and fsGroup: 1000
	// (the Kubernetes fix for root-owned volume contents such as the 0400
	// L2 master key) must land on the container and the pod respectively.
	uid := int64(0)
	fsGroup := int64(1000)
	privileged := true
	engine := minimalEngine()
	engine.Spec.SecurityContext = &corev1.SecurityContext{RunAsUser: &uid}
	engine.Spec.PodSecurityContext = &corev1.PodSecurityContext{FSGroup: &fsGroup}
	engine.Spec.Privileged = &privileged

	ds := BuildDaemonSet(engine)
	sc := ds.Spec.Template.Spec.Containers[0].SecurityContext
	if sc == nil || sc.RunAsUser == nil || *sc.RunAsUser != 0 {
		t.Fatalf("securityContext.runAsUser must pass through to the engine container, got %+v", sc)
	}
	if sc.Privileged == nil || !*sc.Privileged {
		t.Fatalf("spec.privileged must force the privileged bit, got %+v", sc)
	}
	podSC := ds.Spec.Template.Spec.SecurityContext
	if podSC == nil || podSC.FSGroup == nil || *podSC.FSGroup != 1000 {
		t.Fatalf("podSecurityContext.fsGroup must land on the pod, got %+v", podSC)
	}
}

func TestBuildDaemonSet_SecurityContextPrivilegedWins(t *testing.T) {
	// The builder is the last line of defense for the privileged contract:
	// validation rejects securityContext.privileged, and even if a stray
	// value reaches the builder it must not take effect.
	stray := true
	engine := minimalEngine()
	engine.Spec.SecurityContext = &corev1.SecurityContext{Privileged: &stray}

	ds := BuildDaemonSet(engine)
	sc := ds.Spec.Template.Spec.Containers[0].SecurityContext
	if sc == nil || sc.Privileged == nil || *sc.Privileged {
		t.Fatalf("stray privileged inside securityContext must never take effect, got %+v", sc)
	}
}
