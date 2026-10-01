//go:build e2e
// +build e2e

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

package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"

	"github.com/LMCache/LMCache/test/utils"
)

// Validate that applying an LMCacheEngine produces the documented K8s
// objects with the documented shape. Two fixtures exercise the same
// assertions: lmc_minimal (defaults) and lmc_custom_port (port=6555,
// chunkSize=128). The minimal fixture additionally diffs the connection
// ConfigMap against a checked-in golden file so schema drift in
// kv-transfer-config.json fails loudly.
//
// The first It block is a harness sanity check — kept intentionally
// cheap so a broken helper fails before the richer reconciliation
// specs run.
var _ = Describe("LMCacheEngine smoke (no-GPU)", Ordered, func() {
	var (
		ctx    context.Context
		nsName string
	)

	BeforeEach(func() {
		ctx = context.Background()
		nsName = createTestNamespace(ctx)
	})

	AfterEach(func() {
		recordOnFailure(nsName)
	})

	It("reconciles a minimal CR and cleans up on delete (harness check)", func() {
		lmc, err := utils.NewLMCFromFixture("lmc_minimal.yaml", nsName, "smoke-harness")
		Expect(err).NotTo(HaveOccurred())

		By("applying the minimal LMCacheEngine")
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		By("waiting for the operator to observe the spec and validate the config")
		Expect(utils.WaitLMCReconciled(ctx, k8sClient,
			engineKey(nsName, "smoke-harness"),
			60*time.Second,
		)).To(Succeed())

		By("ensuring the connection ConfigMap has been produced")
		cfg, err := utils.GetConnectionConfig(ctx, k8sClient, engineKey(nsName, "smoke-harness"))
		Expect(err).NotTo(HaveOccurred())
		Expect(cfg.KVConnector).To(Equal("LMCacheMPConnector"))
		Expect(cfg.KVConnectorModulePath).To(Equal("lmcache.integration.vllm.lmcache_mp_connector"))

		By("deleting the CR and waiting for owned objects to be garbage-collected")
		Expect(utils.DeleteLMCAndWaitGC(ctx, k8sClient,
			engineKey(nsName, "smoke-harness"),
			60*time.Second,
		)).To(Succeed())
	})

	It("reconciles a minimal CR into the documented K8s artifacts", func() {
		// Use the fixture's own metadata.name so the golden file's
		// hostname matches verbatim ("smoke-minimal.<ns>.svc...").
		lmc, err := utils.NewLMCFromFixture("lmc_minimal.yaml", nsName, "")
		Expect(err).NotTo(HaveOccurred())
		Expect(lmc.Name).To(Equal("smoke-minimal"))

		By("applying the minimal LMCacheEngine")
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)

		By("waiting for the operator to observe the spec")
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("validating the DaemonSet pod template shape")
		ds := &appsv1.DaemonSet{}
		Expect(k8sClient.Get(ctx, types.NamespacedName(key), ds)).To(Succeed())
		assertDaemonSetShape(ds, 5555 /* default server port */)

		By("validating the lookup Service shape")
		svc := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName(key), svc)).To(Succeed())
		assertLookupServiceShape(svc, 5555)

		By("validating the connection ConfigMap matches the documented contract")
		cfg, err := utils.GetConnectionConfig(ctx, k8sClient, key)
		Expect(err).NotTo(HaveOccurred())
		Expect(cfg.KVConnector).To(Equal("LMCacheMPConnector"))
		Expect(cfg.KVConnectorModulePath).To(Equal("lmcache.integration.vllm.lmcache_mp_connector"))
		Expect(cfg.KVRole).To(Equal("kv_both"))
		Expect(cfg.KVConnectorExtraConfig.Host).To(Equal(
			fmt.Sprintf("tcp://smoke-minimal.%s.svc.cluster.local", nsName)))
		Expect(cfg.KVConnectorExtraConfig.Port).To(Equal("5555"))

		By("diffing the raw kv-transfer-config.json against the golden snapshot")
		assertGoldenKvTransferConfig(ctx, key, "kv_transfer_config_minimal.json", nsName)
	})

	It("propagates server.port and server.chunkSize into args, Service, and ConfigMap", func() {
		lmc, err := utils.NewLMCFromFixture("lmc_custom_port.yaml", nsName, "")
		Expect(err).NotTo(HaveOccurred())

		By("applying the custom-port LMCacheEngine")
		Expect(utils.ApplyLMC(ctx, k8sClient, lmc)).To(Succeed())

		key := engineKey(nsName, lmc.Name)

		By("waiting for the operator to observe the spec")
		Expect(utils.WaitLMCReconciled(ctx, k8sClient, key, 60*time.Second)).To(Succeed())

		By("validating the DaemonSet pod template")
		ds := &appsv1.DaemonSet{}
		Expect(k8sClient.Get(ctx, types.NamespacedName(key), ds)).To(Succeed())
		assertDaemonSetShape(ds, 6555 /* spec.server.port */)

		args := containerArgs(ds)
		Expect(argValue(args, "--port")).To(Equal("6555"))
		Expect(argValue(args, "--chunk-size")).To(Equal("128"))

		By("validating the lookup Service uses the custom port")
		svc := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName(key), svc)).To(Succeed())
		assertLookupServiceShape(svc, 6555)

		By("validating the ConfigMap reflects the custom port")
		cfg, err := utils.GetConnectionConfig(ctx, k8sClient, key)
		Expect(err).NotTo(HaveOccurred())
		Expect(cfg.KVConnectorExtraConfig.Port).To(Equal("6555"))
	})
})

// assertDaemonSetShape verifies the operator's auto-injected pod-level
// settings: the /dev/shm hostPath mount for CUDA IPC (hostIPC is opt-in via
// spec.hostIPC and defaults to false), runtimeClassName=nvidia, a container
// security context that is non-privileged by default (privileged is opt-in via
// spec.privileged), --host 0.0.0.0 always present in container args, and the
// absence of any emptyDir /dev/shm volume that would shadow the host's
// /dev/shm and break CUDA IPC.
func assertDaemonSetShape(ds *appsv1.DaemonSet, expectedServerPort int32) {
	GinkgoHelper()
	pod := ds.Spec.Template.Spec
	Expect(pod.HostIPC).To(BeFalse(), "pod.hostIPC must default to false (opt-in via spec.hostIPC)")
	Expect(pod.RuntimeClassName).NotTo(BeNil(), "pod.runtimeClassName must be set")
	Expect(*pod.RuntimeClassName).To(Equal("nvidia"))
	Expect(pod.Containers).To(HaveLen(1))

	c := pod.Containers[0]
	Expect(c.SecurityContext).NotTo(BeNil(), "container.securityContext must be set")
	Expect(c.SecurityContext.Privileged).NotTo(BeNil())
	Expect(*c.SecurityContext.Privileged).To(BeFalse(),
		"container.privileged must default to false (opt-in via spec.privileged)")
	Expect(c.Args).To(ContainElements("--host", "0.0.0.0"))
	Expect(argValue(c.Args, "--port")).To(Equal(fmt.Sprintf("%d", expectedServerPort)))

	// The host's /dev/shm is shared into the container via a hostPath mount —
	// that (not the IPC namespace) is what cudaIpcOpenMemHandle between the
	// LMCache and vLLM pods needs. An emptyDir there would shadow
	// the host tmpfs and break CUDA IPC.
	var shmMount *corev1.VolumeMount
	for i, vm := range c.VolumeMounts {
		if vm.MountPath == "/dev/shm" {
			shmMount = &c.VolumeMounts[i]
		}
	}
	Expect(shmMount).NotTo(BeNil(), "missing /dev/shm volume mount on lmcache container")
	var shmVol *corev1.Volume
	for i, v := range pod.Volumes {
		Expect(v.EmptyDir).To(BeNil(),
			"unexpected emptyDir volume %q — an emptyDir at /dev/shm shadows the host tmpfs", v.Name)
		if v.Name == shmMount.Name {
			shmVol = &pod.Volumes[i]
		}
	}
	Expect(shmVol).NotTo(BeNil(), "no volume backs the /dev/shm mount")
	Expect(shmVol.HostPath).NotTo(BeNil(), "/dev/shm volume must be a hostPath")
	Expect(shmVol.HostPath.Path).To(Equal("/dev/shm"))
}

// assertLookupServiceShape checks the node-local discovery Service has
// internalTrafficPolicy=Local (so kube-proxy routes only to the on-node
// LMCache pod) and exposes the spec'd server port.
func assertLookupServiceShape(svc *corev1.Service, expectedServerPort int32) {
	GinkgoHelper()
	Expect(svc.Spec.InternalTrafficPolicy).NotTo(BeNil())
	Expect(*svc.Spec.InternalTrafficPolicy).To(Equal(corev1.ServiceInternalTrafficPolicyLocal))
	var found bool
	for _, p := range svc.Spec.Ports {
		if p.Name == "server" {
			found = true
			Expect(p.Port).To(Equal(expectedServerPort))
		}
	}
	Expect(found).To(BeTrue(), "lookup Service must expose a port named 'server'")
}

// assertGoldenKvTransferConfig compares the on-cluster ConfigMap data
// to the embedded golden snapshot. The placeholder __NAMESPACE__ in the
// golden file is substituted with the live test namespace before
// comparison so the diff sees a single reproducible string.
func assertGoldenKvTransferConfig(ctx context.Context, key types.NamespacedName, goldenName, namespace string) {
	GinkgoHelper()
	golden, err := utils.LoadGolden(goldenName)
	Expect(err).NotTo(HaveOccurred(), "load golden %s", goldenName)
	expected := strings.ReplaceAll(string(golden), "__NAMESPACE__", namespace)

	cm := &corev1.ConfigMap{}
	cmKey := types.NamespacedName{Namespace: key.Namespace, Name: key.Name + "-connection"}
	Expect(k8sClient.Get(ctx, cmKey, cm)).To(Succeed())
	actual, ok := cm.Data["kv-transfer-config.json"]
	Expect(ok).To(BeTrue(), "ConfigMap missing kv-transfer-config.json")

	// Normalize both sides through json.Unmarshal+Marshal to absorb any
	// trailing-newline / line-ending differences while still catching
	// genuine schema drift. We compare on the canonical re-encoding.
	Expect(canonicalJSON(actual)).To(Equal(canonicalJSON(expected)),
		"kv-transfer-config.json drift vs %s\nactual:\n%s\nexpected:\n%s",
		goldenName, actual, expected)
}

func canonicalJSON(s string) string {
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		// Return as-is so the caller's diff reveals the malformed input.
		return s
	}
	out, _ := json.MarshalIndent(v, "", "  ")
	return string(out)
}
