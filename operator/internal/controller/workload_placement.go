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

package controller

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"sort"
	"strconv"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

// Workload-aware engine placement: the controller mirrors a per-engine label
// (resources.WorkloadPlacementLabel) onto every node hosting a bound pod
// matching spec.workloadAwareScheduling.workloadSelector, and the DaemonSet
// nodeSelector (resources.EffectiveNodeSelector) requires it. Engine pods
// therefore only exist where the workload runs; a node loses its label — and
// its engine pod — releaseDelaySeconds after the node is FIRST OBSERVED
// without the workload. Anchoring the window on observed absence (rather than
// last-seen presence) means a controller interruption that misses the actual
// departure still grants the full delay, and a labeled node whose
// bookkeeping predates this operator gets a full window from its first
// handling rather than being cleaned up instantly.

// removalRetryDelay backs the requeue scheduled when an expired label
// cannot be removed (sticky-engine path): quick enough to converge a
// transient node-patch failure, far enough not to spin.
const removalRetryDelay = 30 * time.Second

// workloadSelectorFor parses spec.workloadAwareScheduling.workloadSelector.
// Malformed selectors error here. Validation runs at RECONCILE time
// (validateAndSetCondition — the operator has no validating admission
// webhook, only mutating pod injectors), so the API server accepts
// invalid selectors and the controller's ValidateSpec is the gate: a
// parse failure here means the engine already went Failed via the
// ConfigValid condition; the error is belt-and-suspenders.
func workloadSelectorFor(engine *lmcachev1alpha1.LMCacheEngine) (labels.Selector, error) {
	parsed, err := metav1.LabelSelectorAsSelector(engine.Spec.WorkloadAwareScheduling.WorkloadSelector)
	if err != nil {
		return nil, fmt.Errorf("parsing workloadSelector: %w", err)
	}
	return parsed, nil
}

// workloadMatches reports whether podLabels belong to the engine's workload.
// Malformed selectors never match.
func workloadMatches(engine *lmcachev1alpha1.LMCacheEngine, podLabels map[string]string) bool {
	sel, err := workloadSelectorFor(engine)
	if err != nil {
		return false
	}
	return sel.Matches(labels.Set(podLabels))
}

// matchingWorkloadPods returns the workload pods that count for placement
// AND for the dynamic-resource-allocation tier multiplier: matching
// spec.workloadAwareScheduling.workloadSelector, bound (spec.nodeName
// set — the engine must be booting while the workload still loads
// weights), not terminating, not Failed/Succeeded. Node selection and
// GPU summation must consume this ONE filtered list so the two can never
// drift (e.g. finished Job pods would otherwise inflate the tier).
func matchingWorkloadPods(ctx context.Context, c client.Client, engine *lmcachev1alpha1.LMCacheEngine) ([]*corev1.Pod, error) {
	sel, err := workloadSelectorFor(engine)
	if err != nil {
		return nil, err
	}
	pods := &corev1.PodList{}
	if err := c.List(ctx, pods,
		client.InNamespace(engine.Namespace),
		client.MatchingLabelsSelector{Selector: sel},
	); err != nil {
		return nil, err
	}
	out := make([]*corev1.Pod, 0, len(pods.Items))
	for i := range pods.Items {
		pod := &pods.Items[i]
		if pod.Spec.NodeName == "" || pod.DeletionTimestamp != nil {
			continue
		}
		if pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodSucceeded {
			continue
		}
		out = append(out, pod)
	}
	return out, nil
}

// reconcileWorkloadPlacement syncs the managed node labels for one engine
// and returns the wait until the next timed transition — a node exiting
// its release window or a tier change leaving its hysteresis window
// (zero when nothing is pending), which the caller turns into a
// RequeueAfter.
//
// The label value is the node's tier multiplier: the base "1" unless
// dynamic resource allocation is enabled, in which case it is derived
// from the workload pods' GPU requests. A FIRST label (placement) is
// applied immediately; a CHANGED tier is applied only once it has been
// continuously observed for tierChangeDelaySeconds — the relabel
// restarts the engine pod and drops its warm L1, so transient GPU-sum
// changes (workload rolling updates, surge pods) must not flap it.
//
// Failure direction: an error adding a needed label fails loudly (the
// workload would lack an engine); a node that must lose its label but cannot
// is retried — sticky engines are the safe side of the asymmetry, and the
// only side affinity-only placement ever had.
func reconcileWorkloadPlacement(
	ctx context.Context, c client.Client, engine *lmcachev1alpha1.LMCacheEngine, now time.Time,
) (time.Duration, map[int32]struct{}, error) {
	key := resources.WorkloadPlacementLabel(engine.Namespace, engine.Name)

	if !engine.WorkloadSchedulingEnabled() {
		return 0, nil, reconcileWorkloadPlacementOff(ctx, c, engine, key)
	}

	pods, err := matchingWorkloadPods(ctx, c, engine)
	if err != nil {
		return 0, nil, fmt.Errorf("listing workload pods: %w", err)
	}
	desired := sortedNodeNames(pods)
	desiredSet := make(map[string]struct{}, len(desired))
	for _, n := range desired {
		desiredSet[n] = struct{}{}
	}

	// Per-node tier multipliers from the same filtered pod list that
	// drove node selection — the filter cannot drift between the two.
	tiers := computeNodeTiers(engine, pods)

	labeled, err := listLabeledNodes(ctx, c, key)
	if err != nil {
		return 0, nil, err
	}

	// Copy-on-write bookkeeping: entries are only persisted when a pass
	// actually changed them, guarding against chronic status churn.
	absentSince := maps.Clone(engine.Status.NodeAbsentSince)
	if absentSince == nil {
		absentSince = map[string]metav1.Time{}
	}
	tierChanges := maps.Clone(engine.Status.NodeTierChanges)
	if tierChanges == nil {
		tierChanges = map[string]lmcachev1alpha1.NodeTierChange{}
	}

	if err := ensureDesiredWorkloadLabels(ctx, c, labeled, desired, desiredSet, key, tiers, absentSince, tierChanges, now, engine); err != nil {
		return 0, nil, err
	}

	nextExpiry := releaseExpiredWorkloadNodes(ctx, c, labeled, desiredSet, absentSince, engine.WorkloadReleaseDelay(), now, key)
	// Tier-change bookkeeping dies with the release.
	for nodeName := range labeled {
		if _, ok := desiredSet[nodeName]; !ok {
			delete(tierChanges, nodeName)
		}
	}

	pruneStaleWorkloadAbsence(absentSince, labeled, desiredSet)
	pruneStaleTierChanges(tierChanges, labeled, desiredSet)

	nextTierApply := nextTierChange(tierChanges, engine, now)

	if err := updateWorkloadPlacementStatus(ctx, c, engine, key, desiredSet, absentSince, tierChanges); err != nil {
		return 0, nil, err
	}

	// Effective tier set: what the tier DaemonSets must cover RIGHT NOW —
	// the label values the placement pass left on the nodes (hysteresis
	// may hold some at their old tier). Computed from the in-memory node
	// objects this pass just patched, NOT by re-reading node labels
	// through the (possibly stale) watch cache in the same reconcile.
	effective := make(map[int32]struct{}, len(labeled)+len(desiredSet))
	for _, node := range labeled {
		if v, err := strconv.Atoi(node.Labels[key]); err == nil && v >= 1 {
			effective[int32(v)] = struct{}{}
		}
	}
	// First-placement nodes are not in `labeled`, so their desired tier
	// applies; a NotFound ghost dropped from desiredSet cannot activate a tier DS.
	for node := range desiredSet {
		if _, ok := labeled[node]; ok {
			continue
		}
		if tier, ok := tiers[node]; ok {
			effective[tier] = struct{}{}
		} else {
			effective[1] = struct{}{}
		}
	}

	next := earliest(nextExpiry, nextTierApply)
	if next.IsZero() {
		return 0, effective, nil
	}
	return next.Sub(now), effective, nil
}

// sortedNodeNames deduplicates and sorts the nodes of the given pods.
func sortedNodeNames(pods []*corev1.Pod) []string {
	seen := map[string]struct{}{}
	nodes := make([]string, 0, len(pods))
	for _, pod := range pods {
		if _, dup := seen[pod.Spec.NodeName]; dup {
			continue
		}
		seen[pod.Spec.NodeName] = struct{}{}
		nodes = append(nodes, pod.Spec.NodeName)
	}
	sort.Strings(nodes)
	return nodes
}

// computeNodeTiers derives each node's tier multiplier from the GPU
// requests of its workload pods: multiplier = min(maxMultiplier,
// max(1, ceil(totalGPUs / gpusPerUnit))). Only meaningful when DRA is
// enabled; without it every node is tier 1. Init containers and ephemeral
// containers do not count: init containers release their GPUs before app
// containers run, so they misrepresent steady-state occupancy.
func computeNodeTiers(engine *lmcachev1alpha1.LMCacheEngine, pods []*corev1.Pod) map[string]int32 {
	tiers := make(map[string]int32, len(pods))
	dra := engine.Spec.WorkloadAwareScheduling.DynamicResourceAllocation
	if dra == nil || !dra.Enabled {
		return tiers
	}
	resourceName := corev1.ResourceName("nvidia.com/gpu")
	if engine.Spec.GPUVendor != nil && *engine.Spec.GPUVendor == lmcachev1alpha1.GPUVendorAMD {
		resourceName = "amd.com/gpu"
	}
	gpusPerUnit := float64(dra.EffectiveDRAGPUsPerUnit())
	maxMultiplier := dra.EffectiveDRAMaxMultiplier()

	gpus := map[string]float64{}
	for _, pod := range pods {
		total := 0.0
		for i := range pod.Spec.Containers {
			if q, ok := pod.Spec.Containers[i].Resources.Requests[resourceName]; ok {
				total += q.AsApproximateFloat64()
			}
		}
		gpus[pod.Spec.NodeName] += total
	}
	for node, total := range gpus {
		m := int32(math.Ceil(total / gpusPerUnit))
		m = max(int32(1), min(m, maxMultiplier))
		tiers[node] = m
	}
	return tiers
}

// nextTierChange returns the earliest pending tier-change application
// deadline (zero when none), for the caller's requeue.
func nextTierChange(tierChanges map[string]lmcachev1alpha1.NodeTierChange, engine *lmcachev1alpha1.LMCacheEngine, now time.Time) time.Time {
	dra := engine.Spec.WorkloadAwareScheduling.DynamicResourceAllocation
	if dra == nil || !dra.Enabled {
		return time.Time{}
	}
	delay := dra.EffectiveDRATierChangeDelay()
	var next time.Time
	for _, pending := range tierChanges {
		deadline := pending.Since.Add(delay)
		if now.Before(deadline) && (next.IsZero() || deadline.Before(next)) {
			next = deadline
		}
	}
	return next
}

// pruneStaleTierChanges drops tier-change bookkeeping for nodes that are
// neither labeled nor desired.
func pruneStaleTierChanges(
	tierChanges map[string]lmcachev1alpha1.NodeTierChange,
	labeled map[string]*corev1.Node,
	desiredSet map[string]struct{},
) {
	for nodeName := range tierChanges {
		_, isLabeled := labeled[nodeName]
		_, isDesired := desiredSet[nodeName]
		if !isLabeled && !isDesired {
			delete(tierChanges, nodeName)
		}
	}
}

// earliest returns the earlier of two zeroable times.
func earliest(a, b time.Time) time.Time {
	switch {
	case a.IsZero():
		return b
	case b.IsZero():
		return a
	case b.Before(a):
		return b
	default:
		return a
	}
}

// reconcileWorkloadPlacementOff handles a disabled feature: strip residue
// from any previous enable and reset the bookkeeping, only writing status
// when something actually changed.
func reconcileWorkloadPlacementOff(ctx context.Context, c client.Client, engine *lmcachev1alpha1.LMCacheEngine, key string) error {
	if err := purgeWorkloadLabels(ctx, c, key); err != nil {
		return err
	}
	changed := false
	if engine.Status.NodeAbsentSince != nil {
		engine.Status.NodeAbsentSince = nil
		changed = true
	}
	if engine.Status.NodeTierChanges != nil {
		engine.Status.NodeTierChanges = nil
		changed = true
	}
	if len(engine.Status.WorkloadNodes) > 0 {
		engine.Status.WorkloadNodes = nil
		changed = true
	}
	if engine.Status.WorkloadLabelKey != "" {
		engine.Status.WorkloadLabelKey = ""
		changed = true
	}
	if !changed {
		return nil
	}
	return c.Status().Update(ctx, engine)
}

// listLabeledNodes returns the nodes currently carrying the managed label
// (any value: with dynamic resource allocation the value is the node's
// tier multiplier). A key-presence scan, not a value match, because the
// label value varies ("1", "2", ...) once dynamic resource allocation
// is enabled.
func listLabeledNodes(ctx context.Context, c client.Client, key string) (map[string]*corev1.Node, error) {
	nodeList := &corev1.NodeList{}
	if err := c.List(ctx, nodeList); err != nil {
		return nil, fmt.Errorf("listing nodes: %w", err)
	}
	labeled := map[string]*corev1.Node{}
	for i := range nodeList.Items {
		if _, ok := nodeList.Items[i].Labels[key]; ok {
			labeled[nodeList.Items[i].Name] = &nodeList.Items[i]
		}
	}
	return labeled, nil
}

// ensureDesiredWorkloadLabels labels every desired node with its tier
// value and clears absence bookkeeping the moment the workload returns
// to a node.
//
// Tier handling: a node NOT yet labeled gets the label immediately
// (placement — delaying it would leave the workload engineless); a node
// already labeled at a different tier is relabeled only once the change
// has been continuously observed for tierChangeDelaySeconds
// (hysteresis: the relabel restarts the engine pod and drops its warm L1,
// and workload rolling updates transiently change the GPU sum). An
// unlabeled DRA-off node is (re)labeled with the base value "1".
//
// Add failures are collected per node rather than returned eagerly so one
// stuck node does not prevent labeling the others; the joined error is
// returned once the pass completes — loud, with total progress. A Get
// NotFound is skipped rather than failed: a pod can be bound to a node that
// was just deleted before pod GC catches up, and that transient must not
// wedge the whole reconcile.
func ensureDesiredWorkloadLabels(
	ctx context.Context, c client.Client,
	labeled map[string]*corev1.Node,
	desired []string,
	desiredSet map[string]struct{},
	key string,
	tiers map[string]int32,
	absentSince map[string]metav1.Time,
	tierChanges map[string]lmcachev1alpha1.NodeTierChange,
	now time.Time,
	engine *lmcachev1alpha1.LMCacheEngine,
) error {
	dra := engine.Spec.WorkloadAwareScheduling.DynamicResourceAllocation
	tierDelay := time.Duration(0)
	if dra != nil && dra.Enabled {
		tierDelay = dra.EffectiveDRATierChangeDelay()
	}

	var errs []error
	for _, node := range desired {
		value := resources.WorkloadLabelValueBase
		targetTier := int32(1)
		if tier, ok := tiers[node]; ok {
			value = resources.TierLabelValue(tier)
			targetTier = tier
		}

		var target *corev1.Node
		if n, ok := labeled[node]; ok {
			target = n
		} else {
			fresh := &corev1.Node{}
			if err := c.Get(ctx, client.ObjectKey{Name: node}, fresh); err != nil {
				if apierrors.IsNotFound(err) {
					// Pod bound to a deleted node (pre-GC race): drop the
					// entry from the visible set too; pod GC will remove
					// the stale pod from the next desired listing.
					delete(desiredSet, node)
					continue
				}
				errs = append(errs, err)
				continue
			}
			target = fresh
		}

		if target.Labels[key] == value {
			// Already at the desired tier: any pending change is moot.
			delete(tierChanges, node)
			delete(absentSince, node)
			continue
		}
		if _, wasLabeled := labeled[node]; !wasLabeled {
			// First placement — label immediately.
			errs = append(errs, addWorkloadLabel(ctx, c, target, key, value))
			delete(absentSince, node)
			delete(tierChanges, node)
			continue
		}
		// Labeled at a different tier: apply after the hysteresis window.
		// Non-numeric residue (e.g. the "true" value of pre-DRA
		// intermediate builds) is not a tier change: relabel immediately,
		// or the base DS ("1") would not match the node during the wait.
		if current, ok := target.Labels[key]; ok && current != "" && !isNumericLabel(current) {
			errs = append(errs, addWorkloadLabel(ctx, c, target, key, value))
			delete(tierChanges, node)
			delete(absentSince, node)
			continue
		}
		// The window is anchored on the TARGET tier: a changed target
		// (1->2 aging out, then the sum moves to 3) restarts the window
		// so 3 is never applied on 2's stale timestamp.
		pending, tracked := tierChanges[node]
		if !tracked || pending.Tier != targetTier {
			pending = lmcachev1alpha1.NodeTierChange{Tier: targetTier, Since: metav1.Time{Time: now}}
			tierChanges[node] = pending
		}
		if now.Before(pending.Since.Add(tierDelay)) {
			delete(absentSince, node)
			continue // still inside the tier-change window
		}
		errs = append(errs, addWorkloadLabel(ctx, c, target, key, value))
		delete(tierChanges, node)
		delete(absentSince, node)
	}
	return errors.Join(errs...)
}

// releaseExpiredWorkloadNodes runs the release windows for labeled nodes the
// workload has left: the window starts at the first reconcile that observes
// the absence — never at a historical last-seen time, so a controller outage
// crossing the actual departure still grants the full delay, and untracked
// labels (operator predating the bookkeeping) get a full window too. Returns
// the earliest pending expiry for the caller's requeue.
func releaseExpiredWorkloadNodes(
	ctx context.Context,
	c client.Client,
	labeled map[string]*corev1.Node,
	desiredSet map[string]struct{},
	absentSince map[string]metav1.Time,
	releaseDelay time.Duration,
	now time.Time,
	key string,
) time.Time {
	// Removal failures are logged, not returned: a node that must lose its
	// label but cannot is the sticky-engine side of the asymmetry — the
	// safe side — and returning the error would block DaemonSet, Service,
	// ConfigMap and status reconciliation for the whole engine on a stuck
	// node patch. The scheduled requeue retries failed removals while the
	// label (and its engine pod) simply stays.
	var nextExpiry time.Time
	var removeRetry time.Time
	for nodeName, node := range labeled {
		if _, ok := desiredSet[nodeName]; ok {
			continue
		}
		since, tracked := absentSince[nodeName]
		if !tracked {
			since = metav1.Time{Time: now}
			absentSince[nodeName] = since
		}
		deadline := since.Add(releaseDelay)
		if now.Before(deadline) {
			if nextExpiry.IsZero() || deadline.Before(nextExpiry) {
				nextExpiry = deadline
			}
			continue // still inside the release window
		}
		if err := removeWorkloadLabel(ctx, c, node, key); err != nil {
			if apierrors.IsNotFound(err) {
				delete(absentSince, nodeName) // node object is gone; nothing to remove
				continue
			}
			log.FromContext(ctx).Error(err, "failed to remove workload label; retrying next reconcile", "node", nodeName)
			if removeRetry.IsZero() {
				removeRetry = now.Add(removalRetryDelay)
			}
			continue
		}
		delete(absentSince, nodeName)
	}
	if nextExpiry.IsZero() || (!removeRetry.IsZero() && removeRetry.Before(nextExpiry)) {
		return removeRetry
	}
	return nextExpiry
}

// pruneStaleWorkloadAbsence drops bookkeeping for nodes that are neither
// labeled nor desired (the label was stripped by other means).
func pruneStaleWorkloadAbsence(
	absentSince map[string]metav1.Time,
	labeled map[string]*corev1.Node,
	desiredSet map[string]struct{},
) {
	for nodeName := range absentSince {
		_, isLabeled := labeled[nodeName]
		_, isDesired := desiredSet[nodeName]
		if !isLabeled && !isDesired {
			delete(absentSince, nodeName)
		}
	}
}

// updateWorkloadPlacementStatus persists the kept bookkeeping and the visible
// node set. Empty bookkeeping maps are written as nil to avoid chronic churn.
func updateWorkloadPlacementStatus(
	ctx context.Context,
	c client.Client,
	engine *lmcachev1alpha1.LMCacheEngine,
	key string,
	desiredSet map[string]struct{},
	absentSince map[string]metav1.Time,
	tierChanges map[string]lmcachev1alpha1.NodeTierChange,
) error {
	changed := false
	if !timesEqual(absentSince, engine.Status.NodeAbsentSince) {
		if len(absentSince) > 0 {
			engine.Status.NodeAbsentSince = absentSince
		} else {
			engine.Status.NodeAbsentSince = nil
		}
		changed = true
	}
	if !tierChangesEqual(tierChanges, engine.Status.NodeTierChanges) {
		if len(tierChanges) > 0 {
			engine.Status.NodeTierChanges = tierChanges
		} else {
			engine.Status.NodeTierChanges = nil
		}
		changed = true
	}

	visible := make([]string, 0, len(desiredSet)+len(absentSince))
	for n := range desiredSet {
		visible = append(visible, n)
	}
	for n := range absentSince {
		visible = append(visible, n)
	}
	sort.Strings(visible)
	if !stringsEqual(visible, engine.Status.WorkloadNodes) {
		engine.Status.WorkloadNodes = visible
		changed = true
	}

	if engine.Status.WorkloadLabelKey != key {
		engine.Status.WorkloadLabelKey = key
		changed = true
	}

	if !changed {
		return nil
	}
	return c.Status().Update(ctx, engine)
}

// stripWorkloadPlacementLabels removes every node label managed by this
// engine. Called from the CR finalizer: node labels are cluster-scoped, so
// ownership garbage collection cannot clean them.
func stripWorkloadPlacementLabels(ctx context.Context, c client.Client, engine *lmcachev1alpha1.LMCacheEngine) error {
	key := resources.WorkloadPlacementLabel(engine.Namespace, engine.Name)
	return purgeWorkloadLabels(ctx, c, key)
}

func purgeWorkloadLabels(ctx context.Context, c client.Client, key string) error {
	nodes, err := listLabeledNodes(ctx, c, key)
	if err != nil {
		return err
	}
	for _, node := range nodes {
		if err := removeWorkloadLabel(ctx, c, node, key); err != nil {
			return err
		}
	}
	return nil
}

func addWorkloadLabel(ctx context.Context, c client.Client, node *corev1.Node, key, value string) error {
	if node.Labels[key] == value {
		return nil
	}
	patch := client.MergeFrom(node.DeepCopy())
	if node.Labels == nil {
		node.Labels = map[string]string{}
	}
	node.Labels[key] = value
	return c.Patch(ctx, node, patch)
}

func removeWorkloadLabel(ctx context.Context, c client.Client, node *corev1.Node, key string) error {
	if _, ok := node.Labels[key]; !ok {
		return nil
	}
	patch := client.MergeFrom(node.DeepCopy())
	delete(node.Labels, key)
	return c.Patch(ctx, node, patch)
}

// stringsEqual compares two ordered string slices; used to guard status
// writes against no-op churn.
func stringsEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func timesEqual(a, b map[string]metav1.Time) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		ob, ok := b[k]
		if !ok || !v.Equal(&ob) {
			return false
		}
	}
	return true
}

// isNumericLabel reports whether a label value is a decimal number —
// i.e. a tier value this operator wrote. Anything else (notably the
// "true" of pre-DRA intermediate builds) is residue to relabel
// immediately rather than gate behind the tier-change hysteresis.
func isNumericLabel(v string) bool {
	if v == "" {
		return false
	}
	for _, r := range v {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// tierChangesEqual compares two pending-tier bookkeeping maps; guards
// status writes against no-op churn (mirrors timesEqual).
func tierChangesEqual(a, b map[string]lmcachev1alpha1.NodeTierChange) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		ob, ok := b[k]
		if !ok || v.Tier != ob.Tier || !v.Since.Equal(&ob.Since) {
			return false
		}
	}
	return true
}
