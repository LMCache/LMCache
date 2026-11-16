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
	"maps"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	k8sptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

const finalizerName = "lmcache.ai/cleanup"

// dsSelectorSettleDelay backs the retry when the disabled-path cleanup finds
// the live DaemonSet still referencing the managed label: short enough to
// converge on an in-flight rolling update, long enough not to spin on
// scheduled-but-not-yet-observed state.
const dsSelectorSettleDelay = 5 * time.Second

// workloadPlacementFinalizer guards the cluster-scoped node labels a
// workload-aware engine manages; ownership garbage collection cannot clean
// node labels, so the finalizer runs placement.StripAll before the CR goes.
const workloadPlacementFinalizer = "lmcache.ai/workload-placement-cleanup"

// LMCacheEngineReconciler reconciles a LMCacheEngine object.
type LMCacheEngineReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=lmcacheengines,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=lmcacheengines/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=lmcacheengines/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=daemonsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch;patch
// +kubebuilder:rbac:groups=monitoring.coreos.com,resources=servicemonitors,verbs=get;list;watch;create;update;patch;delete

// Reconcile reconciles the LMCacheEngine CR.
func (r *LMCacheEngineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	// 1. Fetch CR
	engine := &lmcachev1alpha1.LMCacheEngine{}
	if err := r.Get(ctx, req.NamespacedName, engine); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// 2. Handle finalizer
	if err, done := r.handleFinalizer(ctx, engine); done {
		return ctrl.Result{}, err
	}

	// 3. Set defaults
	engine.SetDefaults()

	// 4. Validate
	if err := r.validateAndSetCondition(ctx, engine); err != nil {
		return ctrl.Result{}, err
	}

	// 5. Reconcile RESP auth secret (cross-namespace copy if needed)
	if err := r.reconcileRESPAuthSecret(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile RESP auth secret")
		return ctrl.Result{}, err
	}

	// 6. Workload-aware scheduling: mirror the managed node label onto the
	//    nodes hosting matching workload pods and schedule the release of
	//    nodes the workload has left. The ENABLED path runs before the
	//    DaemonSet step on purpose: labels must exist before the DaemonSet
	//    selector that requires them. The DISABLED path purges after the
	//    DaemonSet accepts the label-free selector (see below) so the
	//    DaemonSet controller never sees its selector reference a label
	//    that no node carries — purging first would delete every engine
	//    pod and drop every warm KV cache while the old selector lives.
	var requeueAfter time.Duration
	cleanUpWorkloadPlacement := false
	var effectiveTiers map[int32]struct{}
	if engine.WorkloadSchedulingEnabled() {
		if !controllerutil.ContainsFinalizer(engine, workloadPlacementFinalizer) {
			// Metadata-only patch: the in-memory engine has already been
			// through SetDefaults() (step 3) and may hold defaulted fields
			// (gpuVendor, logLevel, the injected nodeSelector) the user
			// never wrote — a full Update would persist them into the
			// stored spec, bumping the generation and fighting GitOps.
			if err := r.patchFinalizer(ctx, engine, workloadPlacementFinalizer, true); err != nil {
				return ctrl.Result{}, err
			}
		}
		delay, tiers, err := reconcileWorkloadPlacement(ctx, r.Client, engine, time.Now())
		if err != nil {
			log.Error(err, "Failed to reconcile workload-aware placement labels")
			return ctrl.Result{}, err
		}
		requeueAfter = delay
		effectiveTiers = tiers
	} else if controllerutil.ContainsFinalizer(engine, workloadPlacementFinalizer) {
		// Disabled-after-enabled: nothing happens here yet — only the DS
		// step below drops the label key from the live selector first.
		// Removing the finalizer requires the placement remain enabled (see
		// the WorkloadSchedulingEnabled check inside the cleanup), so a
		// working set cannot be rebuilt "stale" if re-enabled mid-delete.
		cleanUpWorkloadPlacement = true
	}

	// 7. Reconcile DaemonSet
	if err := r.reconcileDaemonSet(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile DaemonSet")
		return ctrl.Result{}, err
	}

	// 7.2 Reconcile tier DaemonSets (dynamic resource allocation): create
	//     or patch one DS per active tier N>=2, delete owned tier DSs
	//     whose tier is no longer active. The tier set comes from the
	//     placement pass above (the label values it just wrote), not from
	//     a re-read of node labels — the watch cache may not have caught
	//     up with our own patches, which would delay tier DS creation by
	//     a reconcile.
	if err := r.reconcileTierDaemonSets(ctx, engine, effectiveTiers); err != nil {
		log.Error(err, "Failed to reconcile tier DaemonSets")
		return ctrl.Result{}, err
	}

	// 7.5 Disabled-after-enabled cleanup: run only AFTER the DaemonSet step
	// above has accepted the label-free selector. purgeWorkloadPlacement
	// then re-reads the live DaemonSet and refuses to purge while its
	// nodeSelector still references the managed label, so no reconcile
	// order, watch race, or failed earlier step can trigger the
	// "selector points at a label nobody carries" evict-every-pod state.
	if cleanUpWorkloadPlacement {
		done, err := r.purgeWorkloadPlacement(ctx, engine)
		if err != nil {
			return ctrl.Result{}, err
		}
		if !done {
			// Live DS still references the key (update not landed yet);
			// retry shortly. Nothing has been purged.
			return ctrl.Result{RequeueAfter: dsSelectorSettleDelay}, nil
		}
	}

	// 8. Reconcile lookup Service (node-local discovery for vLLM)
	if err := r.reconcileLookupService(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile lookup Service")
		return ctrl.Result{}, err
	}

	// 9. Reconcile metrics Service
	if err := r.reconcileMetricsService(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile metrics Service")
		return ctrl.Result{}, err
	}

	// 10. Reconcile ConfigMap
	if err := r.reconcileConnectionConfigMap(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile ConfigMap")
		return ctrl.Result{}, err
	}

	// 11. Reconcile ServiceMonitor
	if err := r.reconcileServiceMonitor(ctx, engine); err != nil {
		log.Error(err, "Failed to reconcile ServiceMonitor")
		return ctrl.Result{}, err
	}

	// 12. Update status
	if err := r.updateStatus(ctx, engine); err != nil {
		log.Error(err, "Failed to update status")
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: requeueAfter}, nil
}

// purgeWorkloadPlacement runs the disabled-path cleanup after the DaemonSet
// step: strips residue labels, resets the status bookkeeping and removes the
// workload-placement finalizer. It returns done=false — purging nothing —
// while the live DaemonSet's nodeSelector still references the managed
// label: purging in that window makes the DaemonSet controller delete every
// engine pod (dropping every warm KV cache) until the next reconcile lands
// the label-free selector. The re-read is the enforceable form of the
// "labels are never stripped while a live selector references them"
// ordering contract; the full-Reconcile test asserts it.
func (r *LMCacheEngineReconciler) purgeWorkloadPlacement(ctx context.Context, engine *lmcachev1alpha1.LMCacheEngine) (bool, error) {
	if engine.WorkloadSchedulingEnabled() {
		// Re-enabled between the step-6 decision and here — nothing to do;
		// the finalizer must stay.
		return true, nil
	}
	key := resources.WorkloadPlacementLabel(engine.Namespace, engine.Name)

	// The live DaemonSet is authoritative: if it still selects on the key,
	// its update has not landed — do not purge. Tier DaemonSets are
	// deleted beforehand by reconcileTierDaemonSets and are deliberately
	// NOT read here: a tier DS still terminating cannot block the purge,
	// because the guard only inspects the base DS's nodeSelector.
	ds := &appsv1.DaemonSet{}
	err := r.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: engine.Namespace}, ds)
	switch {
	case err == nil && ds.Spec.Template.Spec.NodeSelector != nil && ds.Spec.Template.Spec.NodeSelector[key] != "":
		return false, nil // selector update not accepted yet; retry later
	case err != nil && !apierrors.IsNotFound(err):
		return false, err
	}

	// Selector is label-free (or no DaemonSet exists): safe to purge.
	if _, _, err := reconcileWorkloadPlacement(ctx, r.Client, engine, time.Now()); err != nil {
		return false, err
	}
	// reconcileWorkloadPlacement's off path does not know about the
	// finalizer; strip it via a metadata-only patch. A full Update here
	// would be worse than drift: reconcileDaemonSet earlier overwrote
	// engine.Spec.Coordinator with its RESOLVED value (ref AND url set),
	// and writing that back would permanently break coordinator-ref
	// engines ("exactly one of ref or url must be set" on every later
	// validation).
	if err := r.patchFinalizer(ctx, engine, workloadPlacementFinalizer, false); err != nil {
		return false, err
	}
	return true, nil
}

// enginesForPod maps a pod event to the workload-aware engines whose workload
// selector matches it. Pod events are the trigger for label refreshes: a
// workload arriving on or leaving a node must add or (after the release
// delay) drop that node's engine label. Informer selectors are static at
// manager start, so the pod watch is broad and matching happens here per
// event against each enabled engine's selector.
func (r *LMCacheEngineReconciler) enginesForPod(ctx context.Context, obj client.Object) []reconcile.Request {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return nil
	}
	// Scope the list to the pod's namespace: workload-aware engines only
	// ever match pods in their own namespace, so a cluster-wide copy on
	// every pod event would be pure waste (unrelated system pods included).
	engines := &lmcachev1alpha1.LMCacheEngineList{}
	if err := r.List(ctx, engines, client.InNamespace(pod.Namespace)); err != nil {
		return nil
	}
	var out []reconcile.Request
	for i := range engines.Items {
		engine := &engines.Items[i]
		if !engine.WorkloadSchedulingEnabled() {
			continue
		}
		if !workloadMatches(engine, pod.Labels) {
			continue
		}
		out = append(out, reconcile.Request{
			NamespacedName: types.NamespacedName{Namespace: engine.Namespace, Name: engine.Name},
		})
	}
	return out
}

// SetupWithManager sets up the controller with the Manager.
func (r *LMCacheEngineReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&lmcachev1alpha1.LMCacheEngine{}).
		Owns(&appsv1.DaemonSet{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.Service{}).
		Owns(&corev1.Secret{}).
		Watches(
			&corev1.Pod{},
			handler.EnqueueRequestsFromMapFunc(r.enginesForPod),
			builder.WithPredicates(workloadPlacementPodPredicate()),
		).
		Watches(
			&corev1.Node{},
			handler.EnqueueRequestsFromMapFunc(r.enginesForNode),
			builder.WithPredicates(workloadPlacementNodePredicate()),
		).
		Named("lmcacheengine").
		Complete(r)
}

// workloadPlacementPodPredicate filters pod updates to the fields placement
// reacts to (nodeName, deletionTimestamp, labels); ignoring status or
// condition churn, which dominates pod event volume in busy namespaces.
func workloadPlacementPodPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc:  func(ev event.CreateEvent) bool { return placementPod(ev.Object) },
		DeleteFunc:  func(_ event.DeleteEvent) bool { return true }, // leaving a node releases it
		GenericFunc: func(_ event.GenericEvent) bool { return false },
		UpdateFunc: func(ev event.UpdateEvent) bool {
			old, okOld := ev.ObjectOld.(*corev1.Pod)
			new, okNew := ev.ObjectNew.(*corev1.Pod)
			switch {
			case !okOld || !okNew:
				return false
			case old.Spec.NodeName != new.Spec.NodeName:
				return true
			case !k8sptr.Equal(old.DeletionTimestamp, new.DeletionTimestamp):
				return true
			case !maps.Equal(old.Labels, new.Labels):
				return true
			case old.Status.Phase != new.Status.Phase &&
				(new.Status.Phase == corev1.PodSucceeded || new.Status.Phase == corev1.PodFailed):
				// A pod reaching a terminal phase no longer counts for
				// placement or the tier multiplier, and nothing else in
				// the pod object changes on that transition — without
				// this case a finished Job pod (e.g. lingering on
				// ttlSecondsAfterFinished) would strand its node's
				// label and engine forever.
				return true
			default:
				return false
			}
		},
	}
}

func placementPod(obj client.Object) bool {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return false
	}
	return pod.Spec.NodeName != "" || len(pod.Labels) > 0
}

// workloadPlacementNodePredicate filters node events to managed-label
// CHANGES: if someone or something removed or retiered a `lmcache.ai/wlas-`
// label on a node that still hosts the workload, nothing repairs it until
// this watch enqueues the owning engines. Comparing the managed-label
// subset — not mere presence — also keeps routine kubelet status churn on
// labeled nodes from fanning out to every engine reconcile.
func workloadPlacementNodePredicate() predicate.Predicate {
	wlasLabels := func(ls map[string]string) map[string]string {
		out := map[string]string{}
		for k, v := range ls {
			if strings.HasPrefix(k, resources.WorkloadLabelPrefix) {
				out[k] = v
			}
		}
		return out
	}
	return predicate.Funcs{
		CreateFunc: func(ev event.CreateEvent) bool { return len(wlasLabels(ev.Object.GetLabels())) > 0 },
		DeleteFunc: func(ev event.DeleteEvent) bool { return true }, // a labeled node loss re-reconciles all engines
		UpdateFunc: func(ev event.UpdateEvent) bool {
			old, okOld := ev.ObjectOld.(*corev1.Node)
			new, okNew := ev.ObjectNew.(*corev1.Node)
			if !okOld || !okNew {
				return false
			}
			return !maps.Equal(wlasLabels(old.Labels), wlasLabels(new.Labels))
		},
		GenericFunc: func(_ event.GenericEvent) bool { return false },
	}
}

// enginesForNode maps a managed-label change on a node to every enabled
// engine in the cluster: node labels are cluster-scoped, so only the map
// function narrows via each engine's spec.
func (r *LMCacheEngineReconciler) enginesForNode(ctx context.Context, _ client.Object) []reconcile.Request {
	engines := &lmcachev1alpha1.LMCacheEngineList{}
	if err := r.List(ctx, engines); err != nil {
		return nil
	}
	var out []reconcile.Request
	for i := range engines.Items {
		if !engines.Items[i].WorkloadSchedulingEnabled() {
			continue
		}
		out = append(out, reconcile.Request{
			NamespacedName: types.NamespacedName{
				Namespace: engines.Items[i].Namespace,
				Name:      engines.Items[i].Name,
			},
		})
	}
	return out
}

// Ensure resources package is used (for linter).
var _ = resources.StandardLabels

// patchFinalizer adds or removes the workload-placement finalizer via a
// metadata-only patch against a freshly fetched copy of the CR. The
// in-memory engine passed through Reconcile may have been mutated since
// its fetch — SetDefaults() fills defaulted spec fields and
// reconcileDaemonSet resolves spec.coordinator (ref AND url) — and a
// full-object Update would persist those mutations into the stored spec,
// bumping the generation and, for coordinator.ref engines, permanently
// failing validation ("exactly one of ref or url must be set"). A
// strategic-merge patch scoped to metadata.finalizers writes nothing else.
func (r *LMCacheEngineReconciler) patchFinalizer(
	ctx context.Context,
	engine *lmcachev1alpha1.LMCacheEngine,
	finalizer string,
	add bool,
) error {
	fresh := &lmcachev1alpha1.LMCacheEngine{}
	if err := r.Get(ctx, types.NamespacedName{Name: engine.Name, Namespace: engine.Namespace}, fresh); err != nil {
		return err
	}
	patch := client.MergeFrom(fresh.DeepCopy())
	if add {
		controllerutil.AddFinalizer(fresh, finalizer)
	} else {
		controllerutil.RemoveFinalizer(fresh, finalizer)
	}
	// Keep the in-memory engine's finalizer state in sync so later steps
	// in this reconcile (e.g. the finalizer-removal check at step 6.5)
	// agree with the API server. resourceVersion staleness is fine:
	// updateStatus re-fetches the engine before its status write.
	if add {
		controllerutil.AddFinalizer(engine, finalizer)
	} else {
		controllerutil.RemoveFinalizer(engine, finalizer)
	}
	if err := r.Patch(ctx, fresh, patch); err != nil {
		return err
	}
	// Sync the in-memory engine so later full-object writes in THIS
	// reconcile (Status().Update from the placement step) carry the
	// post-patch resourceVersion instead of conflicting.
	engine.ResourceVersion = fresh.ResourceVersion
	return nil
}
