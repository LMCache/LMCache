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
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
	"github.com/LMCache/LMCache/internal/resources"
)

// reconcileRouterDeployment creates or updates the vllm-router Deployment.
func (r *VllmRouterReconciler) reconcileRouterDeployment(ctx context.Context, router *lmcachev1alpha1.VllmRouter) error {
	desired := resources.BuildRouterDeployment(router)

	existing := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(router, desired, r.Scheme); err != nil {
				return err
			}
			return r.Create(ctx, desired)
		}
		return err
	}

	// Preserve immutable selector.
	desired.Spec.Selector = existing.Spec.Selector
	desired.Spec.Template.Labels = resources.MergeLabels(
		existing.Spec.Selector.MatchLabels,
		desired.Spec.Template.Labels,
	)

	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec.Replicas = desired.Spec.Replicas
	existing.Spec.Template = desired.Spec.Template
	existing.Labels = desired.Labels

	if err := ctrl.SetControllerReference(router, existing, r.Scheme); err != nil {
		return err
	}

	return r.Patch(ctx, existing, patch)
}

// reconcileRouterService creates or updates the frontend Service.
func (r *VllmRouterReconciler) reconcileRouterService(ctx context.Context, router *lmcachev1alpha1.VllmRouter) error {
	desired := resources.BuildRouterService(router)

	existing := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(router, desired, r.Scheme); err != nil {
				return err
			}
			return r.Create(ctx, desired)
		}
		return err
	}

	patch := client.MergeFrom(existing.DeepCopy())
	existing.Spec.Ports = desired.Spec.Ports
	existing.Spec.Type = desired.Spec.Type
	existing.Labels = desired.Labels

	if err := ctrl.SetControllerReference(router, existing, r.Scheme); err != nil {
		return err
	}

	return r.Patch(ctx, existing, patch)
}

// updateRouterStatus re-fetches the Deployment and updates the CR status.
func (r *VllmRouterReconciler) updateRouterStatus(ctx context.Context, router *lmcachev1alpha1.VllmRouter) error {
	// Re-fetch to avoid resourceVersion conflicts from earlier reconcile steps.
	if err := r.Get(ctx, types.NamespacedName{Name: router.Name, Namespace: router.Namespace}, router); err != nil {
		return err
	}

	deploy := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: router.Name, Namespace: router.Namespace}, deploy)
	if err != nil {
		if apierrors.IsNotFound(err) {
			router.Status.Phase = lmcachev1alpha1.PhasePending
			router.Status.ReadyReplicas = 0
			router.Status.ObservedGeneration = router.Generation
			return r.Status().Update(ctx, router)
		}
		return err
	}

	router.Status.ReadyReplicas = deploy.Status.ReadyReplicas
	router.Status.ObservedGeneration = router.Generation

	desired := int32(1)
	if router.Spec.Replicas != nil {
		desired = *router.Spec.Replicas
	}

	switch {
	case deploy.Status.ReadyReplicas == desired:
		router.Status.Phase = lmcachev1alpha1.PhaseRunning
	case deploy.Status.ReadyReplicas > 0:
		router.Status.Phase = lmcachev1alpha1.PhaseDegraded
	default:
		router.Status.Phase = lmcachev1alpha1.PhasePending
	}

	available := deploy.Status.ReadyReplicas > 0
	meta.SetStatusCondition(&router.Status.Conditions, metav1.Condition{
		Type:               lmcachev1alpha1.ConditionAvailable,
		Status:             conditionBool(available),
		Reason:             reasonFromReady(available, "RouterAvailable", "RouterUnavailable"),
		Message:            fmt.Sprintf("%d/%d replicas ready", deploy.Status.ReadyReplicas, desired),
		ObservedGeneration: router.Generation,
	})

	return r.Status().Update(ctx, router)
}
