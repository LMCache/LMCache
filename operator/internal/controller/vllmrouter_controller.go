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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// VllmRouterReconciler reconciles a VllmRouter object.
type VllmRouterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=vllmrouters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=vllmrouters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=lmcache.lmcache.ai,resources=vllmrouters/finalizers,verbs=update

// Reconcile reconciles the VllmRouter CR. It converges a vllm-router
// Deployment and a frontend Service, then updates status from the Deployment.
func (r *VllmRouterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	// 1. Fetch CR
	router := &lmcachev1alpha1.VllmRouter{}
	if err := r.Get(ctx, req.NamespacedName, router); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// 2. Reconcile Deployment
	if err := r.reconcileRouterDeployment(ctx, router); err != nil {
		log.Error(err, "Failed to reconcile router Deployment")
		return ctrl.Result{}, err
	}

	// 3. Reconcile Service
	if err := r.reconcileRouterService(ctx, router); err != nil {
		log.Error(err, "Failed to reconcile router Service")
		return ctrl.Result{}, err
	}

	// 4. Update status
	if err := r.updateRouterStatus(ctx, router); err != nil {
		log.Error(err, "Failed to update router status")
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *VllmRouterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&lmcachev1alpha1.VllmRouter{}).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.Service{}).
		Named("vllmrouter").
		Complete(r)
}
