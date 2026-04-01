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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestConditionBool(t *testing.T) {
	if conditionBool(true) != metav1.ConditionTrue {
		t.Fatal("expected ConditionTrue")
	}
	if conditionBool(false) != metav1.ConditionFalse {
		t.Fatal("expected ConditionFalse")
	}
}

func TestReasonFromReady(t *testing.T) {
	if r := reasonFromReady(true, "Ready", "NotReady"); r != "Ready" {
		t.Fatalf("expected Ready, got %s", r)
	}
	if r := reasonFromReady(false, "Ready", "NotReady"); r != "NotReady" {
		t.Fatalf("expected NotReady, got %s", r)
	}
}

func TestFinalizerName(t *testing.T) {
	if finalizerName != "lmcache.ai/cleanup" {
		t.Fatalf("expected lmcache.ai/cleanup, got %s", finalizerName)
	}
}
