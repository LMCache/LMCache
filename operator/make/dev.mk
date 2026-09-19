##@ Development

.PHONY: manifests
manifests: controller-gen ## Generate WebhookConfiguration, ClusterRole and CustomResourceDefinition objects.
	"$(CONTROLLER_GEN)" rbac:roleName=manager-role crd:allowDangerousTypes=true webhook paths="./..." output:crd:artifacts:config=config/crd/bases
	@mkdir -p charts/lmcache-operator/files/crds
	@rm -f charts/lmcache-operator/files/crds/*.yaml
	@cp config/crd/bases/*.yaml charts/lmcache-operator/files/crds/
	@cp config/rbac/role.yaml charts/lmcache-operator/files/role.yaml
	@cp config/webhook/manifests.yaml charts/lmcache-operator/files/webhooks.yaml

.PHONY: generate
generate: controller-gen ## Generate code containing DeepCopy, DeepCopyInto, and DeepCopyObject method implementations.
	"$(CONTROLLER_GEN)" object:headerFile="hack/boilerplate.go.txt" paths="./..."

.PHONY: fmt
fmt: ## Run gofmt against code (covers build-tagged files that `go fmt ./...` skips).
	gofmt -w .

.PHONY: vet
vet: ## Run go vet against code.
	go vet ./...
