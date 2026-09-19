##@ Deployment

CHART := charts/lmcache-operator
RELEASE ?= lmcache-operator
NAMESPACE ?= lmcache-operator-system
HELM_EXTRA_ARGS ?=
HELM_TIMEOUT ?= 5m
ignore-not-found ?= false

# Pass the same explicitly tagged image through all rendering/deployment paths.
define helm-image-args
image="$(IMG)"; \
case "$${image}" in *@*) echo 'IMG must use repository:tag, not a digest' >&2; exit 1;; esac; \
case "$${image##*/}" in ?*:?*) ;; *) echo 'IMG must include an explicit :tag' >&2; exit 1;; esac;
endef

.PHONY: install
install: manifests ## Install CRDs only (Helm deploy already includes them).
	"$(KUBECTL)" apply -f config/crd/bases

.PHONY: uninstall
uninstall: ## Delete CRDs and ALL their custom resources; run only for full cleanup.
	"$(KUBECTL)" delete --ignore-not-found=$(ignore-not-found) -f config/crd/bases

.PHONY: deploy
deploy: manifests helm ## Install or upgrade the Operator with Helm; cert-manager must already be installed.
	@$(helm-image-args) \
	"$(HELM)" upgrade --install "$(RELEASE)" "$(CHART)" \
		--namespace "$(NAMESPACE)" --create-namespace --wait --timeout "$(HELM_TIMEOUT)" \
		--set-string "image.repository=$${image%:*}" --set-string "image.tag=$${image##*:}" $(HELM_EXTRA_ARGS)

.PHONY: undeploy
undeploy: helm ## Uninstall the Operator, retaining CRDs, instances and their workloads.
	"$(HELM)" uninstall "$(RELEASE)" --namespace "$(NAMESPACE)" --wait --timeout "$(HELM_TIMEOUT)" $(if $(filter true,$(ignore-not-found)),--ignore-not-found,)
