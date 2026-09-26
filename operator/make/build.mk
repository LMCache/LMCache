##@ Build

.PHONY: build
build: manifests generate fmt vet ## Build manager binary.
	go build -o bin/manager cmd/main.go

.PHONY: run
run: manifests generate fmt vet ## Run a controller from your host (webhook off; no host certs needed).
	ENABLE_WEBHOOKS=false go run ./cmd/main.go

# If you wish to build the manager image targeting other platforms you can use the --platform flag.
# (i.e. docker build --platform linux/arm64). However, you must enable docker buildKit for it.
# More info: https://docs.docker.com/develop/develop-images/build_enhancements/
.PHONY: docker-build
docker-build: ## Build docker image with the manager.
	$(CONTAINER_TOOL) build -t ${IMG} .

.PHONY: docker-push
docker-push: ## Push docker image with the manager.
	$(CONTAINER_TOOL) push ${IMG}

# PLATFORMS defines the target platforms for the manager image be built to provide support to multiple
# architectures. (i.e. make docker-buildx IMG=myregistry/mypoperator:0.0.1). To use this option you need to:
# - be able to use docker buildx. More info: https://docs.docker.com/build/buildx/
# - have enabled BuildKit. More info: https://docs.docker.com/develop/develop-images/build_enhancements/
# - be able to push the image to your registry (i.e. if you do not set a valid value via IMG=<myregistry/image:<tag>> then the export will fail)
# To adequately provide solutions that are compatible with multiple platforms, you should consider using this option.
PLATFORMS ?= linux/amd64
.PHONY: docker-buildx
docker-buildx: ## Build and push docker image for the manager for cross-platform support
	# copy existing Dockerfile and insert --platform=${BUILDPLATFORM} into Dockerfile.cross, and preserve the original Dockerfile
	sed -e '1 s/\(^FROM\)/FROM --platform=\$$\{BUILDPLATFORM\}/; t' -e ' 1,// s//FROM --platform=\$$\{BUILDPLATFORM\}/' Dockerfile > Dockerfile.cross
	- $(CONTAINER_TOOL) buildx create --name operator-builder
	$(CONTAINER_TOOL) buildx use operator-builder
	- $(CONTAINER_TOOL) buildx build --push --platform=$(PLATFORMS) --tag ${IMG} -f Dockerfile.cross .
	- $(CONTAINER_TOOL) buildx rm operator-builder
	rm Dockerfile.cross

.PHONY: build-installer
build-installer: manifests helm ## Render the chart as a standalone YAML installer.
	mkdir -p dist
	@$(helm-image-args) \
	printf 'apiVersion: v1\nkind: Namespace\nmetadata:\n  name: %s\n' "$(NAMESPACE)" > dist/install.yaml; \
	"$(HELM)" template "$(RELEASE)" "$(CHART)" --namespace "$(NAMESPACE)" \
		--set-string "image.repository=$${image%:*}" --set-string "image.tag=$${image##*:}" $(HELM_EXTRA_ARGS) >> dist/install.yaml

VERSION ?= v0.5.5
CHART_VERSION = $(shell printf '%s' '$(VERSION)' | sed -E \
	-e 's/^v//' \
	-e 's/^([0-9]+\.[0-9]+\.[0-9]+)(alpha|beta|rc)([0-9]+)$$/\1-\2.\3/' \
	-e 's/^nightly-([0-9]{4})-([0-9]{2})-([0-9]{2})$$/0.0.0-nightly.\1\2\3/')

.PHONY: print-chart-version
print-chart-version:
	@printf '%s\n' "$(CHART_VERSION)"

.PHONY: package-chart
package-chart: manifests helm ## Package the chart using the Operator VERSION.
	"$(HELM)" package "$(CHART)" --destination dist --version "$(CHART_VERSION)" --app-version "$(VERSION)"

.PHONY: lint-chart
lint-chart: manifests helm ## Lint the Helm chart.
	"$(HELM)" lint "$(CHART)" --strict
