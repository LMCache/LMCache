##@ Unit tests + coverage

# Runs envtest-driven unit tests across everything except the
# build-tagged e2e packages. Writes operator/cover.out so the
# downstream test-cover* targets have something to summarise.
.PHONY: test
test: manifests generate fmt vet setup-envtest ## Run unit tests via envtest. Writes cover.out.
	KUBEBUILDER_ASSETS="$(shell "$(ENVTEST)" use $(ENVTEST_K8S_VERSION) --bin-dir "$(LOCALBIN)" -p path)" \
		go test $$(go list ./... | grep -v /e2e) -coverprofile cover.out

# Strip auto-generated DeepCopy from the profile before reporting —
# 100% coverage on `zz_generated.deepcopy.go` is meaningless and
# either inflates or dilutes the headline depending on volume. We
# also drop test/utils (its consumers are e2e specs, not unit tests)
# and cmd/ (entrypoint wiring; integration-tested implicitly).
COVER_PROFILE     ?= cover.out
COVER_FILTERED    := $(COVER_PROFILE).filtered
COVER_FILTER_RE   ?= zz_generated|test/utils|cmd/

.PHONY: test-cover
test-cover: test ## Print a per-function coverage summary (filtered) + total.
	@grep -vE "$(COVER_FILTER_RE)" $(COVER_PROFILE) > $(COVER_FILTERED)
	@echo ""
	@go tool cover -func=$(COVER_FILTERED) | tail -20
	@echo ""
	@echo "==> Total (filtered):"
	@go tool cover -func=$(COVER_FILTERED) | grep "^total:"

.PHONY: test-cover-html
test-cover-html: test ## Generate an HTML coverage heatmap at cover.html.
	@grep -vE "$(COVER_FILTER_RE)" $(COVER_PROFILE) > $(COVER_FILTERED)
	go tool cover -html=$(COVER_FILTERED) -o cover.html
	@echo "==> Open cover.html in your browser."
