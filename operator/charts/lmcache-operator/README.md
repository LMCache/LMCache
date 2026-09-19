# LMCache Operator

This chart installs the LMCache Operator.

Install cert-manager and wait for it to be ready before installing this chart.
cert-manager is an external prerequisite, not a chart dependency.

Install the OCI chart with Helm 3.8 or newer, selecting a chart version from
the [Operator releases](https://github.com/LMCache/LMCache/releases):

```sh
helm upgrade --install lmcache-operator \
  oci://registry-1.docker.io/lmcache/lmcache-operator-chart \
  --version "<chart-version>" \
  --namespace lmcache-operator-system --create-namespace --wait
```

The same chart archive is attached to each Operator release. To install a
downloaded archive:

```sh
helm upgrade --install lmcache-operator "./lmcache-operator-chart-<chart-version>.tgz" \
  --namespace lmcache-operator-system --create-namespace --wait
```

Use `-f operator-values.yaml` to customize the settings in `values.yaml`.
The operator watches all namespaces; install only one release per cluster.
Use the same release name, namespace, and values file for subsequent upgrades.

Helm upgrades also update CRD schemas. Helm uninstall removes the operator and its
chart-managed infrastructure, but keeps CRDs, user-created instances, and their
workloads. The release does not own its namespace. Reinstall with the same
release name and namespace to resume management of retained instances.

```sh
helm uninstall lmcache-operator --namespace lmcache-operator-system
# From the source tree, the equivalent is: make helm-undeploy
```

Existing YAML installations require an explicit ownership transfer with Helm
3.17 or newer. See the [Operator installation guide](https://docs.lmcache.ai/mp/operator.html)
for migration and full cleanup instructions. Do not delete the YAML installer
resources when migrating, because that also deletes CRDs and the namespace.

For contributors: `make manifests` in `operator/` generates the CRD schemas,
controller RBAC, and webhook definitions directly from Go markers into `files/`.
Tests, CRD installation, and Helm all use these files. Do not edit them directly.
`make lint-chart` lints the chart; `make build-installer` renders the same templates
as YAML and adds the namespace to `dist/install.yaml`.

There are two deployment entry points using these templates:

- `make deploy IMG=...` renders the installer with Helm and applies it with
  `kubectl apply`; it does not create a Helm release. Developers need Helm for
  rendering, while users of the published `install.yaml` only need `kubectl`.
- `make helm-deploy IMG=...` runs `helm upgrade --install` against the chart.
  Remove this Helm release with `make helm-undeploy`, retaining the namespace,
  CRDs, instances, and their workloads.

**YAML cleanup is destructive:** `make undeploy` renders the full installer
again and deletes it with `kubectl delete`, including the namespace, all three
CRDs, their instances, and owned workloads. `helm.sh/resource-policy: keep`
only affects Helm; it does not prevent this deletion. Use the same namespace
and rendering settings as the YAML installation.

Release and nightly workflows publish OCI charts to Docker Hub repository
`lmcache/lmcache-operator-chart`, separately from the operator image repository
`lmcache/lmcache-operator`. They reuse the GitHub variable `DOCKERHUB_USERNAME`
and secret `DOCKERHUB_TOKEN`. Before publishing, create the chart repository,
grant that credential write access, and make the repository public for anonymous
Helm pulls. No additional publishing secret is required.
