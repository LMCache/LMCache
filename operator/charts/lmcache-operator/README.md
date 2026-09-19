# LMCache Operator

This chart installs one cluster-wide LMCache Operator, its three CRDs, RBAC,
admission webhooks, and webhook certificate resources. Create `LMCacheEngine`,
`CacheBlendEngine`, and `LMCacheCoordinator` instances separately.

Install cert-manager and wait for it to be ready before installing this chart.
cert-manager is an external prerequisite, not a chart dependency.

```sh
helm upgrade --install lmcache-operator "./lmcache-operator-<version>.tgz" \
  --namespace lmcache-operator-system --create-namespace --wait
```

Use `-f operator-values.yaml` to customize the settings in `values.yaml`.
The operator watches all namespaces; install only one release per cluster.
Use the same release name, namespace, and values file for subsequent upgrades.

Helm upgrades also update CRD schemas. Uninstall removes the operator and its
chart-managed infrastructure, but keeps CRDs, user-created instances, and their
workloads. The release does not own its namespace. Reinstall with the same
release name and namespace to resume management of retained instances.

```sh
helm uninstall lmcache-operator --namespace lmcache-operator-system
```

Existing YAML installations require an explicit ownership transfer with Helm
3.17 or newer. See the [Operator installation guide](https://docs.lmcache.ai/mp/operator.html)
for migration and full cleanup instructions. Do not delete the YAML installer
resources when migrating, because that also deletes CRDs and the namespace.

The optional `metrics.serviceMonitor.enabled` setting requires the Prometheus
Operator CRDs. Bind the Prometheus ServiceAccount to the
`lmcache-operator-metrics-reader` ClusterRole and configure Prometheus to discover
ServiceMonitors in the release namespace.

For contributors: `make manifests` in `operator/` regenerates the CRD schemas,
controller RBAC, and webhook definitions from Go markers and copies them into
`files/`. Do not edit these generated chart inputs directly. `make test-chart`
validates the chart; `make build-installer` renders the same templates as YAML.
