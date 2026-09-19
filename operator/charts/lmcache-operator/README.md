# LMCache Operator

Requires Helm 3.8+ and cert-manager installed and ready.

```sh
helm upgrade --install lmcache-operator \
  oci://registry-1.docker.io/lmcache/lmcache-operator-chart \
  --version "<chart-version>" \
  --namespace lmcache-operator-system --create-namespace --wait
```

Use `-f operator-values.yaml` to customize the settings in `values.yaml`.
The operator watches all namespaces; install only one release per cluster.

For development and publishing, see the [operator development guide](../../README.md#development).
