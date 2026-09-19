# LMCache Operator

LMCache Operator automates the deployment and lifecycle management of LMCache cache servers on Kubernetes.

```sh
helm upgrade --install lmcache-operator \
  oci://registry-1.docker.io/lmcache/lmcache-operator-chart \
  --version "<chart-version>" \
  --namespace lmcache-operator-system --create-namespace --wait
```
