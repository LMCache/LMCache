# llm-d + LMCache MP integration

Kubernetes manifests for vLLM model servers behind the
[llm-d](https://github.com/llm-d/llm-d) Router, with KV cache in an
**LMCache multiprocess (MP) server**. No change to llm-d is necessary.

The guide is
[llm-d Integration](https://docs.lmcache.ai/production/llm_d.html)
(`docs/source/production/llm_d.rst`).

## Layout

| Path | What it is |
|------|------------|
| [`kubernetes/lmcache_engine.yaml`](kubernetes/lmcache_engine.yaml) | LMCache server without the operator: DaemonSet, node-local Service and connection ConfigMap. Apply it first. |
| [`kubernetes/router_values.yaml`](kubernetes/router_values.yaml) | Values for the `llm-d-router-standalone` chart. |
| [`kubernetes/decode.yaml`](kubernetes/decode.yaml) | vLLM decode Deployment connected to the LMCache server. |
| [`kubernetes/pd_router_values.yaml`](kubernetes/pd_router_values.yaml) | Router values with the P/D plugins. |
| [`kubernetes/pd_pods.yaml`](kubernetes/pd_pods.yaml) | Prefill and decode Deployments for P/D disaggregation. The decode pod runs the llm-d routing sidecar. |

Install only one router release in the namespace: `router_values.yaml` or
`pd_router_values.yaml`. Each release makes a ConfigMap with the name `envoy`.

## Versions

The tests for the guide used one GPU node, llm-d `v1.0.0` (router chart and
P/D sidecar) and `lmcache/vllm-openai:v0.5.6`.
