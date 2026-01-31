# LMCache + vLLM: Enterprise Platform Deployment Guide

## 1. Introduction

**Target workload**
- Enterprise production deployments
- Integration with existing ML infrastructure
- Observability, monitoring, and routing requirements
- **Production-grade LLM serving with LMCache**

**LMCache mode**
- **Storage Mode or Transport Mode (PD)**
- Multi-node with shared cache
- Enterprise backends (Redis Cluster, tiered storage)

This recipe provides a **comprehensive enterprise deployment guide** combining:

1. **Production Stack** - Helm deployment with LMCache
2. **Observability** - Metrics, logging, and monitoring
3. **Routing** - Intelligent request routing
4. **High Availability** - Redis Sentinel, multi-zone deployment
5. **Security** - TLS, authentication, network policies

> **Prerequisites:** This recipe assumes familiarity with Kubernetes, Helm, and basic LMCache concepts (R-001, R-010, R-018).

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                         Enterprise Deployment                    │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Ingress   │    │   Ingress   │    │   Ingress   │        │
│  │  Controller │    │  Controller │    │  Controller │        │
│  │  (TLS/Auth) │    │  (TLS/Auth) │    │  (TLS/Auth) │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │               │
│         └──────────────────┼──────────────────┘               │
│                            │                                    │
│                   ┌────────▼────────┐                          │
│                   │  Load Balancer  │                          │
│                   │  (Session-aware)│                          │
│                   └────────┬────────┘                          │
│                            │                                    │
│  ┌─────────────────────────┼─────────────────────────────┐    │
│  │              Kubernetes Cluster                         │    │
│  │  ┌──────────────────────┼──────────────────────┐      │    │
│  │  │              vLLM Pods (HPA)                  │      │    │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │      │    │
│  │  │  │vLLM+   │ │vLLM+   │ │vLLM+   │ │vLLM+   │ │      │    │
│  │  │  │LMCache │ │LMCache │ │LMCache │ │LMCache │ │      │    │
│  │  │  │  +     │ │  +     │ │  +     │ │  +     │ │      │    │
│  │  │  │Sidecar │ │Sidecar │ │Sidecar │ │Sidecar │ │      │    │
│  │  │  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ │      │    │
│  │  │       │          │          │          │      │      │    │
│  │  └───────┼──────────┼──────────┼──────────┼──────┘      │    │
│  │          │          │          │          │              │    │
│  │          └──────────┴──────────┴──────────┘              │    │
│  │                     │                                    │    │
│  │         ┌───────────▼───────────┐                        │    │
│  │         │   Redis Cluster       │                        │    │
│  │         │   (HA + Sharding)     │                        │    │
│  │         └───────────────────────┘                        │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Observability Stack                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │   │
│  │  │Prometheus│  │ Grafana │  │  Loki   │  │  Jaeger │     │   │
│  │  │ (Metrics)│  │(Dashboard)│ │ (Logs)  │  │(Traces) │     │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 2. When to Use Enterprise Platform

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Production at scale | **Enterprise platform** (this recipe) | Full observability, HA |
| Development/testing | Standard recipes (R-001) | Simpler, faster setup |
| Single team | Standard recipes | Lower operational overhead |
| Multi-team enterprise | **Enterprise platform** | Shared infrastructure |
| Regulatory requirements | **Enterprise platform** | Audit logging, security |

## 3. Installing Prerequisites

### 3.1 Infrastructure
- Kubernetes cluster (1.24+)
- Helm 3.x
- kubectl configured
- Container registry access

### 3.2 Resources
- Redis Cluster (3+ nodes)
- Prometheus + Grafana (observability)
- Ingress controller (NGINX/Traefik)
- Cert-manager (TLS)

## 4. Configuration

## 5. Launching the Enterprise Platform

### 5.1 Helm Deployment

### 4.1 Values file

Create `recipes/vllm_enterprise_values.yaml`:

```yaml
# vLLM Production Stack values
servingEngineSpec:
  modelSpec:
  - name: "llama-3-8b"
    repository: "vllm/vllm-openai"
    tag: "latest"
    modelURL: "meta-llama/Llama-3.1-8B-Instruct"
    
    # LMCache Configuration
    lmcacheConfig: |
      chunk_size: 256
      local_cpu: true
      max_local_cpu_size: 48
      remote_url: "redis-cluster://redis-cluster:6379"
      enable_blending: true
      use_layerwise: true
      save_unfull_chunk: true
    
    # Resources
    resources:
      requests:
        memory: "32Gi"
        cpu: "8"
        nvidia.com/gpu: 1
      limits:
        memory: "64Gi"
        cpu: "16"
        nvidia.com/gpu: 1
    
    # Horizontal Pod Autoscaler
    hpa:
      enabled: true
      minReplicas: 3
      maxReplicas: 10
      targetCPUUtilizationPercentage: 70
      targetMemoryUtilizationPercentage: 80
    
    # Pod Disruption Budget
    pdb:
      enabled: true
      minAvailable: 2

# Ingress Configuration
ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: "llm-api.company.com"
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: llm-api-tls
      hosts:
        - "llm-api.company.com"

# Monitoring
metrics:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: "monitoring"
    interval: 30s

# Logging
logging:
  enabled: true
  format: json
  level: INFO
```

### 4.2 Deploy

```bash
# Add Helm repo
helm repo add vllm https://vllm-project.github.io/production-stack
helm repo update

# Install with enterprise values
helm install vllm-enterprise vllm/vllm-stack \
  -f recipes/vllm_enterprise_values.yaml \
  --namespace llm-serving \
  --create-namespace
```

## 5. Redis Cluster Setup

### 5.1 Redis Cluster for HA

```yaml
# redis-cluster.yaml
apiVersion: databases.spotahome.com/v1
kind: RedisFailover
metadata:
  name: redis-cluster
  namespace: llm-serving
spec:
  sentinel:
    replicas: 3
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 500m
        memory: 256Mi
  redis:
    replicas: 3
    resources:
      requests:
        cpu: 500m
        memory: 4Gi
      limits:
        cpu: 2000m
        memory: 16Gi
    storage:
      persistentVolumeClaim:
        spec:
          resources:
            requests:
              storage: 100Gi
```

```bash
kubectl apply -f redis-cluster.yaml
```

## 6. Observability Setup

### 6.1 Prometheus ServiceMonitor

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-lmcache-metrics
  namespace: monitoring
  labels:
    release: prometheus
spec:
  namespaceSelector:
    matchNames:
    - llm-serving
  selector:
    matchLabels:
      app: vllm
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### 6.2 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "LMCache Enterprise",
    "panels": [
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(lmcache_cache_hits_total[5m]) / rate(lmcache_cache_requests_total[5m])"
          }
        ]
      },
      {
        "title": "TTFT by Tier",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(lmcache_ttft_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Storage Usage",
        "targets": [
          {
            "expr": "lmcache_storage_usage_bytes"
          }
        ]
      }
    ]
  }
}
```

### 6.3 Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `lmcache_cache_hit_rate` | Cache hit percentage | < 70% |
| `lmcache_ttft_p99` | P99 time to first token | > 500ms |
| `lmcache_storage_usage` | Storage utilization | > 80% |
| `lmcache_worker_queue_depth` | Async queue depth | > 100 |

## 7. Security Configuration

### 7.1 Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vllm-network-policy
  namespace: llm-serving
spec:
  podSelector:
    matchLabels:
      app: vllm
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis-cluster
    ports:
    - protocol: TCP
      port: 6379
```

### 7.2 Pod Security Context

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop:
    - ALL
```

## 6. Validation

## 7. Benchmarking

### 7.1 Load test

```bash
# Enterprise load test
vllm bench serve \
  --url https://llm-api.company.com \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 1000 \
  --max-concurrency 100 \
  --request-rate 50
```

### 7.2 Expected results

| Metric | Target |
|--------|--------|
| p50 TTFT | < 150ms |
| p99 TTFT | < 300ms |
| Cache hit rate | > 70% |
| Availability | > 99.9% |

## 8. Maintenance

### 8.1 Check deployment

```bash
# Check pods
kubectl get pods -n llm-serving

# Check HPA
kubectl get hpa -n llm-serving

# Check ingress
kubectl get ingress -n llm-serving

# Test endpoint
curl https://llm-api.company.com/v1/models \
  -H "Authorization: Bearer $API_TOKEN"
```

### 8.2 Load test

```bash
# Enterprise load test
vllm bench serve \
  --url https://llm-api.company.com \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 1000 \
  --max-concurrency 100 \
  --request-rate 50
```

## 9. Maintenance

### 9.1 Rolling updates

```bash
# Update with zero downtime
helm upgrade vllm-enterprise vllm/vllm-stack \
  -f recipes/vllm_enterprise_values.yaml \
  --namespace llm-serving \
  --set rollingUpdate.maxSurge=2 \
  --set rollingUpdate.maxUnavailable=0
```

### 9.2 Backup and restore

```bash
# Backup Redis
kubectl exec -it redis-cluster-0 -- redis-cli BGSAVE

# Backup disk cache
kubectl cp llm-serving/vllm-pod-0:/var/lib/lmcache ./lmcache-backup
```

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| High latency | Cache misses | Check cache hit rate |
| OOM errors | Memory limits | Increase resources |
| HPA not scaling | Metrics not available | Check ServiceMonitor |
| TLS errors | Cert not ready | Check cert-manager |
| Redis connection failed | Network policy | Allow egress to Redis |

## 11. Additional Resources
- Production Stack docs: https://github.com/vllm-project/production-stack
- Redis Operator: https://github.com/spotahome/redis-operator
- Prometheus Operator: https://github.com/prometheus-operator/prometheus-operator
