GCS
===

An L2 adapter that stores KV cache objects as Google Cloud Storage (GCS) blobs.
Suited for GCP deployments where the LMCache server runs in the same region as
the GCS bucket.

**Prerequisites:**

Install the GCS client library:

.. code-block:: bash

    pip install google-cloud-storage

**Required fields:**

- ``gcs_bucket`` (str): GCS bucket name (e.g. ``"my-lmcache-bucket"``).

**Optional fields:**

- ``gcs_credentials_file`` (str): Path to a service-account JSON key file.
  Omit to use `Application Default Credentials
  <https://cloud.google.com/docs/authentication/application-default-credentials>`_
  (``gcloud auth application-default login``, Workload Identity, etc.).
- ``gcs_project`` (str): GCP project ID. Required only when the credentials do
  not carry a default project (e.g. some service-account key files).
- ``gcs_num_workers`` (int, default ``64``): Thread-pool size for GCS I/O.
  Higher values increase concurrent upload/download throughput.
- ``max_capacity_gb`` (float, default ``0.0``): Aggregate capacity used by
  ``get_usage()``. A value of ``0`` disables watermark-triggered L2 eviction
  (``usage_fraction == -1.0``).
- ``max_connection_failures`` (int, default ``3``): Number of consecutive GCS
  connection errors before the circuit breaker disables the adapter.

**Configuration examples:**

.. code-block:: bash

    # Application Default Credentials (recommended on GCP VMs / GKE)
    --l2-adapter '{"type": "gcs", "gcs_bucket": "my-lmcache-bucket"}'

    # Explicit service-account key file
    --l2-adapter '{"type": "gcs", "gcs_bucket": "my-lmcache-bucket", "gcs_credentials_file": "/path/to/key.json"}'

    # With project ID and 128 worker threads
    --l2-adapter '{"type": "gcs", "gcs_bucket": "my-lmcache-bucket", "gcs_project": "my-gcp-project", "gcs_num_workers": 128}'

    # Capacity-capped (100 GB) with LRU eviction
    --l2-adapter '{
      "type": "gcs",
      "gcs_bucket": "my-lmcache-bucket",
      "max_capacity_gb": 100,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.9,
        "eviction_ratio": 0.1
      }
    }'

Authentication
--------------

The adapter supports two authentication paths:

1. **Application Default Credentials (ADC)** — recommended for GCP VMs, GKE
   pods (Workload Identity), and Cloud Run. No configuration needed; the SDK
   picks up credentials automatically.

   .. code-block:: bash

       gcloud auth application-default login   # for local development

2. **Service-account key file** — set ``gcs_credentials_file`` to the path of
   a downloaded JSON key. Required when running outside GCP without ADC.

Performance Notes
-----------------

GCS throughput is primarily determined by network proximity:

- **Same-region GCP VM → GCS bucket**: typical throughput 500–700 MB/s store,
  100–140 MB/s load (concurrent, batch ≥ 32 objects).
- **Cross-region or on-premises**: bandwidth is WAN-limited (~5–50 MB/s).

The adapter uses a ``ThreadPoolExecutor`` (``gcs_num_workers`` threads, default
64) to issue all blobs in a batch concurrently. Throughput scales with batch
size up to the network ceiling.

Benchmark (same-region GCP VM ``n2-standard-4``, ``us-central1-a``,
32 MB chunks — LLaMA-3 8B equivalent):

.. list-table::
   :header-rows: 1
   :widths: 22 12 14 14 12 14 14

   * - Config
     - Total
     - Store
     - Store MB/s
     - Lookup
     - Load
     - Load MB/s
   * - 32 MB × 32 chunks
     - 1 GB
     - 1,980 ms
     - 542
     - 2.1 ms
     - 7,825 ms
     - 137
   * - 32 MB × 64 chunks
     - 2 GB
     - 3,457 ms
     - 622
     - 2.1 ms
     - 22,920 ms
     - 94
   * - 32 MB × 96 chunks
     - 3 GB
     - 4,702 ms
     - 686
     - 2.2 ms
     - 34,432 ms
     - 94

Lookup latency is always ~2 ms regardless of batch size — served from an
in-memory size cache after the first store, with no GCS HEAD request.

Benchmarking
------------

The repository ships a standalone benchmark script at
``benchmarks/gcs_l2_adapter/gcs_l2_adapter_benchmark.py``.

**Install deps:**

.. code-block:: bash

    pip install google-cloud-storage opentelemetry-sdk opentelemetry-api \
                sortedcontainers prometheus_client

**Object-size matrix sweep:**

.. code-block:: bash

    python benchmarks/gcs_l2_adapter/gcs_l2_adapter_benchmark.py \
        --bucket my-lmcache-bucket \
        --object-sizes-kb 1024,4096,16384 \
        --batch-sizes 1,4,8,16

**1 GB / 2 GB / 3 GB KV-cache offload** (32 MB chunks, LLaMA-3 8B config):

.. code-block:: bash

    python benchmarks/gcs_l2_adapter/gcs_l2_adapter_benchmark.py \
        --bucket my-lmcache-bucket \
        --object-sizes-kb 32768 \
        --batch-sizes 32,64,96

**Token-aware mode** (auto-computes chunk size from model config):

.. code-block:: bash

    python benchmarks/gcs_l2_adapter/gcs_l2_adapter_benchmark.py \
        --bucket my-lmcache-bucket \
        --tokens 1000 \
        --model-layers 32 \
        --kv-heads 8 \
        --head-dim 128 \
        --chunk-tokens 256 \
        --dtype-bytes 2 \
        --batch-sizes 1,4,8

**With explicit credentials:**

.. code-block:: bash

    python benchmarks/gcs_l2_adapter/gcs_l2_adapter_benchmark.py \
        --bucket my-lmcache-bucket \
        --credentials /path/to/key.json \
        --object-sizes-kb 32768 \
        --batch-sizes 32,64,96
