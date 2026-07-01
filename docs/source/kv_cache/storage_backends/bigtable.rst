Google Cloud Bigtable
=====================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


.. _bigtable-overview:

Overview
--------

Google Cloud Bigtable is a petabyte-scale, fully managed NoSQL database. Integrating Cloud Bigtable as a built-in remote storage connector inside LMCache bridges volatile high-cost in-memory tiers (Redis) and low-cost, high-latency archival object stores (S3). 

For more information, see the `Cloud Bigtable Overview <https://cloud.google.com/bigtable/docs/overview>`_ and `Cloud Bigtable Pricing <https://cloud.google.com/bigtable/pricing>`_.

Architecture & Payload Limits
-----------------------------

- **Chunk Size Optimization**: Set LMCache's logical ``chunk_size`` to **256 tokens**. This groups payloads to minimize sequential Point-Read gRPC calls, preventing Python event-loop (GIL) bottlenecks.
- **MutateRow Limit**: Enforces a strict **90.0 MB request limit** for a single ``MutateRow`` gRPC request.
- **Storage Tier Row Limits**: The **SSD Tier** ceiling is **100 MiB per cell/row**. The **Enterprise Plus In-Memory Tier** is limited to **1.0 MiB per row**.
- **TTLCache Shielding**: Embeds a thread-safe ``TTLCache`` (10-second TTL default) to shield Bigtable nodes from concurrent prefetch lookup spikes.

Infrastructure Setup
--------------------

**1. Enable GCP APIs**

.. code-block:: bash

gcloud services enable bigtable.googleapis.com bigtableadmin.googleapis.com --project=your-gcp-project-id

**2. Provision Bigtable Instance**

Refer to the `gcloud beta bigtable Reference <https://cloud.google.com/sdk/gcloud/reference/beta/bigtable>`_ for additional parameter details.

.. code-block:: bash

gcloud beta bigtable instances create your-bigtable-instance-id \
    --display-name="LMCache SSD Instance" \
    --edition=ENTERPRISE \
    --cluster-storage-type=ssd \
    --cluster-config=id=your-cluster-id,zone=us-central1-a,nodes=1 \
    --project=your-gcp-project-id

**3. Create Database Table & Column Family**

.. code-block:: bash

gcloud bigtable instances tables create lmcache-benchmark-v1 \
    --instance=your-bigtable-instance-id \
    --column-families=cf \
    --project=your-gcp-project-id

**4. Install LMCache & Bigtable SDK**

.. code-block:: bash

export NO_NATIVE_EXT=1
pip install --no-cache-dir lmcache google-cloud-bigtable

Configuration
-------------

**Example A: Standard Bigtable SSD Integration (L2 Only)**

.. code-block:: yaml

chunk_size: 256

local_cpu: true
max_local_cpu_size: 10.0
remote_url: "bigtable://your-gcp-project-id/your-bigtable-instance-id"

remote_serde: "naive"

extra_config:
  bigtable_project_id: "your-gcp-project-id"
  bigtable_instance_id: "your-bigtable-instance-id"
  bigtable_table_name: "lmcache-benchmark-v1"

.. note::
   Alternatively, you can set the environment variables ``BT_PROJECT_ID``, ``BT_INSTANCE_ID``, and ``BT_TABLE_NAME`` instead of using ``extra_config``.

**Example B: 3-Tier Multi-Connector Hybrid (Local CPU -> Redis L2 -> Bigtable SSD L3)**

Deploy Redis for hot-cache loopbacks while offloading long-tail persistent storage to Bigtable SSD, using LMCache's dynamic OrderedDict routing.

.. code-block:: yaml

chunk_size: 256
local_cpu: true
max_local_cpu_size: 15.0

remote_storage_plugins:
  - "redis"
  - "bigtable"

extra_config:
  remote_storage_plugin.redis.redis_url: "redis://your-redis-host:6379"
  
  remote_storage_plugin.bigtable.bigtable_project_id: "your-gcp-project-id"
  remote_storage_plugin.bigtable.bigtable_instance_id: "your-bigtable-instance-id"
  remote_storage_plugin.bigtable.bigtable_table_name: "lmcache-benchmark-v1"
  remote_storage_plugin.bigtable.bigtable_family_name: "cf"
  remote_storage_plugin.bigtable.bigtable_column_name: "data"
  
  remote_storage_plugin.bigtable.credentials_path: "/etc/gcp/key.json"
  
  remote_storage_plugin.bigtable.bigtable_max_chunk_size_mb: 90.0
  remote_storage_plugin.bigtable.exists_cache_ttl_seconds: 10.0
  remote_storage_plugin.bigtable.exists_cache_size: 10000
  
  remote_storage_plugin.bigtable.bigtable_write_timeout_ms: 10000.0
  remote_storage_plugin.bigtable.bigtable_read_timeout_ms: 5000.0

Authentication
--------------

- **Application Default Credentials (ADC)**: If ``credentials_path`` is omitted or ``null``, the connector natively invokes ADC. Compatible with local development via ``gcloud auth application-default login`` or GKE Workload Identity Federation.
- **Explicit Keys**: Pass the absolute filesystem path containing a mounted GCP Service Account JSON secret to ``credentials_path``.

Verification
------------

Ensure you have installed the required dependencies:

.. code-block:: bash

pip install cachetools google-cloud-bigtable

Run the unit tests:

.. code-block:: bash

pytest tests/v1/storage_backend/test_bigtable_connector.py

Troubleshooting Large Payload Warnings
--------------------------------------

If you see a warning in the logs indicating that a chunk size exceeds the limit and is skipped (e.g. ``Bigtable chunk size ... MB exceeds threshold ... MB. Skipping write to prevent hard failures``), choose one of the following approaches:

1. **Reduce the LMCache Chunk Size (Recommended)**:
   The serialized chunk size depends on LMCache's logical ``chunk_size`` (number of tokens per chunk) and model shape. You can reduce ``chunk_size`` (e.g., from ``256`` to ``128``) in your configuration file to shrink individual chunk payloads.
   
2. **Increase the Max Chunk Size**:
   If your Bigtable instance uses the SSD storage tier (which supports up to 100 MB per cell/row), you can raise the maximum allowed write threshold in the configuration up to ``99.0`` MB using the ``bigtable_max_chunk_size_mb`` config key (or the ``BT_MAX_CHUNK_SIZE_MB`` environment variable).
   
   .. warning::
      Do not set ``bigtable_max_chunk_size_mb`` higher than ``100.0`` MB. While Cloud Bigtable supports up to ``256.0`` MB for a single row, a single cell value (which LMCache uses to store the chunk payload) has a hard limit of ``100.0`` MB. Exceeding this will trigger hard gRPC exceptions.
