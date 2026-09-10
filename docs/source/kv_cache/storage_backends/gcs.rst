Google Cloud Storage Backend
============================

The Google Cloud Storage (GCS) backend stores LMCache chunks in a GCS bucket
using LMCache's built-in remote storage plugin framework. This is a persistent
remote backend suited to warm and cold KV cache persistence rather than the
lowest-latency hot tier.

When to use it
--------------

Use the GCS backend when you want:

* A managed object store for persistent KV cache data.
* A built-in backend configured through ``remote_storage_plugins``.
* Multiple named GCS-backed instances in one LMCache deployment.

Avoid using it as the primary hot path for the lowest-latency cache lookups.
Local CPU, local disk, and other lower-latency backends are a better fit for
the hottest cache tier.


Requirements and limitations
----------------------------

* LMCache uses the ``google-cloud-storage`` Python client for uploads,
  downloads, listing, and deletes.
* The phase-1 built-in release is intentionally conservative:

  * Only full chunks are supported.
  * Partial chunk uploads are rejected.
  * Downloads are rejected when the stored object size does not match the
    expected full LMCache chunk size.
  * Chunk metadata is not stored alongside the GCS objects.


Minimal configuration
---------------------

.. code-block:: yaml

   chunk_size: 256
   local_cpu: false
   save_unfull_chunk: false
   remote_serde: "naive"
   blocking_timeout_secs: 10
   remote_storage_plugins: ["gcs"]
   extra_config:
     remote_storage_plugin.gcs.bucket_uri: "gs://my-lmcache-bucket/prod"
     remote_storage_plugin.gcs.project: "my-gcp-project"
     remote_storage_plugin.gcs.credentials_path: "/etc/gcp/service-account.json"
     remote_storage_plugin.gcs.metadata_cache_ttl_secs: 30


Multiple instances
------------------

Use instance-qualified plugin names to configure more than one bucket-backed
remote store in the same LMCache config.

.. code-block:: yaml

   remote_storage_plugins: ["gcs.us", "gcs.eu"]
   extra_config:
     remote_storage_plugin.gcs.us.bucket_uri: "gs://lmcache-us/prod"
     remote_storage_plugin.gcs.us.project: "project-us"
     remote_storage_plugin.gcs.eu.bucket_uri: "gs://lmcache-eu/prod"
     remote_storage_plugin.gcs.eu.project: "project-eu"


Configuration reference
-----------------------

All configuration keys live under
``extra_config.remote_storage_plugin.<plugin_name>.*`` where ``plugin_name`` is
either ``gcs`` or an instance-qualified name such as ``gcs.prod``.

* ``bucket_uri`` (required): GCS bucket URI in ``gs://<bucket>[/<prefix>]``
  format.
* ``project`` (optional): GCP project override passed to the storage client.
  Leave unset to rely on Application Default Credentials project resolution.
* ``credentials_path`` (optional): Service-account JSON path passed to the
  storage client. Leave unset to use Application Default Credentials.
* ``metadata_cache_ttl_secs`` (optional, default ``30``): TTL for cached exact
  existence and size metadata.


Authentication
--------------

The connector works with either:

* **Application Default Credentials (recommended):** omit
  ``credentials_path`` and let the GCS client resolve credentials from the
  runtime environment.
* **Explicit service-account JSON:** set ``credentials_path`` when the LMCache
  process must use a specific credential file.


Notes
-----

* The backend stores objects under the configured GCS prefix using a reversible
  encoding of LMCache keys, so ``list()`` returns LMCache key strings instead
  of raw object names.
* This built-in connector is non-MP only. MP GCS L2 adapter work is out of
  scope for this release.
