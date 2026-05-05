Hugging Face Buckets Backend
============================

The Hugging Face Buckets backend stores LMCache chunks in a Hugging Face Bucket
using LMCache's built-in remote storage plugin framework. This is a persistent
remote backend that fits warm and cold KV cache persistence better than the
hottest local tiers.

When to use it
--------------

Use the HFBucket backend when you want:

* A Hub-native persistent store for KV cache data.
* A remote backend that can be configured through ``remote_storage_plugins``.
* Multiple named bucket instances in one LMCache deployment.

Avoid using it as the primary hot path for the lowest-latency cache lookups.
Local CPU, local disk, and other lower-latency backends are a better fit for
the hottest cache tier.


Requirements and limitations
----------------------------

* LMCache uses ``huggingface_hub`` bucket APIs for uploads, downloads, listing,
  and deletes.
* The first built-in release is intentionally conservative:

  * Only full chunks are supported.
  * Partial chunk uploads are rejected.
  * Downloads are rejected when the stored object size does not match the
    expected full LMCache chunk size.
  * Chunk metadata is not stored in the bucket objects.


Minimal configuration
---------------------

.. code-block:: yaml

   chunk_size: 256
   local_cpu: false
   save_unfull_chunk: false
   remote_serde: "naive"
   blocking_timeout_secs: 10
   remote_storage_plugins: ["hfbucket"]
   extra_config:
     remote_storage_plugin.hfbucket.bucket_handle: "hf://buckets/my-org/lmcache-kv/prod"
     remote_storage_plugin.hfbucket.token_env: "HF_TOKEN"
     remote_storage_plugin.hfbucket.create_bucket_if_missing: false
     remote_storage_plugin.hfbucket.download_tmp_dir: "/tmp/lmcache-hfbucket"
     remote_storage_plugin.hfbucket.metadata_cache_ttl_secs: 30


Multiple instances
------------------

Use instance-qualified plugin names to configure more than one bucket-backed
remote store in the same LMCache config.

.. code-block:: yaml

   remote_storage_plugins: ["hfbucket.us", "hfbucket.eu"]
   extra_config:
     remote_storage_plugin.hfbucket.us.bucket_handle: "hf://buckets/my-org/lmcache-kv/us"
     remote_storage_plugin.hfbucket.us.token_env: "HF_US_TOKEN"
     remote_storage_plugin.hfbucket.eu.bucket_handle: "hf://buckets/my-org/lmcache-kv/eu"
     remote_storage_plugin.hfbucket.eu.token_env: "HF_EU_TOKEN"


Configuration reference
-----------------------

All configuration keys live under
``extra_config.remote_storage_plugin.<plugin_name>.*`` where ``plugin_name`` is
either ``hfbucket`` or an instance-qualified name such as ``hfbucket.prod``.

* ``bucket_handle`` (required): Hugging Face Bucket handle in
  ``hf://buckets/<namespace>/<bucket>[/<prefix>]`` format.
* ``token_env`` (optional, default ``HF_TOKEN``): Environment variable used to
  resolve the Hugging Face access token.
* ``token`` (optional): Direct token override. ``token_env`` takes precedence
  when both are set.
* ``create_bucket_if_missing`` (optional, default ``false``): Lazily create the
  bucket on the first write path.
* ``download_tmp_dir`` (optional): Root directory for connector-local download
  scratch space. On Linux, pointing this at a tmpfs mount such as
  ``/dev/shm/lmcache-hfbucket`` avoids the disk write on the download path.
* ``metadata_cache_ttl_secs`` (optional, default ``30``): TTL for cached exact
  existence and size metadata.


Notes
-----

* The backend stores objects under the configured bucket prefix using a
  reversible encoding of LMCache keys, so ``list()`` returns LMCache key strings
  instead of raw bucket object paths.
