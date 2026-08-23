MinIO
=====

`MinIO \<https://min.io\>`_ is a high-performance, S3-compatible object store
often deployed on-premise or in self-hosted environments.  Because MinIO
implements the S3 API, LMCache's **S3 L2 adapter** works with MinIO
out of the box — no separate connector is needed.

This page walks through setting up a local MinIO instance and
configuring LMCache to use it as an L2 cache backend.


Prerequisites
-------------

- Docker (for running a local MinIO server)
- LMCache with AWS CRT dependencies installed (``pip install awscrt``)
- The ``mc`` (MinIO Client) CLI tool, *or* the MinIO web console, for
  creating buckets


Starting a Local MinIO Server
-----------------------------

.. code-block:: bash

    # Start MinIO with default credentials (minioadmin / minioadmin)
    docker run -d --name minio \
      -p 9000:9000 -p 9001:9001 \
      -e MINIO_ROOT_USER=minioadmin \
      -e MINIO_ROOT_PASSWORD=minioadmin \
      minio/minio server /data --console-address ":9001"

The S3 API is available at ``http://localhost:9000`` and the web console at
``http://localhost:9001``.


Creating a Bucket
-----------------

Using the ``mc`` CLI:

.. code-block:: bash

    mc alias set local http://localhost:9000 minioadmin minioadmin
    mc mb local/lmcache-kv

Or log in to the MinIO console at ``http://localhost:9001`` and create a
bucket named ``lmcache-kv`` through the UI.


LMCache Configuration (MP Mode)
--------------------------------

MinIO is configured via the S3 adapter (``"type": "s3"``) with two
MinIO-specific settings:

- ``disable_tls: true`` — MinIO typically serves plain HTTP.
- Static credentials via ``aws_access_key_id`` / ``aws_secret_access_key``
  — MinIO does not use the AWS default credential chain.

The ``s3_endpoint`` must use **virtual-hosted style**: include the bucket
name as part of the host (e.g. ``lmcache-kv.localhost:9000``).

.. code-block:: bash

    # Local MinIO, plain HTTP, explicit credentials
    --l2-adapter '{
      "type": "s3",
      "s3_endpoint": "lmcache-kv.localhost:9000",
      "s3_region": "us-east-1",
      "disable_tls": true,
      "aws_access_key_id": "minioadmin",
      "aws_secret_access_key": "minioadmin"
    }'

.. note::

   MinIO does not require a real AWS region, but the S3 adapter uses the
   region for request signing.  ``us-east-1`` is the conventional default.

**With capacity tracking and LMCache-side eviction:**

.. code-block:: bash

    --l2-adapter '{
      "type": "s3",
      "s3_endpoint": "lmcache-kv.localhost:9000",
      "s3_region": "us-east-1",
      "disable_tls": true,
      "aws_access_key_id": "minioadmin",
      "aws_secret_access_key": "minioadmin",
      "max_capacity_gb": 50,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.85,
        "eviction_ratio": 0.2
      }
    }'


Full vLLM Launch Example
-------------------------

.. code-block:: bash

    lmcache_server start \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --l2-adapter '{
        "type": "s3",
        "s3_endpoint": "lmcache-kv.localhost:9000",
        "s3_region": "us-east-1",
        "disable_tls": true,
        "aws_access_key_id": "minioadmin",
        "aws_secret_access_key": "minioadmin"
      }'


Environment Variables
---------------------

Credentials can also be provided via standard AWS environment variables
instead of embedding them in the adapter JSON:

.. code-block:: bash

    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin

When these are set, the ``aws_access_key_id`` and ``aws_secret_access_key``
fields can be omitted from the ``--l2-adapter`` JSON.  The adapter falls
back to ``boto3``'s credential resolution chain.


Configuration Reference
-----------------------

All fields are the same as the :doc:`S3 adapter <s3>`.  The table below
highlights the settings most relevant to MinIO deployments.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Field
     - Default
     - MinIO Notes
   * - ``s3_endpoint``
     - *(required)*
     - Virtual-hosted bucket URL, e.g. ``lmcache-kv.localhost:9000``
   * - ``s3_region``
     - *(required)*
     - Set to ``us-east-1`` unless your MinIO is configured otherwise
   * - ``disable_tls``
     - ``false``
     - Set to ``true`` for plain-HTTP MinIO instances
   * - ``aws_access_key_id``
     - *(none)*
     - MinIO root or service-account access key
   * - ``aws_secret_access_key``
     - *(none)*
     - MinIO root or service-account secret key
   * - ``s3_prefer_http2``
     - ``true``
     - Can be set to ``false`` for MinIO (HTTP/2 is not required)
   * - ``max_capacity_gb``
     - ``0``
     - Set > 0 to enable LMCache-side eviction


Troubleshooting
---------------

**Connection refused / timeout**
    Verify MinIO is running: ``curl http://localhost:9000/minio/health/live``
    should return HTTP 200.

**Access denied (403)**
    Check that ``aws_access_key_id`` and ``aws_secret_access_key`` match
    the credentials used to start MinIO (``MINIO_ROOT_USER`` /
    ``MINIO_ROOT_PASSWORD``).

**Bucket not found (404)**
    Ensure the bucket exists.  Create it with ``mc mb local/lmcache-kv``
    or through the MinIO console.

**TLS errors when disable_tls is not set**
    If MinIO is running without TLS (the default Docker configuration),
    set ``"disable_tls": true``.

**DNS resolution failure for virtual-hosted bucket URL**
    If ``lmcache-kv.localhost`` does not resolve, add an entry to
    ``/etc/hosts``: ``127.0.0.1 lmcache-kv.localhost``, or use the IP
    form: ``lmcache-kv.127.0.0.1:9000``.

.. seealso::

   :doc:`S3 adapter reference <s3>` for the full list of configuration
   fields and advanced options (S3 Express, HTTP/2 tuning, etc.).
