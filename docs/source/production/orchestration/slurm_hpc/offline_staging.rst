.. _hpc_offline_staging:

Network-less / offline model staging
====================================

Many HPC sites **restrict or fully block outbound internet on compute nodes**.
This page assumes the fully offline case: anything that would normally be
downloaded at runtime -- model weights, tokenizers, pip packages -- is staged
from a login node first. It covers staging the model and tokenizer into a
Hugging Face cache on the shared filesystem, then running fully offline; it is
used by the :ref:`single-node sbatch template <hpc_single_node_submission>`.

Pre-download the model **and** tokenizer into a Hugging Face cache on the
shared filesystem from a login node (which has internet), then run fully
offline inside the job. With current ``huggingface_hub`` releases the CLI is
``hf`` (the old ``huggingface-cli`` entry point is deprecated and refuses to
run):

.. code-block:: bash

    # On a login node.
    export PROJECT=<shared_fs_path>/lmcache              # your shared project directory
    export HF_HOME=$PROJECT/hf_cache
    hf download <org/model>

Inside the job, point the container at that cache and forbid network access so a
missing file fails fast instead of hanging on a blocked download:

.. code-block:: bash

    export HF_HOME=$PROJECT/hf_cache
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

Bind-mount ``$HF_HOME`` into the container (see the
:ref:`single-node sbatch template <hpc_single_node_submission>`).

With ``HF_HUB_OFFLINE=1``, a model that is missing from the cache fails
immediately with a clear "couldn't connect" error instead of stalling on the
compute node's blocked network -- stage every model you plan to serve.
