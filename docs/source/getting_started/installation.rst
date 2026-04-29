.. _installation_guide:

Installation
============

**Prerequisites:** Linux · Python 3.9–3.13 · NVIDIA GPU (compute 7.0+) · CUDA 12.1+ · `uv <https://astral.sh/uv>`_

Install LMCache
---------------

.. tab-set::

    .. tab-item:: Python (pip / uv)

        .. tab-set::

            .. tab-item:: Stable  

                .. tab-set::

                    .. tab-item:: CUDA 12.9  

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            uv pip install lmcache

                        .. important::

                            You're all set! You can now start using LMCache. For hands-on guides and more
                            usage examples, see the :ref:`quickstart_examples` section.

                    .. tab-item:: CUDA 13.0

                        The CUDA 13.0 wheel is published to a dedicated
                        `GitHub Release <https://github.com/LMCache/LMCache/releases>`__ rather than PyPI.

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            VERSION=0.4.3  # replace with target release
                            uv pip install lmcache==${VERSION} \
                                --extra-index-url https://download.pytorch.org/whl/cu130 \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v${VERSION}-cu13 \
                                --index-strategy unsafe-best-match

                        .. note::

                            ``--extra-index-url https://download.pytorch.org/whl/cu130`` ensures the CUDA 13.0
                            build of PyTorch is resolved. Without it, pip may select a mismatched CUDA variant.

            .. tab-item:: Nightly

                Nightly wheels are built from the latest ``dev`` branch each day at 07:30 UTC
                and published to GitHub Releases. No version pinning required — ``--pre``
                picks the latest nightly automatically.

                .. tab-set::

                    .. tab-item:: CUDA 12.9

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            uv pip install lmcache --pre \
                                --extra-index-url https://download.pytorch.org/whl/cu129 \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/nightly \
                                --index-strategy unsafe-best-match

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            uv pip install lmcache --pre \
                                --extra-index-url https://download.pytorch.org/whl/cu130 \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/nightly-cu13 \
                                --index-strategy unsafe-best-match

            .. tab-item:: From Source

                ``--no-build-isolation`` ensures the kernels are compiled against the same torch
                already installed in your environment, preventing undefined symbol errors at runtime.

                .. tab-set::

                    .. tab-item:: CUDA

                        .. code-block:: bash

                            git clone https://github.com/LMCache/LMCache.git
                            cd LMCache

                            uv venv --python 3.12
                            source .venv/bin/activate

                            uv pip install -r requirements/build.txt
                            uv pip install vllm  # pulls in required torch version
                            uv pip install -e . --no-build-isolation

                    .. tab-item:: ROCm

                        .. code-block:: bash

                            git clone https://github.com/LMCache/LMCache.git
                            cd LMCache

                            uv venv --python 3.12
                            source .venv/bin/activate

                            # Need to install these packages manually to avoid build isolation
                            uv pip install -r requirements/build.txt

                            # Install torch from the ROCm wheel index
                            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0

                            # Build LMCache. BUILD_WITH_HIP=1 makes setup.py pick cupy-rocm-7-0 automatically.
                            PYTORCH_ROCM_ARCH="gfx942" \
                            TORCH_DONT_CHECK_COMPILER_ABI=1 \
                            CXX=hipcc \
                            BUILD_WITH_HIP=1 \
                            uv pip install -e . --no-build-isolation

    .. tab-item:: Docker

        .. tab-set::

            .. tab-item:: Stable

                .. tab-set::

                    .. tab-item:: CUDA 12.9

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai:latest-cu13

            .. tab-item:: Nightly

                .. tab-set::

                    .. tab-item:: CUDA 12.9

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai:latest-nightly

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai:latest-nightly-cu13

            .. tab-item:: ROCm

                .. code-block:: bash

                    docker pull rocm/vllm-dev:nightly_0624_rc2_0624_rc2_20250620

        See :ref:`docker_deployment` for running the container and ROCm images.

    .. tab-item:: CLI Only  

        Lightweight CLI-only package for querying or benchmarking a remote LMCache server.
        No CUDA required, works on any OS.

        .. code-block:: bash

            pip install lmcache-cli

        .. note::

            ``lmcache-cli`` and ``lmcache`` ship the same ``lmcache`` CLI command.
            Do not install both in the same environment.

Verify Installation
-------------------

.. code-block:: bash

    python -c "import lmcache.c_ops"

Compatibility Matrix
~~~~~~~~~~~~~~~~~~~~

✅ compatible · ❌ API incompatible · 🕯️ torch mismatch (use ``--no-build-isolation``)


.. container:: compat-table-scroll

   .. csv-table::
      :file: Installation_compatibility_matrix.csv
      :header-rows: 1
