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

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            uv pip install lmcache

                        .. important::

                            You're all set! You can now start using LMCache. For hands-on guides and more
                            usage examples, see the :ref:`quickstart_examples` section.

                        .. note::

                            NIXL support (e.g. for disaggregated prefill and P2P KV
                            sharing) is an optional extra:

                            .. code-block:: bash

                                uv pip install lmcache[nixl]

                    .. tab-item:: CUDA 12.9

                        The CUDA 12.9 wheel is published to a dedicated
                        `GitHub Release <https://github.com/LMCache/LMCache/releases>`__ rather than PyPI.

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            VERSION=0.4.3  # replace with target release
                            uv pip install lmcache==${VERSION} \
                                --extra-index-url https://download.pytorch.org/whl/cu129 \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v${VERSION}-cu129 \
                                --index-strategy unsafe-best-match

                        .. note::

                            ``--extra-index-url https://download.pytorch.org/whl/cu129`` ensures the CUDA 12.9
                            build of PyTorch is resolved. Without it, pip may select a mismatched CUDA variant.

                    .. tab-item:: ROCm / torch 2.11

                        The ROCm wheel targets AMD Instinct **gfx942** (MI300X / MI325X) and
                        **gfx950** (MI350X / MI355X) in one fat binary, and is ABI-matched to the
                        upstream ``vllm/vllm-openai-rocm`` image (torch 2.11, ROCm 7.2, Python 3.12).
                        It is published to a dedicated
                        `GitHub Release <https://github.com/LMCache/LMCache/releases>`__ rather than PyPI.

                        Install directly inside an upstream vLLM ROCm container — torch and the
                        ROCm runtime are already present, so ``--no-deps`` binds against them:

                        .. code-block:: bash

                            docker run -it --device /dev/kfd --device /dev/dri \
                                --group-add video --security-opt seccomp=unconfined \
                                --entrypoint bash vllm/vllm-openai-rocm:v0.25.0

                            VERSION=0.5.3  # replace with target release
                            pip install lmcache==${VERSION}+rocm7.2 --no-deps \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v${VERSION}-rocm

                        .. note::

                            The wheel excludes torch and the ROCm runtime libraries (they bind to the
                            host image at runtime). Match the wheel's minor torch/ROCm version to your
                            container; for other bases, use the **From Source** tab.

                        .. note::

                            The ROCm wheel carries a ``+rocm7.2`` PEP 440 local version, so
                            ``pip show lmcache`` reports which build is installed and the ROCm
                            build can be requested explicitly. A bare ``lmcache==${VERSION}``
                            also resolves it, since ``==`` ignores the local segment.

                    .. tab-item:: ROCm 7.2.4 / torch 2.10

                        This wheel targets AMD Instinct **gfx942** (MI300X / MI325X) and
                        **gfx950** (MI350X / MI355X). It is built and smoke-tested in the
                        public AMD PyTorch image
                        ``rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0``
                        pinned at digest
                        ``sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e``.
                        It does not require the ATOM image.

                        The supported ABI tuple is exact:

                        * AMD wheel source:
                          `torch-2.10.0+rocm7.2.4.lw.git3d3aa833-cp312-cp312-linux_x86_64.whl
                          <https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torch-2.10.0%2Brocm7.2.4.lw.git3d3aa833-cp312-cp312-linux_x86_64.whl>`__
                        * torch runtime version: ``2.10.0+rocm7.2.4.git3d3aa833``
                          (git ``3d3aa833db84eed6b7f5595cb5f162c2f78300a4``)
                        * ROCm ``7.2.4`` with HIP runtime ``7.2.53211``
                        * Python/platform tag: ``cp312-cp312-manylinux_2_39_x86_64``
                        * C++ ABI: ``_GLIBCXX_USE_CXX11_ABI=1``

                        Install the matching wheel inside that pinned image:

                        .. code-block:: bash

                            docker run -it --device /dev/kfd --device /dev/dri \
                                --group-add video --security-opt seccomp=unconfined \
                                --entrypoint bash \
                                rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e

                            VERSION=0.5.4  # replace with target release
                            pip install \
                                lmcache==${VERSION}+rocm7.2.4.torch2.10.git3d3aa833.cxx11abi1 \
                                --no-deps \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v${VERSION}-rocm-torch210

                        The wheel links torch and ROCm libraries from the container at runtime.
                        Other torch 2.10, ROCm 7.2.x, Python, or C++ ABI combinations are not
                        covered by this artifact; build from source for those environments.

            .. tab-item:: Nightly

                Nightly wheels are built from the latest ``dev`` branch each day at 07:30 UTC
                and published to GitHub Releases. No version pinning required — ``--pre``
                picks the latest nightly automatically.

                .. tab-set::

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            uv pip install lmcache --pre \
                                --extra-index-url https://download.pytorch.org/whl/cu130 \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/nightly \
                                --index-strategy unsafe-best-match

                    .. tab-item:: CUDA 12.9

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            uv pip install lmcache --pre \
                                --extra-index-url https://download.pytorch.org/whl/cu129 \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/nightly-cu129 \
                                --index-strategy unsafe-best-match

                    .. tab-item:: ROCm 7.2

                        Run inside an upstream vLLM ROCm container so torch and the ROCm
                        runtime are already present, then install with ``--no-deps``:

                        .. code-block:: bash

                            docker run -it --device /dev/kfd --device /dev/dri \
                                --group-add video --security-opt seccomp=unconfined \
                                --entrypoint bash vllm/vllm-openai-rocm:v0.26.0

                            pip install lmcache --pre --no-deps --no-index \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/nightly-rocm

                        Nightly ROCm wheels are versioned like the CUDA nightlies with the
                        ROCm local segment appended, e.g. ``0.5.4.dev15+rocm7.2``.

                        .. note::

                            ``--no-index`` is required here. ``--find-links`` only *adds* a
                            source, so without it pip also considers PyPI — and under PEP 440 a
                            pre-release such as ``0.5.4rc4`` outranks ``0.5.4.dev15+rocm7.2``, so
                            ``--pre`` would install the CUDA wheel instead. The stable tab does
                            not need it because ``lmcache==${VERSION}+rocm7.2`` is an exact pin
                            that only the ROCm release can satisfy. ``--no-deps`` is what makes
                            ``--no-index`` safe here: torch and the ROCm runtime come from the
                            container, so nothing else needs resolving.

            .. tab-item:: From Source

                ``--no-build-isolation`` ensures the kernels are compiled against the same torch
                already installed in your environment, preventing undefined symbol errors at runtime.

                .. tab-set::

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            git clone https://github.com/LMCache/LMCache.git
                            cd LMCache

                            uv venv --python 3.12
                            source .venv/bin/activate

                            uv pip install -r requirements/build.txt
                            uv pip install vllm  # pulls in required torch version (cu13)
                            uv pip install -e . --no-build-isolation

                    .. tab-item:: CUDA 12.9

                        .. code-block:: bash

                            git clone https://github.com/LMCache/LMCache.git
                            cd LMCache

                            uv venv --python 3.12
                            source .venv/bin/activate

                            uv pip install -r requirements/build.txt
                            # Pin vLLM (and torch) to the cu12.9 wheel index so the local
                            # CUDA 12 toolchain matches what the extensions are built against.
                            uv pip install vllm \
                                --extra-index-url https://download.pytorch.org/whl/cu129 \
                                --index-strategy unsafe-best-match
                            # LMCACHE_CUDA_MAJOR=12 makes setup.py pick cupy-cuda12x
                            # for install_requires instead of the cu13 default.
                            LMCACHE_CUDA_MAJOR=12 \
                                uv pip install -e . --no-build-isolation

                    .. tab-item:: ROCm

                        .. code-block:: bash

                            git clone https://github.com/LMCache/LMCache.git
                            cd LMCache

                            uv venv --python 3.12
                            source .venv/bin/activate

                            # Need to install these packages manually to avoid build isolation
                            uv pip install -r requirements/build.txt

                            # Install torch from the ROCm wheel index. Use the rocm7.2 index to
                            # match the upstream vllm/vllm-openai-rocm image (torch 2.11, ROCm 7.2).
                            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2

                            # Build LMCache. BUILD_WITH_HIP=1 makes setup.py pick cupy-rocm-7-0 automatically.
                            # PYTORCH_ROCM_ARCH selects the target GPU(s):
                            #   gfx942  -> MI300X / MI325X
                            #   gfx950  -> MI350X / MI355X
                            # Comma-separate to build a fat binary for multiple archs.
                            PYTORCH_ROCM_ARCH="gfx942,gfx950" \
                            TORCH_DONT_CHECK_COMPILER_ABI=1 \
                            CXX=hipcc \
                            BUILD_WITH_HIP=1 \
                            uv pip install -e . --no-build-isolation

                    .. tab-item:: Intel XPU

                        .. code-block:: bash

                            git clone https://github.com/LMCache/LMCache.git
                            cd LMCache

                            uv venv --python 3.12
                            source .venv/bin/activate

                            # Need to install these packages manually to avoid build isolation
                            uv pip install -r requirements/build.txt

                            # Build LMCache with SYCL backend.
                            BUILD_WITH_SYCL=1 uv pip install --no-build-isolation -e .


    .. tab-item:: Docker

        .. tab-set::

            .. tab-item:: Stable

                .. tab-set::

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai

                    .. tab-item:: CUDA 12.9

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai:latest-cu129

            .. tab-item:: Nightly

                .. tab-set::

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai:latest-nightly

                    .. tab-item:: CUDA 12.9

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai:latest-nightly-cu129

            .. tab-item:: ROCm

                .. code-block:: bash

                    docker pull rocm/vllm-dev:nightly_0624_rc2_0624_rc2_20250620

            .. tab-item:: Intel XPU

                .. code-block:: bash

                    docker pull intel/vllm:0.17.0-xpu

        See :ref:`docker_deployment` for running the container and ROCm images.

    .. tab-item:: CLI Only  

        Lightweight CLI-only package for querying or benchmarking a remote LMCache server.
        No CUDA required, works on any OS.

        .. code-block:: bash

            pip install lmcache-cli

        .. note::

            ``lmcache-cli`` and ``lmcache`` ship the same ``lmcache`` CLI command.
            Do not install both in the same environment.

Build the Docker Image
----------------------

Instead of pulling a prebuilt image, you can build the LMCache (integrated with
vLLM) image yourself from the provided Dockerfile, located in
`docker/ <https://github.com/LMCache/LMCache/tree/dev/docker>`_.

From the root of the LMCache repository:

.. code-block:: bash

    docker build --tag <IMAGE_NAME>:<TAG> --target image-build --file docker/Dockerfile .

Replace ``<IMAGE_NAME>`` and ``<TAG>`` with your desired image name and tag. See
the example build file in `docker/ <https://github.com/LMCache/LMCache/tree/dev/docker>`_
for an explanation of all build arguments.

Verify Installation
-------------------

.. code-block:: bash

    python -c "import lmcache.cuda_ops"
