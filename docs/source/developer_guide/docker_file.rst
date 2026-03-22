Dockerfile
==========

We provide a Dockerfile to help you build a container image for LMCache integrated with vLLM.
More information about deploying LMCache image using Docker can be found here - :ref:`Docker deployment guide <docker_deployment>`.

Building the container image
----------------------------

You can build the LMCache (integrated with vLLM) image using Docker from source via the provided Dockerfile.
The Dockerfile is located at `docker <https://github.com/LMCache/LMCache/tree/dev/docker>`_.

To build the container image, run the following command from the root directory of the LMCache repository:

.. code-block:: bash

    docker build --tag <IMAGE_NAME>:<TAG> --target image-build --file docker/Dockerfile .

Replace `<IMAGE_NAME>` and `<TAG>` with your desired image name and tag. See example build file in `docker <https://github.com/LMCache/LMCache/tree/dev/docker>`_
for explanation of all arguments.

CUDA-specific behavior
----------------------

The Dockerfile supports both CUDA 12 and CUDA 13 container builds.

- CUDA 12 uses the published LMCache wheel from PyPI together with the CUDA 12 dependency stack.
- CUDA 13 switches the image to the CUDA 13 runtime packages (for example ``cupy-cuda13x`` and ``nixl-cu13``) and builds LMCache from source inside the image so the extension matches the local CUDA and Torch toolchain.

When you need to target a non-default CUDA version, pass ``--build-arg CUDA_VERSION=<major.minor>`` to ``docker build``.



