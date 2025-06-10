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




