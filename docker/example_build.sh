# Example script to build the LMCache container image

# Update the following variables accordingly
# Note: latest (built from GitHub) images of vLLM are in AWS ECR registry:
# public.ecr.aws/q9t5s3a7/vllm-ci-test-repo
# The latest release version of vLLM imnage are on DockerHub: vllm/vllm-openai
CUDA_VERSION=12.8.1
DOCKERFILE_NAME='Dockerfile'
DOCKER_BUILD_PATH='../' # This path should point to the LMCache root for access to 'requirements' directory
UBUNTU_VERSION=22.04
VLLM_IMAGE_REPO='vllm/vllm-openai'
VLLM_TAG='latest'

docker build \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg UBUNTU_VERSION=$UBUNTU_VERSION \
    --build-arg VLLM_IMAGE_REPO=$VLLM_IMAGE_REPO \
    --build-arg VLLM_TAG=$VLLM_TAG \
    -f $DOCKERFILE_NAME $DOCKER_BUILD_PATH
