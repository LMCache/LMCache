# Temporary PR #5685 image validation

This draft-only experiment adds two steps to the existing K3 unit and correctness
pipelines. Close it after collecting the results; it is not intended for merging.

Both images come from the unmodified Dockerfiles at
[PR #5685 commit 1541a889](https://github.com/LMCache/LMCache/commit/1541a8898702ec5160bb1c4d36db3f3bd7e852b0).
They target Linux amd64, CUDA 13, and NVIDIA SM80/SM90 (A100/H200). The pipeline
references must use the published immutable digests.

The added steps leave the container UID to the image's `USER lmcache`; they do
not set `runAsUser`. Kubernetes supplies `HOME=/home/lmcache` and shared-memory
storage. The scripts run from that writable home directory, outside the checkout,
with `PYTHONPATH` unset and Python safe-path mode enabled. They check the installed
package path and generated source revision before running any GPU tests. They do not source
the normal per-job install scripts or reinstall LMCache, Torch, or vLLM.

- Unit pipeline: default user, cache/shared-memory writes, native extension import,
  pinned host memory, LMCache CUDA IPC data round trip, and standalone server RPC.
- Correctness pipeline: the same GPU probe, followed by two identical requests to
  Qwen/Qwen2.5-0.5B-Instruct with vLLM prefix caching disabled. Both answers must be
  `42`; LMCache cached tokens must be zero cold and positive warm. Local disk cache
  files must also exist. This is a focused image smoke test, not the complete
  ShareGPT correctness suite.

The original unit and correctness jobs still run independently with their usual
CI base image. Only the added steps provide evidence about these published images.
The IPC check verifies transferred data; it does not assert allocator lifecycle
behavior. Server, request, package-version, and UID logs are uploaded as artifacts.

## Rebuild

Prerequisites: Linux amd64 Docker with Buildx, Git, sufficient build-cache storage
(allow roughly 100 GB), outbound package/registry access, and Docker Hub credentials
with push access to the chosen test repository. GPU hardware is needed to run the
CI steps, but not to compile these images. Use a full clone so setuptools-scm can
resolve the package version. The vLLM nightly dependency can change on a later
rebuild; the published digest identifies the binaries tested by this experiment.

```bash
git clone https://github.com/LMCache/LMCache.git LMCache-nonroot
git -C LMCache-nonroot fetch origin refs/pull/5685/head
git -C LMCache-nonroot checkout --detach 1541a8898702ec5160bb1c4d36db3f3bd7e852b0
cd LMCache-nonroot
git describe --tags --long --match 'v[0-9]*.[0-9]*' --exclude 'v*-*'
docker login
docker buildx create --name nonroot-validation --driver docker-container --use
docker buildx inspect --bootstrap

docker buildx build --push --platform linux/amd64 \
  --target image-build --file docker/Dockerfile \
  --build-arg 'torch_cuda_arch_list=8.0 9.0' \
  --build-arg max_jobs=1 --build-arg nvcc_threads=1 \
  --build-arg LMCACHE_COMMIT_ID=1541a8898702ec5160bb1c4d36db3f3bd7e852b0 \
  --tag maobaolong/lmcache-nonroot:vllm-1541a889-cu130-sm80-sm90 .

docker buildx build --push --platform linux/amd64 \
  --target lmcache-final --file docker/Dockerfile.standalone \
  --build-arg 'torch_cuda_arch_list=8.0 9.0' \
  --build-arg max_jobs=1 --build-arg nvcc_threads=1 \
  --build-arg LMCACHE_COMMIT_ID=1541a8898702ec5160bb1c4d36db3f3bd7e852b0 \
  --tag maobaolong/lmcache-nonroot:standalone-1541a889-cu130-sm80-sm90 .
```

Choose tags in your own registry namespace when reproducing the build. A dedicated
container-driver builder keeps its cache separate from other jobs; `--push` avoids
loading the full images into the host daemon's image store. Check the actual
builder volume filesystem and free space before building on a shared machine.

## Run the published checks outside CI

Use a Linux amd64 GPU host with Docker and NVIDIA Container Toolkit. Choose an
idle supported GPU in place of `device=0`. Fetch this draft's test scripts into a
separate checkout; the source revision used to build the images does not contain
them. No Python packages need installing on the host.

```bash
git clone --branch test/nonroot-image-ci \
  https://github.com/maobaolong/LMCache.git LMCache-image-tests
cd LMCache-image-tests

docker run --name pr5685-standalone-check --gpus device=0 --shm-size=2g \
  --entrypoint bash -v "$PWD":/tests:ro \
  maobaolong/lmcache-nonroot@sha256:7bf268a0ec9a68498245440fab65b2a603c8c0c5b1ede0075232b3efa51309df \
  /tests/.buildkite/k3_tests/unit/run-nonroot-image.sh
docker cp pr5685-standalone-check:/workspace/build ./standalone-logs
docker rm pr5685-standalone-check

docker run --name pr5685-vllm-check --gpus device=0 --shm-size=2g \
  --entrypoint bash -v "$PWD":/tests:ro \
  maobaolong/lmcache-nonroot@sha256:110ed03488f7688f4552f7c61ae0ff41814c8b013c17b711c27cded862f02174 \
  /tests/.buildkite/k3_tests/correctness/run-nonroot-image.sh
docker cp pr5685-vllm-check:/workspace/build ./vllm-logs
docker rm pr5685-vllm-check
```

The checks run under the image's default user. The vLLM check downloads the public
model into the container's writable Hugging Face cache and can take several minutes
for download and compilation. Keep the exit status and logs from each `docker run`;
copying logs or removing the stopped container does not establish test success.
