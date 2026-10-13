# Temporary PR #5685 image validation

This draft-only experiment adds two image steps to the existing K3 unit pipeline.
Close it after collecting the results; it is not intended for merging.

Both images come from the unmodified Dockerfiles at
[PR #5685 commit 1541a889](https://github.com/LMCache/LMCache/commit/1541a8898702ec5160bb1c4d36db3f3bd7e852b0).
They target Linux amd64 and CUDA 13, using the Dockerfiles' default CUDA
architecture list: `7.5 8.0 8.6 8.9 9.0 10.0 12.0+PTX`. This includes the SM120
Blackwell GPUs in the K3 queue. The pipeline references use immutable digests.

The added steps leave the container UID to the image's `USER lmcache`; they do
not set `runAsUser`. Kubernetes supplies `HOME=/home/lmcache` and shared-memory
storage. The scripts run from that writable home directory, outside the checkout,
with `PYTHONPATH` unset and Python safe-path mode enabled. They check the installed
package path and generated source revision before running any GPU tests. They do
not source the normal per-job install scripts or reinstall LMCache, Torch, or vLLM.

- Standalone image step: default user, cache/shared-memory writes, native extension
  import, pinned host memory, native LMCache KV-transfer and CUDA IPC data round
  trips, and standalone server RPC.
- vLLM image step: the same GPU probe, followed by two identical requests to
  Qwen/Qwen2.5-0.5B-Instruct with vLLM prefix caching disabled. Both answers must be
  `42`; LMCache cached tokens must be zero cold and positive warm. Local disk cache
  files must also exist. This is a focused image smoke test, not the complete
  ShareGPT correctness suite.

The original unit job still runs independently with its usual CI base image.
Only the added image steps provide evidence about these published images. The
Buildkite correctness pipeline currently has a server-side skip for vLLM
`BATCH_INVARIANT`; it does not upload its repository pipeline definition. Both
image checks therefore live in the active unit pipeline.
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
  --build-arg max_jobs=1 --build-arg nvcc_threads=1 \
  --build-arg LMCACHE_COMMIT_ID=1541a8898702ec5160bb1c4d36db3f3bd7e852b0 \
  --tag maobaolong/lmcache-nonroot:vllm-1541a889-cu130-ci-v2 .

docker buildx build --push --platform linux/amd64 \
  --target lmcache-final --file docker/Dockerfile.standalone \
  --build-arg max_jobs=1 --build-arg nvcc_threads=1 \
  --build-arg LMCACHE_COMMIT_ID=1541a8898702ec5160bb1c4d36db3f3bd7e852b0 \
  --tag maobaolong/lmcache-nonroot:standalone-1541a889-cu130-ci-v2 .
```

Choose tags in your own registry namespace when reproducing the build. A dedicated
container-driver builder keeps its cache separate from other jobs; `--push` avoids
loading the full images into the host daemon's image store. Check the actual
builder volume filesystem and free space before building on a shared machine.

## Run the published checks outside CI

Use a Linux amd64 GPU host with Docker, NVIDIA Container Toolkit, and a CUDA 13
compatible NVIDIA driver. Choose an idle supported GPU in place of `device=0`. Fetch this draft's test scripts into a
separate checkout; the source revision used to build the images does not contain
them. No Python packages need installing on the host.

```bash
git clone --branch test/nonroot-image-ci \
  https://github.com/maobaolong/LMCache.git LMCache-image-tests
git -C LMCache-image-tests checkout --detach 4bc72686925d562e12839ac73d24ffc588168f86
cd LMCache-image-tests

docker run --name pr5685-standalone-check --gpus device=0 --shm-size=2g \
  --entrypoint bash -v "$PWD":/tests:ro \
  maobaolong/lmcache-nonroot@sha256:fb51f6d384ef5f3ee49ef365101b3535ddde18d68f3bed2554dcb285b17f4164 \
  /tests/.buildkite/k3_tests/unit/run-nonroot-image.sh
docker cp pr5685-standalone-check:/workspace/build ./standalone-logs
docker rm pr5685-standalone-check

docker run --name pr5685-vllm-check --gpus device=0 --shm-size=2g \
  --entrypoint bash -v "$PWD":/tests:ro \
  maobaolong/lmcache-nonroot@sha256:bf8b9e72bed42b8830e987420e4169ac32081433eccabc96b785475998951695 \
  /tests/.buildkite/k3_tests/correctness/run-nonroot-image.sh
docker cp pr5685-vllm-check:/workspace/build ./vllm-logs
docker rm pr5685-vllm-check
```

The checks run under the image's default user. The vLLM check downloads the public
model into the container's writable Hugging Face cache and can take several minutes
for download and compilation. Keep the exit status and logs from each `docker run`;
copying logs or removing the stopped container does not establish test success.
