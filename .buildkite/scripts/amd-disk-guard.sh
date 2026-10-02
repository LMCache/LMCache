#!/usr/bin/env bash
# Keep the AMD bare-metal runner's root disk from filling up.
#
# Source this from any AMD job script before it pulls images or installs
# packages. When free space on / drops below AMD_DISK_GUARD_MIN_FREE_GB
# (default 60), it removes only what the next build regenerates anyway:
#   - this account's stale pip/setuptools temp dirs under /tmp
#   - the uv package cache
#   - the triton JIT cache
#   - dangling (untagged) images, stopped containers and build cache
# Tagged images, the HuggingFace cache and anything owned by other users are
# never touched. The same check runs again when the job exits so the next job
# can at least start; the agent cannot even initialize a job on a full disk.
#
# The exit handler also hands the checkout back to the agent account. Jobs
# here run containers as root with the checkout bind-mounted, which leaves
# root-owned build output behind; the next job's `git clean` cannot remove it
# and the agent's fallback of deleting the checkout fails the same way.
#
# Usage: source .buildkite/scripts/amd-disk-guard.sh
# Scripts that install their own EXIT trap must call amd_job_cleanup from it.

AMD_DISK_GUARD_MIN_FREE_GB="${AMD_DISK_GUARD_MIN_FREE_GB:-60}"

amd_disk_free_gb() {
    df --output=avail -BG / | tail -1 | tr -dc '0-9'
}

amd_disk_cleanup() {
    local docker=(docker)
    docker info >/dev/null 2>&1 || docker=(sudo docker)
    find /tmp -maxdepth 1 -user "$(id -un)" -name 'tmp*' -mtime +1 \
        -exec rm -rf {} + 2>/dev/null || true
    uv cache prune >/dev/null 2>&1 || true
    rm -rf "${HOME}/.triton/cache" 2>/dev/null || true
    "${docker[@]}" container prune -f >/dev/null 2>&1 || true
    "${docker[@]}" image prune -f >/dev/null 2>&1 || true
    "${docker[@]}" builder prune -f >/dev/null 2>&1 || true
}

amd_disk_guard() {
    local free
    free="$(amd_disk_free_gb)"
    if (( free >= AMD_DISK_GUARD_MIN_FREE_GB )); then
        return 0
    fi
    echo "--- :floppy_disk: ${free}G free on /, below ${AMD_DISK_GUARD_MIN_FREE_GB}G; clearing regenerable caches"
    amd_disk_cleanup
    echo "${free}G -> $(amd_disk_free_gb)G free on /"
}

amd_job_cleanup() {
    sudo -n chown -R "$(id -u):$(id -g)" "${REPO_ROOT:-$PWD}" 2>/dev/null || true
    amd_disk_guard
}

amd_disk_guard
trap 'rc=$?; amd_job_cleanup; exit $rc' EXIT
