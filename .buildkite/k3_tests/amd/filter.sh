#!/usr/bin/env bash
# Device-specific trigger condition and path rules for the AMD pipeline.
# The common path filter reads this file to keep trigger metadata colocated
# with the device-specific pipeline configuration.
FILTER_DEVICE="amd"
FILTER_TRIGGER_CONDITION='build.pull_request.labels includes "amd" || build.pull_request.labels includes "full" || build.branch == "dev"'

_path_filter_amd_should_skip() {
    local changed_file="$1"

    case "$changed_file" in
        tests/v1/platform/rocm/*|lmcache/v1/platform/rocm/*|.buildkite/k3_tests/amd/*)
            return 1
            ;;
        tests/v1/platform/*|lmcache/v1/platform/*)
            # Keep generic platform roots as non-targets for the ROCm pipeline;
            # only the ROCm subtree should trigger it.
            return 0
            ;;
        examples/*)
            # Example-only changes are not relevant to the ROCm pipeline.
            return 0
            ;;
        tests/*)
            return 1
            ;;
    esac

    return 1
}
