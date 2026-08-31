#!/usr/bin/env bash
# Device-specific trigger condition and path rules for the XPU pipeline.
# The common path filter reads this file to keep trigger metadata colocated
# with the device-specific pipeline configuration.
FILTER_DEVICE="xpu"
FILTER_TRIGGER_CONDITION='build.pull_request.labels includes "xpu" || build.pull_request.labels includes "full" || build.branch == "dev"'

_path_filter_xpu_should_skip() {
    local changed_file="$1"

    case "$changed_file" in
        tests/v1/platform/xpu/*|lmcache/v1/platform/xpu/*|.buildkite/k3_tests/xpu/*)
            return 1
            ;;
        tests/v1/platform/*|lmcache/v1/platform/*)
            # Keep generic platform roots as non-targets for xpu while allowing
            # the xpu-specific subtrees to trigger this pipeline.
            return 0
            ;;
        examples/*)
            # Example-only changes are not relevant to the device pipeline.
            return 0
            ;;
        tests/*)
            return 1
            ;;
    esac

    return 1
}
