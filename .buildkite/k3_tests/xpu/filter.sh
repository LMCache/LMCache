#!/usr/bin/env bash
# Device-specific trigger condition and path rules for the XPU pipeline.
# The common path filter reads this file to keep trigger metadata colocated
# with the device-specific pipeline configuration.
FILTER_DEVICE="xpu"
FILTER_TRIGGER_CONDITION='build.pull_request.labels includes "xpu" || build.pull_request.labels includes "full" || build.branch == "dev"'

_path_filter_xpu_should_skip() {
    local changed_file="$1"

    case "$changed_file" in
        .buildkite/k3_tests/xpu/*|.buildkite/k3_tests/common_scripts/*)
            return 1
            ;;
        .buildkite/k3_tests/*/*)
            return 0
            ;;
        tests/v1/platform/base/*|tests/v1/platform/xpu/*|lmcache/v1/platform/base/*|lmcache/v1/platform/xpu/*)
            return 1
            ;;
        tests/v1/platform/*|lmcache/v1/platform/*)
            # Direct files at either platform root are shared changes; nested
            # non-XPU platform paths are not relevant to this pipeline.
            if [[ "$changed_file" == tests/v1/platform/*/* || "$changed_file" == lmcache/v1/platform/*/* ]]; then
                return 0
            fi
            return 1
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
