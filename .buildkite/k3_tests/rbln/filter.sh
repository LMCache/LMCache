#!/usr/bin/env bash
# Device-specific trigger condition and path rules for the RBLN pipeline.
# The common path filter reads this file to keep trigger metadata colocated
# with the device-specific pipeline configuration.
FILTER_DEVICE="rbln"
FILTER_TRIGGER_CONDITION='build.pull_request.labels includes "rbln" || build.pull_request.labels includes "full" || build.branch == "dev"'

_path_filter_rbln_should_skip() {
    local changed_file="$1"

    case "$changed_file" in
        .buildkite/k3_tests/rbln/*)
            return 1
            ;;
        .buildkite/k3_tests/*/*)
            return 0
            ;;
        tests/v1/platform/base/*|tests/v1/platform/devices/rbln/*)
            return 1
            ;;
        lmcache/v1/platform/base/*)
            return 1
            ;;
        lmcache/v1/platform/devices/__init__.py|lmcache/v1/platform/devices/rbln/*)
            return 1
            ;;
        tests/v1/platform/*|lmcache/v1/platform/*)
            # Direct files at either platform root are shared changes; nested
            # non-RBLN platform paths are not relevant to this pipeline.
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
