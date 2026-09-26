#!/usr/bin/env bash
# Returns 0 to skip a runtime-specific suite, 1 to keep it enabled.
# Shared/mixed suites and unrecognized selectors always retain coverage.
should_skip_ci_mode() {
    local suite_mode="${1:-shared}"
    local mode="${LMCACHE_CI_MODE:-}"
    local labels=",${BUILDKITE_PULL_REQUEST_LABELS:-},"

    if [[ "$labels" == *",force-ci,"* || "${BUILDKITE_SOURCE:-}" == "schedule" ]]; then
        return 1
    fi

    if [[ -z "$mode" ]]; then
        mode=all
        if [[ "$labels" == *",ci-mp,"* && "$labels" != *",ci-inprocess,"* ]]; then
            mode=mp
        elif [[ "$labels" == *",ci-inprocess,"* && "$labels" != *",ci-mp,"* ]]; then
            mode=inprocess
        fi
    fi

    case "$mode" in
        all|inprocess|mp) ;;
        *)
            echo "ci-mode: unknown LMCACHE_CI_MODE='$mode'; keeping both modes enabled" >&2
            return 1
            ;;
    esac

    case "$mode:$suite_mode" in
        mp:inprocess|inprocess:mp)
            echo "ci-mode: selected $mode; skipping $suite_mode suite" >&2
            return 0
            ;;
    esac
    return 1
}
