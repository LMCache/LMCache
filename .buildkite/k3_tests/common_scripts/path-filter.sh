#!/usr/bin/env bash
# Path filter: decide whether a CI build can be skipped based on which files
# changed since the base commit and which k3 test pipeline is being uploaded.
#
# Usage:
#   source path-filter.sh
#   if should_skip_ci; then
#       # all changed files are trivial (docs, etc.)
#   fi
#
# Rules:
#   - If EVERY changed file matches a "trivial" pattern (markdown, LICENSE,
#     anything under docs/, asset/, or .github/, etc.), the build can be skipped.
#   - Otherwise, a test-specific dependency surface is used to decide whether
#     the uploaded pipeline is relevant for the changed files. This keeps
#     unrelated test pipelines from running when the PR touches a different
#     subsystem.
#   - Shared CI harness files fan out only to the tests that actually depend
#     on them. Suite-local files under .buildkite/k3_tests/<suite>/ stay scoped
#     to that suite, while shared harness changes under .buildkite/k3_harness/
#     are classified by the harness file they touch.
#   - Special-case platform tests are treated as unrelated for the unit
#     pipeline: tests/v1/platform/{musa,rbln,xpu} do not trigger unit. The xpu
#     pipeline also treats these platform paths as non-targets while allowing
#     tests/platform/* to remain relevant.
#
# Opt-out: add a "force-ci" label to the PR on GitHub. Buildkite exposes
# PR labels via BUILDKITE_PULL_REQUEST_LABELS; if "force-ci" is present
# the filter is bypassed and the full pipeline runs.
#
# Detection of "changed files":
#   - PR builds  → diff against the merge-base with BUILDKITE_PULL_REQUEST_BASE_BRANCH.
#   - Push builds → diff HEAD~1..HEAD.
#   - Anything we can't figure out → fall back to "do not skip".

set -uo pipefail

# ── Pattern lists ─────────────────────────────────────────────
# Use explicit directory prefixes for nested paths so shell `case` matching
# stays predictable across the different test surfaces.

_path_filter_is_trivial() {
    case "$1" in
        *.md) return 0 ;;
        LICENSE|LICENSE.*) return 0 ;;
        NOTICE|NOTICE.*) return 0 ;;
        .gitignore|.gitattributes|.editorconfig|.mailmap) return 0 ;;
        CODEOWNERS) return 0 ;;
        docs/*) return 0 ;;
        asset/*) return 0 ;;
        .github/*) return 0 ;;
    esac
    return 1
}

_path_filter_should_skip_for_pipeline() {
    local pipeline_kind="$1"
    local changed_file="$2"

    # 1) trivial files are always skipped.
    if _path_filter_is_trivial "$changed_file"; then
        return 0
    fi

    case "$pipeline_kind" in
        unit)
            case "$changed_file" in
                tests/v1/platform/*)
                    # skip all files under second-level platform dirs
                    if [[ "$changed_file" == tests/v1/platform/*/* ]]; then
                        return 0
                    fi
                    # direct pytest file under tests/v1/platform/ still counts
                    return 1
                    ;;
                lmcache/v1/platform/*)
                    # For CUDA unit, platform implementation changes under
                    # lmcache/v1/platform/{base,cuda}/ are relevant and should
                    # trigger the pipeline; keep other nested device directories
                    # as non-targets.
                    if [[ "$changed_file" == lmcache/v1/platform/base/* || "$changed_file" == lmcache/v1/platform/cuda/* ]]; then
                        return 1
                    fi
                    if [[ "$changed_file" == lmcache/v1/platform/*/* ]]; then
                        return 0
                    fi
                    return 1
                    ;;
                examples/*)
                    # Example-only changes should not trigger unit tests.
                    return 0
                    ;;
                .buildkite/k3_tests/unit/*|tests/*)
                    return 1
                    ;;
            esac
            ;;
        integration)
            case "$changed_file" in
                .buildkite/k3_tests/integration/*|.buildkite/configs/local_cpu.yaml|.buildkite/configs/local_disk.yaml)
                    return 1
                    ;;
                examples/*)
                    # Example-only changes should not trigger integration tests.
                    return 0
                    ;;
                tests/*|.buildkite/k3_tests/*)
                    return 0
                    ;;
            esac
            ;;
        correctness)
            case "$changed_file" in
                .buildkite/correctness/*|.buildkite/k3_tests/correctness/*)
                    return 1
                    ;;
                examples/*)
                    # Example-only changes should not trigger correctness tests.
                    return 0
                    ;;
                tests/*|.buildkite/k3_tests/*)
                    return 0
                    ;;
            esac
            ;;
        multiprocess)
            case "$changed_file" in
                benchmarks/long_doc_qa/*|.buildkite/k3_tests/multiprocess/*)
                    return 1
                    ;;
                examples/*)
                    # Example-only changes should not trigger multiprocess tests.
                    return 0
                    ;;
                tests/*|.buildkite/k3_tests/*)
                    return 0
                    ;;
            esac
            ;;
        comprehensive)
            case "$changed_file" in
                .buildkite/configs/*|benchmarks/*|.buildkite/k3_tests/comprehensive/*)
                    return 1
                    ;;
                examples/disagg_prefill*/*)
                    return 1
                    ;;
                examples/*)
                    # Keep other example changes out of the comprehensive pipeline.
                    # They are treated as safe no-op paths for this suite.
                    return 0
                    ;;
                tests/*|.buildkite/k3_tests/*)
                    return 0
                    ;;
            esac
            ;;
        blend)
            case "$changed_file" in
                .buildkite/k3_harness/setup-blend-env.sh|.buildkite/k3_tests/blend/*)
                    return 1
                    ;;
                examples/*)
                    # Example-only changes should not trigger blend tests.
                    return 0
                    ;;
                tests/*|.buildkite/k3_tests/*)
                    return 0
                    ;;
            esac
            ;;
        sglang)
            case "$changed_file" in
                .buildkite/k3_harness/setup-sglang-env.sh|lmcache/integration/sglang/*|.buildkite/k3_tests/sglang/*)
                    return 1
                    ;;
                examples/*)
                    # Example-only changes should not trigger sglang tests.
                    return 0
                    ;;
                tests/*|.buildkite/k3_tests/*)
                    return 0
                    ;;
            esac
            ;;
        xpu)
            if declare -F "_path_filter_xpu_should_skip" >/dev/null 2>&1; then
                _path_filter_xpu_should_skip "$changed_file"
                return $?
            fi
            return 1
            ;;
        amd)
            if declare -F "_path_filter_amd_should_skip" >/dev/null 2>&1; then
                _path_filter_amd_should_skip "$changed_file"
                return $?
            fi
            return 1
            ;;
        musa)
            # Placeholder only: no dedicated MUSA pipeline dir is introduced here.
            # Keep the platform kind explicit for future CI extension.
            return 1
            ;;
        rbln)
            # Placeholder only: no dedicated RBLN pipeline dir is introduced here.
            # Keep the platform kind explicit for future CI extension.
            return 1
            ;;
        neuron)
            # Placeholder only: no dedicated Neuron pipeline dir is introduced here.
            # Keep the platform kind explicit for future CI extension.
            return 1
            ;;
        generic)
            return 1
            ;;
    esac

    # 4) fallback: unknown / not explicitly classified files are treated as
    # relevant and therefore do not skip the pipeline.
    return 1
}

_path_filter_pipeline_kind() {
    case "$1" in
        *k3_tests/unit/pipeline.yml) echo unit ;;
        *k3_tests/integration/pipeline.yml) echo integration ;;
        *k3_tests/correctness/pipeline.yml) echo correctness ;;
        *k3_tests/multiprocess/pipeline.yml) echo multiprocess ;;
        *k3_tests/blend/pipeline.yml) echo blend ;;
        *k3_tests/sglang/pipeline.yml) echo sglang ;;
        *k3_tests/xpu/pipeline.yml) echo xpu ;;
        *k3_tests/musa/pipeline.yml) echo musa ;;
        *k3_tests/amd/pipeline.yml) echo amd ;;
        *k3_tests/comprehensive/pipeline.yml) echo comprehensive ;;
        *) echo generic ;;
    esac
}

_path_filter_load_device_filter() {
    local pipeline_file="${1:-}"
    local pipeline_dir filter_script

    [[ -n "$pipeline_file" ]] || return 0
    pipeline_dir="$(cd "$(dirname "$pipeline_file")" 2>/dev/null && pwd)" || return 0
    filter_script="${pipeline_dir}/filter.sh"

    if [[ -f "$filter_script" ]]; then
        # shellcheck disable=SC1090
        source "$filter_script"
        if [[ -n "${FILTER_TRIGGER_CONDITION:-}" ]]; then
            echo "path-filter: loaded ${filter_script} trigger: ${FILTER_TRIGGER_CONDITION}" >&2
        fi
    fi
}

# ── Changed-files detection ───────────────────────────────────

_path_filter_get_changed_files() {
    local base_branch base merge_base

    # Ephemeral pods may not have GitHub's SSH host key yet.
    # Accept new keys automatically so git-fetch doesn't hang on a prompt.
    export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR"

    if [[ -n "${BUILDKITE_PULL_REQUEST:-}" && "${BUILDKITE_PULL_REQUEST:-}" != "false" ]]; then
        base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
        # Buildkite checks out shallow; fetch enough history to find the merge-base.
        git fetch --no-tags --depth=200 origin "$base_branch" 2>/dev/null || \
            git fetch --no-tags origin "$base_branch" 2>/dev/null || true

        if base=$(git rev-parse --verify "origin/${base_branch}" 2>/dev/null); then
            if merge_base=$(git merge-base HEAD "$base" 2>/dev/null); then
                git diff --name-only "$merge_base" HEAD
                return 0
            fi
            # No merge-base (history not deep enough): diff directly.
            git diff --name-only "$base" HEAD
            return 0
        fi
        echo "path-filter: could not resolve origin/${base_branch}" >&2
        return 1
    fi

    # Push build (or unknown context): diff against the previous commit.
    if git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        git diff --name-only HEAD~1 HEAD
        return 0
    fi

    echo "path-filter: no parent commit available" >&2
    return 1
}

# ── Public entry point ────────────────────────────────────────

# Returns 0 if the build can be safely skipped, non-zero otherwise.
# Prints a classification of every changed file to stderr for the build log.
should_skip_ci() {
    local pipeline_file="${1:-}"
    local pipeline_kind

    # PR label opt-out: adding "force-ci" on GitHub forces a full run.
    if [[ ",${BUILDKITE_PULL_REQUEST_LABELS:-}," == *",force-ci,"* ]]; then
        echo "path-filter: PR has 'force-ci' label → not skipping" >&2
        return 1
    fi

    # Never skip scheduled builds (e.g. nightly baselines with NEED_UPLOAD=true).
    if [[ "${BUILDKITE_SOURCE:-}" == "schedule" ]]; then
        echo "path-filter: scheduled build (BUILDKITE_SOURCE=schedule) → not skipping" >&2
        return 1
    fi

    local changed_files
    if ! changed_files=$(_path_filter_get_changed_files); then
        echo "path-filter: could not determine changed files → not skipping" >&2
        return 1
    fi

    if [[ -z "$changed_files" ]]; then
        echo "path-filter: no changed files reported → not skipping (safer default)" >&2
        return 1
    fi

    pipeline_kind="$(_path_filter_pipeline_kind "$pipeline_file")"
    _path_filter_load_device_filter "$pipeline_file"

    local has_non_trivial=0
    local has_relevant_change=0
    local has_unrelated_change=0
    local trivial_count=0
    local total=0

    echo "path-filter: classifying changed files for ${pipeline_kind}:" >&2
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        total=$((total + 1))
        if _path_filter_is_trivial "$f"; then
            trivial_count=$((trivial_count + 1))
            echo "  [trivial]       $f" >&2
        else
            has_non_trivial=1
            if _path_filter_should_skip_for_pipeline "$pipeline_kind" "$f"; then
                has_unrelated_change=1
                echo "  [unrelated]     $f" >&2
            else
                has_relevant_change=1
                echo "  [relevant]      $f" >&2
            fi
        fi
    done <<< "$changed_files"

    echo "path-filter: ${total} files changed (${trivial_count} trivial)" >&2

    if [[ "$has_non_trivial" -eq 1 ]]; then
        if [[ "$has_relevant_change" -eq 1 ]]; then
            echo "path-filter: relevant files changed for ${pipeline_kind} → not skipping" >&2
            return 1
        fi

        if [[ "$has_unrelated_change" -eq 1 ]]; then
            echo "path-filter: only unrelated files changed for ${pipeline_kind} → SKIP" >&2
            return 0
        fi
    fi

    echo "path-filter: no relevant files changed for ${pipeline_kind} → SKIP" >&2
    return 0
}
