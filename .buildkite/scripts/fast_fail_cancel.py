# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Fast-Fail & Cancel: stop sibling CI runs on first required-check failure.

Supports bidirectional cancellation using only the Python standard library:
- Cancels sibling GitHub Actions workflow runs for the same commit (PR events).
- Cancels sibling Buildkite builds for the same commit (and PR number when set).
"""

# Standard
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

PROTECTED_BRANCHES = {"dev", "main", "master"}
CANCELLABLE_BK_STATES = {"running", "scheduled", "failing", "creating"}


def get_token_buildkite(explicit_token: Optional[str]) -> Optional[str]:
    """Retrieve Buildkite API token from CLI argument or environment."""
    if explicit_token:
        return explicit_token
    return os.environ.get("BUILDKITE_API_TOKEN") or os.environ.get("BK_TOKEN")


def get_token_github(explicit_token: Optional[str]) -> Optional[str]:
    """Retrieve GitHub token from CLI argument or environment."""
    if explicit_token:
        return explicit_token
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _normalize_pr_number(raw: Optional[str]) -> Optional[str]:
    """Normalize PR number, ignoring empty or Buildkite 'false'/'0' sentinels."""
    if not raw:
        return None
    val = str(raw).strip()
    if not val or val.lower() in {"false", "0", "none", "null"}:
        return None
    return val


def _emit_warning(msg: str) -> None:
    """Print warning and emit GitHub Actions ::warning:: annotation when in GHA."""
    print(f"  [WARN] {msg}")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning::{msg}")


def _http_request(
    url: str,
    method: str = "GET",
    token: Optional[str] = None,
    auth_header_prefix: str = "Bearer ",
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    retries: int = 3,
) -> Tuple[int, Any]:
    """Send an HTTP request using urllib with retry on transient network errors."""
    req_headers = {
        "Accept": "application/json",
        "User-Agent": "LMCache-CI-FastFail",
    }
    if token:
        req_headers["Authorization"] = f"{auth_header_prefix}{token}"
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    last_err = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                body = response.read()
                if body:
                    try:
                        return status_code, json.loads(body.decode("utf-8"))
                    except Exception:
                        return status_code, body.decode("utf-8")
                return status_code, None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body
        except Exception as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(1.0)
                continue
            return -1, last_err
    return -1, last_err


def cancel_github_actions(
    commit_sha: str,
    repo: str,
    token: str,
    current_run_id: Optional[str],
    pr_number: Optional[str] = None,
    dry_run: bool = False,
    reason: str = "",
) -> List[Dict[str, Any]]:
    """Find and cancel sibling pull_request GitHub Actions runs for commit_sha."""
    print(
        f"\n[Fast-Fail] Querying GitHub Actions runs for commit "
        f"{commit_sha[:8]} in {repo}..."
    )
    cancelled: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    norm_pr = _normalize_pr_number(pr_number)

    for status in ("in_progress", "queued"):
        url = (
            f"https://api.github.com/repos/{repo}/actions/runs"
            f"?head_sha={commit_sha}&status={status}&per_page=100"
        )
        status_code, resp = _http_request(url, token=token, auth_header_prefix="token ")
        if status_code in (401, 403):
            _emit_warning(
                f"GitHub token unauthorized (HTTP {status_code}) listing {status} runs."
            )
            continue
        if status_code != 200 or not isinstance(resp, dict):
            print(
                f"  [WARN] Failed to list GHA {status} runs "
                f"(HTTP {status_code}): {resp}"
            )
            continue
        runs.extend(resp.get("workflow_runs", []))

    unique_runs = {str(r.get("id")): r for r in runs if r.get("id")}

    for run_id_str, run in unique_runs.items():
        if current_run_id and run_id_str == str(current_run_id):
            print(
                f"  -> Skipping current runner run ID: {run_id_str} ({run.get('name')})"
            )
            continue

        # Only cancel pull_request runs; never cancel scheduled/push/rerun jobs.
        event = run.get("event")
        if event and event != "pull_request":
            print(
                f"  -> Skipping non-PR event '{event}': "
                f"run {run_id_str} ({run.get('name')})"
            )
            continue

        run_prs = run.get("pull_requests") or []
        if norm_pr and run_prs:
            pr_nums = {
                str(p.get("number")) for p in run_prs if p.get("number") is not None
            }
            if pr_nums and norm_pr not in pr_nums:
                print(
                    f"  -> Skipping run {run_id_str} for PR(s) {sorted(pr_nums)} "
                    f"(expected PR #{norm_pr})"
                )
                continue

        branch = run.get("head_branch")
        if branch in PROTECTED_BRANCHES and not run_prs:
            print(f"  -> Skipping protected branch {branch}: run {run_id_str}")
            continue

        run_info = {
            "id": run_id_str,
            "name": run.get("name"),
            "status": run.get("status"),
            "url": run.get("html_url"),
        }

        if dry_run:
            print(
                f"  [DRY-RUN] Would cancel GHA run {run_id_str}: "
                f"{run.get('name')} ({run.get('html_url')})"
            )
            cancelled.append(run_info)
        else:
            cancel_url = (
                f"https://api.github.com/repos/{repo}/actions/runs/{run_id_str}/cancel"
            )
            c_status, c_resp = _http_request(
                cancel_url,
                method="POST",
                token=token,
                auth_header_prefix="token ",
            )
            if c_status in (200, 202):
                print(
                    f"  [CANCELLED] GHA run {run_id_str}: {run.get('name')} ({reason})"
                )
                cancelled.append(run_info)
            elif c_status in (401, 403):
                _emit_warning(
                    f"GitHub token lacks permission to cancel run {run_id_str} "
                    f"(HTTP {c_status}): {c_resp}"
                )
            else:
                print(
                    f"  [WARN] Failed to cancel GHA run {run_id_str} "
                    f"(HTTP {c_status}): {c_resp}"
                )

    return cancelled


def cancel_buildkite_builds(
    commit_sha: str,
    org: str,
    token: str,
    current_build_id: Optional[str],
    pr_number: Optional[str] = None,
    dry_run: bool = False,
    reason: str = "",
) -> List[Dict[str, Any]]:
    """Find and cancel sibling PR Buildkite builds for commit_sha."""
    print(
        f"\n[Fast-Fail] Querying Buildkite builds for commit "
        f"{commit_sha[:8]} in org '{org}'..."
    )
    cancelled: List[Dict[str, Any]] = []
    norm_pr = _normalize_pr_number(pr_number)

    url = (
        f"https://api.buildkite.com/v2/organizations/{org}/builds"
        f"?commit={commit_sha}&per_page=100"
    )
    status_code, resp = _http_request(url, token=token)
    if status_code in (401, 403):
        _emit_warning(
            f"Buildkite token unauthorized (HTTP {status_code}) listing builds."
        )
        return cancelled
    if status_code != 200 or not isinstance(resp, list):
        print(f"  [WARN] Failed to list Buildkite builds (HTTP {status_code}): {resp}")
        return cancelled

    for build in resp:
        state = build.get("state")
        if state not in CANCELLABLE_BK_STATES:
            continue

        build_id = str(build.get("id"))
        build_num = build.get("number")
        pipeline = build.get("pipeline", {}).get("slug", "unknown")
        branch = build.get("branch")
        web_url = build.get("web_url")
        build_pr = build.get("pull_request") or {}
        raw_pr_id = build_pr.get("id")
        build_pr_id = str(raw_pr_id) if raw_pr_id is not None else None

        if current_build_id and build_id == str(current_build_id):
            print(f"  -> Skipping current Buildkite build: #{build_num} ({pipeline})")
            continue

        if not build_pr_id:
            print(
                f"  -> Skipping non-PR Buildkite build: "
                f"#{build_num} ({pipeline}, branch={branch})"
            )
            continue

        if norm_pr and build_pr_id != norm_pr:
            print(
                f"  -> Skipping build #{build_num} for PR #{build_pr_id} "
                f"(expected PR #{norm_pr})"
            )
            continue

        build_info = {
            "id": build_id,
            "number": build_num,
            "pipeline": pipeline,
            "web_url": web_url,
        }

        if dry_run:
            print(
                f"  [DRY-RUN] Would cancel Buildkite build #{build_num} "
                f"({pipeline}): {web_url}"
            )
            cancelled.append(build_info)
        else:
            cancel_url = (
                f"https://api.buildkite.com/v2/organizations/{org}"
                f"/pipelines/{pipeline}/builds/{build_num}/cancel"
            )
            c_status, c_resp = _http_request(cancel_url, method="PUT", token=token)
            if c_status in (200, 202):
                print(
                    f"  [CANCELLED] Buildkite build #{build_num} "
                    f"({pipeline}) ({reason})"
                )
                cancelled.append(build_info)
            elif c_status in (401, 403):
                _emit_warning(
                    f"Buildkite token lacks permission to cancel #{build_num} "
                    f"(HTTP {c_status}): {c_resp}"
                )
            else:
                print(
                    f"  [WARN] Failed to cancel Buildkite build #{build_num} "
                    f"(HTTP {c_status}): {c_resp}"
                )

    return cancelled


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Fast-Fail CI Cancel: stop sibling GHA runs and Buildkite builds.")
    )
    parser.add_argument(
        "--commit",
        default=os.environ.get("BUILDKITE_COMMIT") or os.environ.get("GITHUB_SHA"),
        help="Git commit SHA to cancel sibling builds for",
    )
    parser.add_argument(
        "--pr-number",
        default=os.environ.get("BUILDKITE_PULL_REQUEST") or os.environ.get("PR_NUMBER"),
        help="Pull request number to scope GHA and Buildkite cancellation",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "LMCache/LMCache"),
        help="GitHub repository owner/name (default: LMCache/LMCache)",
    )
    parser.add_argument(
        "--org",
        default=os.environ.get("BUILDKITE_ORGANIZATION_SLUG", "lmcache"),
        help="Buildkite organization slug (default: lmcache)",
    )
    parser.add_argument(
        "--current-run-id",
        default=os.environ.get("GITHUB_RUN_ID"),
        help="Current GitHub Actions run ID to exclude from cancellation",
    )
    parser.add_argument(
        "--current-build-id",
        default=os.environ.get("BUILDKITE_BUILD_ID"),
        help="Current Buildkite build ID to exclude from cancellation",
    )
    parser.add_argument(
        "--github-token", help="GitHub token with actions:write permission"
    )
    parser.add_argument(
        "--buildkite-token", help="Buildkite token with write_builds permission"
    )
    parser.add_argument(
        "--no-cancel-gha",
        action="store_true",
        help="Do not cancel GitHub Actions runs",
    )
    parser.add_argument(
        "--no-cancel-bk",
        action="store_true",
        help="Do not cancel Buildkite builds",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without cancelling"
    )
    parser.add_argument(
        "--reason",
        default="Fast-fail: required check failed",
        help="Reason for cancellation",
    )

    args = parser.parse_args()

    if not args.commit:
        print(
            "[Fast-Fail] Error: No commit SHA specified "
            "(--commit or BUILDKITE_COMMIT/GITHUB_SHA).",
            file=sys.stderr,
        )
        return 0

    print("=== Fast-Fail Cancel Triggered ===")
    print(f"Commit: {args.commit}")
    print(f"Reason: {args.reason}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE CANCEL'}")

    gh_token = get_token_github(args.github_token)
    bk_token = get_token_buildkite(args.buildkite_token)

    total_gha_cancelled = 0
    total_bk_cancelled = 0

    if not args.no_cancel_gha:
        if gh_token:
            gha_cancelled = cancel_github_actions(
                commit_sha=args.commit,
                repo=args.repo,
                token=gh_token,
                current_run_id=args.current_run_id,
                pr_number=args.pr_number,
                dry_run=args.dry_run,
                reason=args.reason,
            )
            total_gha_cancelled = len(gha_cancelled)
        else:
            _emit_warning("Skipping GHA cancellation (no GITHUB_TOKEN available).")

    if not args.no_cancel_bk:
        if bk_token:
            bk_cancelled = cancel_buildkite_builds(
                commit_sha=args.commit,
                org=args.org,
                token=bk_token,
                current_build_id=args.current_build_id,
                pr_number=args.pr_number,
                dry_run=args.dry_run,
                reason=args.reason,
            )
            total_bk_cancelled = len(bk_cancelled)
        else:
            _emit_warning(
                "Skipping Buildkite cancellation (no BUILDKITE_API_TOKEN available)."
            )

    print("\n=== Fast-Fail Summary ===")
    print(f"GitHub Actions runs cancelled: {total_gha_cancelled}")
    print(f"Buildkite builds cancelled:    {total_bk_cancelled}")
    print("=========================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
