# SPDX-License-Identifier: Apache-2.0
"""Run the weekly full build on Buildkite and decide whether it passed.

The pipelines and the jobs a full run must contain are listed in
``.buildkite/k3_tests/weekly/manifest.json``. The script creates one build per
pipeline on the manifest's CI-owned branch, waits for all of them, and then
checks every job. A run counts as passed only when every build passed and
every required job ran and passed: a failed, soft-failed, skipped or missing
job, or a suite the path filter skipped, fails the run.

It writes ``report.json`` and ``summary.md`` to the output directory, saves the
logs of failed jobs there when the token may read them, and exits 1 unless the
run passed.
"""

# Standard
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import argparse
import json
import os
import re
import sys
import time

API = "https://api.buildkite.com/v2"
FINISHED_BUILD_STATES = {"passed", "failed", "canceled", "skipped", "not_run"}
# Job states of a step that never ran, e.g. because its `if` was false.
NOT_RUN_JOB_STATES = {"broken", "skipped", "not_run"}
FAILED_JOB_STATES = {"failed", "timed_out"}


def api_request(
    url: str, token: str, payload: dict[str, Any] | None = None, raw: bool = False
) -> Any:
    """Call the Buildkite REST API.

    Args:
        url: Full API URL.
        token: Buildkite API token.
        payload: JSON body for a POST, or None for a GET.
        raw: Return the response body as text instead of parsed JSON.

    Returns:
        The parsed JSON value, or the body text when ``raw`` is true.

    Raises:
        HTTPError: If the API rejects the request, or a transient GET failure
            (HTTP 429 or 5xx) persists after three attempts. POSTs are not
            retried because the build may exist despite a lost response.
        URLError: If the network fails three times for a GET, or once for a
            POST.
        TimeoutError: If reading the response times out three times for a GET,
            or once for a POST.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "text/plain" if raw else "application/json",
            "Content-Type": "application/json",
        },
    )
    attempts = 3 if payload is None else 1
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=60) as response:
                body = response.read()
            return body.decode("utf-8", errors="replace") if raw else json.loads(body)
        except HTTPError as exc:
            transient = exc.code == 429 or 500 <= exc.code < 600
            if not transient or attempt == attempts - 1:
                raise
            exc.close()
        except (URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt * 5)
    raise AssertionError("unreachable")


def fetch_annotations(build_path: str, token: str) -> list[dict[str, Any]]:
    """Fetch all annotations for a Buildkite build across pages.

    Args:
        build_path: Buildkite REST URL for the build.
        token: Buildkite API token.

    Returns:
        All annotation objects on the build.
    """
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = api_request(f"{build_path}/annotations?per_page=100&page={page}", token)
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items


def evaluate(
    manifest: dict[str, Any],
    builds: dict[str, dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Decide whether a set of finished builds is a successful full run.

    Args:
        manifest: Parsed ``manifest.json``.
        builds: Pipeline slug to the Buildkite build object, including its
            ``jobs``. A slug that is missing means its build was not created.
        annotations: Pipeline slug to the build's annotation objects.

    Returns:
        ``{"passed": bool, "pipelines": [...]}``. Each pipeline entry has the
        build URL and state, the ``problems`` that fail the run (each with a
        ``kind``: ``build``, ``failed``, ``soft_failed``, ``not_run``,
        ``missing``, ``suite_skipped`` or ``workload_skipped``), and the
        labels of jobs that passed only after an automatic retry.
    """
    expected_unsupported = [re.compile(p) for p in manifest["expected_unsupported"]]
    skip_prefix = manifest["workload_skip_annotation_prefix"]
    pipelines = []
    for spec in manifest["pipelines"]:
        slug = spec["slug"]
        build = builds.get(slug)
        entry: dict[str, Any] = {
            "slug": slug,
            "name": spec["name"],
            "url": build["web_url"] if build else None,
            "number": build["number"] if build else None,
            "state": build["state"] if build else "not_created",
            "problems": [],
            "retried": [],
        }
        pipelines.append(entry)
        problems = entry["problems"]
        if build is None:
            problems.append({"kind": "build", "detail": "build was not created"})
            continue
        if build["state"] != "passed":
            problems.append({"kind": "build", "detail": f"build {build['state']}"})

        jobs = [job for job in build.get("jobs", []) if job.get("type") == "script"]
        final = [job for job in jobs if not job.get("retried")]
        entry["retried"] = sorted({job["name"] for job in jobs if job.get("retried")})
        allowed_not_run = [re.compile(p) for p in spec["not_run_allowed"]]
        for job in final:
            label = job.get("name") or ""
            state = job.get("state")
            if state == "passed" and not job.get("soft_failed"):
                continue
            if state in NOT_RUN_JOB_STATES:
                if any(p.search(label) for p in allowed_not_run):
                    continue
                kind = "not_run"
            elif state == "passed":
                kind = "soft_failed"
            else:
                kind = "failed"
            problems.append(
                {
                    "kind": kind,
                    "label": label,
                    "state": state,
                    "exit_status": job.get("exit_status"),
                    "url": job.get("web_url"),
                    "job_id": job.get("id"),
                }
            )

        passed_labels = [
            job["name"]
            for job in final
            if job.get("state") == "passed" and not job.get("soft_failed")
        ]
        for pattern in spec["required"]:
            if not any(re.search(pattern, label) for label in passed_labels):
                problems.append({"kind": "missing", "label": pattern})

        for annotation in annotations.get(slug, []):
            context = annotation.get("context", "")
            text = re.sub(r"<[^>]+>", "", annotation.get("body_html", "")).strip()
            if context in manifest["suite_skip_annotations"]:
                problems.append(
                    {"kind": "suite_skipped", "label": context, "detail": text}
                )
            elif context.startswith(skip_prefix) and not any(
                p.search(text) for p in expected_unsupported
            ):
                problems.append(
                    {"kind": "workload_skipped", "label": context, "detail": text}
                )

    return {
        "passed": all(not p["problems"] for p in pipelines),
        "pipelines": pipelines,
    }


def render_summary(report: dict[str, Any]) -> str:
    """Format a report from :func:`evaluate` as GitHub-flavoured Markdown.

    Args:
        report: Report with ``passed``, ``commit``, ``timed_out`` and
            ``pipelines``.

    Returns:
        Markdown for the job summary and the saved ``summary.md``.
    """
    verdict = "PASSED" if report["passed"] else "FAILED"
    lines = [
        f"## Weekly full build: {verdict}",
        "",
        f"Commit `{report['commit']}` on `{report['branch']}`.",
    ]
    if report.get("timed_out"):
        lines.append(
            "The builds did not finish before the deadline; they keep running "
            "on Buildkite."
        )
    lines += ["", "| Pipeline | Build | State | Problems |", "|---|---|---|---|"]
    for p in report["pipelines"]:
        link = f"[#{p['number']}]({p['url']})" if p["url"] else "-"
        lines.append(f"| {p['name']} | {link} | {p['state']} | {len(p['problems'])} |")
    for p in report["pipelines"]:
        if p["problems"]:
            lines += ["", f"### {p['name']}", ""]
            for problem in p["problems"]:
                label = problem.get("label") or problem.get("detail", "")
                url = problem.get("url")
                text = f"[{label}]({url})" if url else f"`{label}`"
                detail = problem.get("detail")
                if detail and problem.get("label"):
                    text += f": {detail}"
                lines.append(f"- **{problem['kind']}** {text}")
        if p["retried"]:
            lines += ["", f"Passed after a retry in {p['name']}:", ""]
            lines += [f"- {label}" for label in p["retried"]]
    return "\n".join(lines) + "\n"


def emit_annotations(report: dict[str, Any]) -> None:
    """Print GitHub workflow commands that annotate the run with the result.

    The dashboard reads these annotations through the public check-runs API,
    so each line keeps a fixed shape: the title is the pipeline slug, and a
    pipeline line is ``<state> <build url>``. Pipeline lines come first
    because GitHub keeps only the first 10 errors of a step.

    Args:
        report: Report from :func:`evaluate`.
    """
    for pipeline in report["pipelines"]:
        level = "notice" if not pipeline["problems"] else "error"
        slug, state, url = pipeline["slug"], pipeline["state"], pipeline["url"]
        print(f"::{level} title={slug}::{state} {url}")
    for pipeline in report["pipelines"]:
        for problem in pipeline["problems"]:
            label = problem.get("label") or problem.get("detail", "")
            where = problem.get("url") or ""
            kind = problem["kind"]
            print(f"::error title={pipeline['slug']}::{kind} {label} {where}".rstrip())


def save_failed_logs(
    report: dict[str, Any], org: str, token: str, out_dir: Path
) -> list[str]:
    """Save the logs of failed jobs next to the report.

    Args:
        report: Report from :func:`evaluate`.
        org: Buildkite organization slug.
        token: Buildkite API token; reading logs needs ``read_build_logs``.
        out_dir: Directory that receives a ``logs`` subdirectory.

    Returns:
        Notes about logs that could not be saved.
    """
    notes = []
    log_dir = out_dir / "logs"
    for pipeline in report["pipelines"]:
        for problem in pipeline["problems"]:
            if problem["kind"] != "failed" or not problem.get("job_id"):
                continue
            url = (
                f"{API}/organizations/{org}/pipelines/{pipeline['slug']}"
                f"/builds/{pipeline['number']}/jobs/{problem['job_id']}/log.txt"
            )
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", problem["label"]).strip("_")
            try:
                text = api_request(url, token, raw=True)
            except (HTTPError, URLError) as exc:
                notes.append(f"log of {problem['label']} not saved: {exc}")
                continue
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{pipeline['slug']}-{name}.log").write_text(text)
    return notes


def run(
    manifest: dict[str, Any], commit: str, token: str, timeout_minutes: int
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], bool]:
    """Create one build per pipeline and wait until all of them finish.

    Args:
        manifest: Parsed ``manifest.json``.
        commit: Commit to test; it must be on the manifest's branch.
        token: Buildkite API token with read_builds and write_builds.
        timeout_minutes: How long to wait before giving up.

    Returns:
        The last build objects by slug, their annotations by slug, and whether
        the deadline passed before every build finished.

    Raises:
        HTTPError: If Buildkite rejects a request.
        URLError: If Buildkite cannot be reached.
    """
    org = manifest["organization"]
    paths = {}
    for spec in manifest["pipelines"]:
        path = f"{API}/organizations/{org}/pipelines/{quote(spec['slug'], safe='')}"
        build = api_request(
            f"{path}/builds",
            token,
            {
                "commit": commit,
                "branch": manifest["branch"],
                "message": f"Weekly full build of {commit[:12]}",
                "env": manifest["env"],
                # Pipelines exclude ci-weekly-full from push builds (README.md).
                "ignore_pipeline_branch_filters": True,
            },
        )
        paths[spec["slug"]] = f"{path}/builds/{build['number']}"
        print(f"Created {spec['slug']}: {build['web_url']}", flush=True)

    builds: dict[str, dict[str, Any]] = {}
    deadline = time.monotonic() + timeout_minutes * 60
    pending = set(paths)
    while True:
        for slug in sorted(pending):
            builds[slug] = api_request(paths[slug], token)
            print(f"{slug}: {builds[slug]['state']}", flush=True)
            if builds[slug]["state"] in FINISHED_BUILD_STATES:
                pending.discard(slug)
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(60)

    annotations = {slug: fetch_annotations(paths[slug], token) for slug in paths}
    return builds, annotations, bool(pending)


def main() -> int:
    """Run the weekly full build and write its report.

    Returns:
        0 when the run passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout-minutes", type=int, default=330)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    token = os.environ["BUILDKITE_API_TOKEN"]
    builds, annotations, timed_out = run(
        manifest, args.commit, token, args.timeout_minutes
    )
    report = evaluate(manifest, builds, annotations)
    report.update(
        commit=args.commit,
        branch=manifest["branch"],
        timed_out=timed_out,
        passed=report["passed"] and not timed_out,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report["log_notes"] = save_failed_logs(
        report, manifest["organization"], token, args.out_dir
    )
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    summary = render_summary(report)
    (args.out_dir / "summary.md").write_text(summary)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(summary)
    print(summary)
    emit_annotations(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
