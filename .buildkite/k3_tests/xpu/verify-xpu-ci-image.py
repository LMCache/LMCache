# SPDX-License-Identifier: Apache-2.0
"""Gate promotion of an XPU CI image on both Buildkite hardware pipelines."""

# Standard
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import json
import os
import time


def buildkite_request(
    path: str, token: str, payload: dict[str, object] | None = None
) -> dict[str, object]:
    """Send an authenticated Buildkite API request and return its JSON object.

    Args:
        path: API path relative to v2.
        token: Buildkite API token with read_builds and write_builds scopes.
        payload: JSON body for a POST, or None for a GET.

    Returns:
        Parsed Buildkite response.

    Raises:
        RuntimeError: If the API request or response is invalid.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"https://api.buildkite.com/v2/{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Buildkite API request failed for {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Invalid Buildkite API response for {path}")
    return result


def require_string(record: dict[str, object], key: str) -> str:
    """Read a required, nonempty string from a Buildkite response.

    Args:
        record: Buildkite build object.
        key: Required field name.

    Returns:
        The field value.

    Raises:
        RuntimeError: If the field is missing or invalid.
    """
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Buildkite build has no valid {key}: {record}")
    return value


def main() -> None:
    """Trigger both XPU suites on the candidate image, failing closed."""
    token = os.environ["BUILDKITE_API_TOKEN"]
    image = os.environ["CANDIDATE_IMAGE"]
    commit = os.environ["SOURCE_COMMIT"]
    branch = os.environ["SOURCE_BRANCH"]
    if not all((token, image, commit, branch)):
        raise RuntimeError("Buildkite token, image, commit, and branch are required")

    builds: dict[str, tuple[str, str, int]] = {}
    for name, slug in (("unit", "unit-tests-xpu"), ("multiprocess", "xpu-mp-test")):
        path = f"organizations/lmcache/pipelines/{quote(slug, safe='')}/builds"
        result = buildkite_request(
            path,
            token,
            {
                "commit": commit,
                "branch": branch,
                "message": f"XPU nightly candidate {image}",
                "env": {
                    "PINNED_XPU_IMAGE": image,
                    "XPU_CANDIDATE_VALIDATION": "1",
                },
            },
        )
        number = result.get("number")
        if not isinstance(number, int):
            raise RuntimeError(f"Buildkite did not return a build number: {result}")
        url = require_string(result, "web_url")
        builds[name] = (f"{path}/{number}", url, number)
        print(f"Triggered {name}: {url}", flush=True)

    pending = dict(builds)
    deadline = time.monotonic() + 210 * 60
    while pending:
        for name, (path, url, _) in list(pending.items()):
            state = require_string(buildkite_request(path, token), "state")
            print(f"{name}: {state} ({url})", flush=True)
            if state == "passed":
                del pending[name]
            elif state in {
                "failed",
                "canceled",
                "canceling",
                "timed_out",
                "blocked",
                "skipped",
                "not_run",
            }:
                raise RuntimeError(f"{name} XPU CI did not pass: {state} ({url})")
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"XPU CI timed out: {pending}")
            time.sleep(30)

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"unit_build_url={builds['unit'][1]}\n")
        output.write(f"unit_build_number={builds['unit'][2]}\n")


if __name__ == "__main__":
    main()
