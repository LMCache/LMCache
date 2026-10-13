# SPDX-License-Identifier: Apache-2.0
"""Block until a just-published release is visible on the PyPI simple index.

The release jobs that build container images install the version this run just
uploaded, and they start seconds after the upload finishes. PyPI's simple index
is served through a CDN, so there is a window where the upload has succeeded
and the index does not list it yet. An installer that resolves inside that
window sees no such version and the job fails outright -- there is nothing to
back off to, because the pin is exact.

Polls the simple index rather than the JSON API: the two are cached
independently, and the simple index is what pip and uv actually read, so it is
the only one whose freshness predicts whether the build will resolve.

Usage: ``wait_for_pypi.py <package> <version> [--timeout SECONDS]``
"""

# Future
from __future__ import annotations

# Standard
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import argparse
import json
import sys
import time

SIMPLE_INDEX = "https://pypi.org/simple/{package}/"

# PEP 691. Without it the index answers with HTML, which _lists is happy to
# scan too, but the JSON form is what we parse when it is offered.
ACCEPT = "application/vnd.pypi.simple.v1+json, text/html;q=0.1"


def _fetch(package: str) -> str:
    """Return the simple-index body for ``package``, bypassing local caches."""
    request = Request(  # noqa: S310 - fixed https scheme, built from argv
        SIMPLE_INDEX.format(package=package),
        headers={"Accept": ACCEPT, "Cache-Control": "no-cache", "Pragma": "no-cache"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def _lists(body: str, package: str, version: str) -> bool:
    """Report whether the index body offers a file for exactly ``version``.

    The trailing separator is what keeps 0.5.6rc5 from matching 0.5.6rc51.
    """
    prefixes = (f"{package}-{version}-", f"{package}-{version}.")
    try:
        filenames = [entry["filename"] for entry in json.loads(body)["files"]]
    except (ValueError, KeyError, TypeError):
        # HTML index, or a shape we do not recognise; the filenames are still
        # in the body verbatim.
        return any(prefix in body.lower() for prefix in prefixes)
    return any(name.lower().startswith(prefixes) for name in filenames)


def main() -> int:
    """Poll until the version is listed. Returns 0 once it is."""
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("version", help="release version; a leading 'v' is stripped")
    parser.add_argument("--timeout", type=int, default=600, help="seconds")
    parser.add_argument("--interval", type=int, default=15, help="seconds")
    args = parser.parse_args()

    package = args.package.lower()
    version = args.version.lstrip("v").lower()

    deadline = time.monotonic() + args.timeout
    attempt = 0
    while True:
        attempt += 1
        try:
            body = _fetch(package)
        except (HTTPError, URLError, TimeoutError) as exc:
            # A 404 is normal for a package whose first release is still
            # propagating, so treat every fetch failure as "not yet".
            print(f"attempt {attempt}: index fetch failed ({exc})")
        else:
            if _lists(body, package, version):
                print(f"OK: {package}=={version} is listed on the PyPI simple index")
                return 0
            print(f"attempt {attempt}: {package}=={version} not listed yet")

        if time.monotonic() + args.interval >= deadline:
            print(
                f"::error::{package}=={version} did not appear on the PyPI simple "
                f"index within {args.timeout}s. The upload may have failed, or the "
                f"CDN is lagging further than expected; re-run this job."
            )
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
