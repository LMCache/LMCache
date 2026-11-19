# SPDX-License-Identifier: Apache-2.0
"""Download the XPU wheel built by the current GitHub Actions run."""

# Standard
from http.client import HTTPMessage
from pathlib import Path
from typing import IO
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import hashlib
import io
import os
import re
import tempfile
import zipfile


class _GitHubRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        url: str,
    ) -> Request | None:
        redirected = super().redirect_request(request, fp, code, msg, headers, url)
        if (
            redirected is not None
            and urlsplit(url).hostname != urlsplit(request.full_url).hostname
        ):
            redirected.remove_header("Authorization")
        return redirected


def main() -> None:
    """Fetch and verify the exact wheel artifact requested by the verifier."""
    artifact_id = os.environ["XPU_WHEEL_ARTIFACT_ID"]
    expected_digest = os.environ["XPU_WHEEL_ARTIFACT_DIGEST"]
    token = os.environ["GITHUB_TOKEN"]
    if (
        not re.fullmatch(r"[0-9]+", artifact_id)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest)
        or not token
    ):
        raise ValueError(
            "Wheel artifact ID, SHA-256 digest, and GitHub token are required"
        )

    request = Request(
        f"https://api.github.com/repos/LMCache/LMCache/actions/artifacts/{artifact_id}/zip",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with build_opener(_GitHubRedirects()).open(request, timeout=120) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != expected_digest.removeprefix("sha256:"):
        raise RuntimeError(
            f"XPU wheel artifact {artifact_id} failed SHA-256 validation"
        )

    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        files = bundle.infolist()
        if len(files) != 1 or not re.fullmatch(
            r"lmcache-[A-Za-z0-9_.+-]+\.whl", files[0].filename
        ):
            raise RuntimeError(
                f"XPU wheel artifact {artifact_id} must contain one wheel"
            )
        wheel_path = (
            Path(tempfile.mkdtemp(prefix="lmcache-xpu-wheel-")) / files[0].filename
        )
        wheel_path.write_bytes(bundle.read(files[0]))
    print(wheel_path)


if __name__ == "__main__":
    main()
