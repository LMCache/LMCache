#!/usr/bin/env python3
"""Resolve the vLLM / torch stack of an LMCache release at build time.

Release images and wheels follow the *latest stable vLLM on PyPI* at the moment
the release is built, and the torch version that vLLM pins. Nothing is pinned
in the repository; this script decides, and the release workflow passes the
result to every wheel and image build.

Rules:

* Candidate = newest stable (no pre-release, not yanked) vLLM on PyPI.
* A candidate is used only if the whole stack exists:
  - vLLM has manylinux x86_64 and aarch64 wheels on PyPI;
  - torch ``==<pin>`` has cu130 wheels for cp310..cp314 on x86_64 and aarch64
    (the LMCache wheel matrix);
  - the ``vllm/vllm-openai:v<version>`` image exists (lightweight image base).
  Otherwise walk back to the next older vLLM release (with a warning).
* CUDA 12.9 is enabled only if torch ``==<pin>`` has cu129 wheels for the same
  matrix *and* a vLLM GitHub release ships ``vllm-<version>+cu129`` wheels.
* A stable tag that points to the same commit as an RC tag reuses the stack
  that RC recorded (``release-stack.env`` asset on the RC GitHub release), so
  the stable release ships exactly what the RC tested.

Outputs ``KEY=VALUE`` lines (VLLM_VERSION, TORCH_VERSION, CU129,
VLLM_CU129_FIND_LINKS, SOURCE) to ``--out`` and, when set, to
``$GITHUB_OUTPUT`` (lower-case keys).

``--cu129-for-torch X`` only answers whether torch X has the cu129 wheels
(``cu129=true|false``); nightly and PR builds use it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

PYTHONS = ("310", "311", "312", "313", "314")
ARCHES = ("x86_64", "aarch64")
STACK_ASSET = "release-stack.env"
_INDEX_CACHE: dict[str, str] = {}


def _get(url: str, *, github: bool = False) -> bytes:
    headers = {"User-Agent": "lmcache-release-resolver"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if github and token:
        headers["Authorization"] = f"Bearer {token}"
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=60
            ) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(2**attempt)
    raise RuntimeError(f"GET {url} failed: {last}")


def _json(url: str, *, github: bool = False):
    return json.loads(_get(url, github=github))


def _exists(url: str) -> bool:
    try:
        _get(url)
        return True
    except urllib.error.HTTPError:
        return False


# Final releases only: X.Y[.Z...] with an optional .postN. Pre-releases (aN, bN,
# rcN), dev releases and local versions never match, so "latest" is always a
# stable vLLM. (vLLM's GitHub release *tag* for a final version can look like an
# RC, e.g. v0.33.0rc3 holds the 0.33.0 wheels, so the version comes from PyPI.)
STABLE_RE = re.compile(r"\d+(?:\.\d+)*(?:\.post\d+)?")


def is_stable(v: str) -> bool:
    return bool(STABLE_RE.fullmatch(v))


def _vkey(v: str) -> tuple[int, ...]:
    base, _, post = v.partition(".post")
    return tuple(int(x) for x in base.split(".")) + (int(post) if post else -1,)


def warn(msg: str) -> None:
    print(f"::warning::{msg}", file=sys.stderr)


def stable_vllm_versions() -> list[str]:
    """Stable vLLM versions on PyPI, newest first (pre-releases and yanked excluded)."""
    rel = _json("https://pypi.org/pypi/vllm/json")["releases"]
    out = [
        v
        for v, files in rel.items()
        if is_stable(v) and files and not all(f.get("yanked") for f in files)
    ]
    return sorted(out, key=_vkey, reverse=True)


def vllm_info(version: str) -> tuple[str | None, list[str]]:
    """(torch pin, wheel file names) of a vLLM release on PyPI."""
    d = _json(f"https://pypi.org/pypi/vllm/{version}/json")
    torch = None
    for req in d["info"].get("requires_dist") or []:
        m = re.match(r"^torch\s*==\s*([0-9][0-9.]*)\s*(;.*)?$", req)
        if m and (not m.group(2) or "extra" not in m.group(2)):
            torch = m.group(1)
            break
    return torch, [u["filename"] for u in d["urls"]]


def torch_on_index(torch: str, cuda: str) -> list[str]:
    """Missing (python, arch) pairs for torch==<torch>+<cuda> on the PyTorch index."""
    if cuda not in _INDEX_CACHE:
        _INDEX_CACHE[cuda] = (
            _get(f"https://download.pytorch.org/whl/{cuda}/torch/")
            .decode()
            .replace("%2B", "+")
        )
    page = _INDEX_CACHE[cuda]
    return [
        f"cp{py}/{arch}"
        for py in PYTHONS
        for arch in ARCHES
        if f"torch-{torch}+{cuda}-cp{py}-cp{py}-manylinux_2_28_{arch}.whl" not in page
    ]


def vllm_cu129_find_links(version: str, repo: str = "vllm-project/vllm") -> str:
    """find-links URL of the vLLM GitHub release that ships the +cu129 wheels, or ''."""
    want = {f"vllm-{version}+cu129-cp38-abi3-manylinux_2_28_{a}.whl" for a in ARCHES}
    for page in (1, 2, 3):
        for rel in _json(
            f"https://api.github.com/repos/{repo}/releases?per_page=30&page={page}",
            github=True,
        ):
            names = {a["name"] for a in rel.get("assets", [])}
            if want <= names:
                return f"https://github.com/{repo}/releases/expanded_assets/{rel['tag_name']}"
    return ""


def frozen_from_rc(repo: str, tag: str, sha: str) -> dict[str, str] | None:
    """Stack recorded by the newest RC tag of <tag> that points to <sha>."""
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag) or not sha:
        return None
    try:
        refs = _json(
            f"https://api.github.com/repos/{repo}/git/matching-refs/tags/{tag}rc",
            github=True,
        )
    except urllib.error.HTTPError:
        return None
    rcs = []
    for ref in refs:
        name = ref["ref"].removeprefix("refs/tags/")
        m = re.fullmatch(re.escape(tag) + r"rc(\d+)", name)
        obj = ref["object"]
        target = obj["sha"]
        if m and obj["type"] == "tag":  # annotated tag: dereference
            target = _json(obj["url"], github=True)["object"]["sha"]
        if m and target == sha:
            rcs.append((int(m.group(1)), name))
    for _, rc in sorted(rcs, reverse=True):
        try:
            rel = _json(
                f"https://api.github.com/repos/{repo}/releases/tags/{rc}", github=True
            )
        except urllib.error.HTTPError:
            continue
        for a in rel.get("assets", []):
            if a["name"] == STACK_ASSET:
                text = _get(a["browser_download_url"]).decode()
                stack = dict(
                    ln.split("=", 1)
                    for ln in text.splitlines()
                    if re.match(r"^[A-Z0-9_]+=", ln)
                )
                stack["SOURCE"] = f"frozen from {rc}"
                return stack
    return None


def resolve(max_walk_back: int) -> dict[str, str]:
    for i, v in enumerate(stable_vllm_versions()[: max_walk_back + 1]):
        torch, files = vllm_info(v)
        problems = []
        if not torch:
            problems.append("no exact torch pin")
        for arch in ARCHES:
            if not any(
                f.endswith(f"manylinux_2_28_{arch}.whl") or f"_{arch}." in f
                for f in files
            ):
                problems.append(f"no {arch} wheel on PyPI")
        if torch:
            miss = torch_on_index(torch, "cu130")
            if miss:
                problems.append(f"torch {torch}+cu130 missing for {', '.join(miss)}")
        if not _exists(
            f"https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v{v}"
        ):
            problems.append(f"no vllm/vllm-openai:v{v} image")
        if problems:
            warn(f"vLLM {v} skipped: {'; '.join(problems)}")
            continue
        if i:
            warn(f"Latest vLLM not usable; using vLLM {v} (walked back {i})")
        cu129_links = ""
        miss129 = torch_on_index(torch, "cu129")
        if miss129:
            print(
                f"CUDA 12.9 off: torch {torch}+cu129 missing for {', '.join(miss129)}",
                file=sys.stderr,
            )
        else:
            cu129_links = vllm_cu129_find_links(v)
            if not cu129_links:
                print(
                    f"CUDA 12.9 off: no vllm-{v}+cu129 wheels on a vLLM GitHub release",
                    file=sys.stderr,
                )
        return {
            "VLLM_VERSION": v,
            "TORCH_VERSION": torch,
            "CU129": "true" if cu129_links else "false",
            "VLLM_CU129_FIND_LINKS": cu129_links,
            "SOURCE": "resolved latest (PyPI) at "
            + time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
        }
    sys.exit(
        f"::error::No vLLM release among the newest {max_walk_back + 1} "
        "has a complete stack"
    )


def emit(values: dict[str, str], out: str | None) -> None:
    lines = [f"{k}={v}" for k, v in values.items()]
    print("\n".join(lines))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as f:
            f.write("".join(f"{k.lower()}={v}\n" for k, v in values.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY", "LMCache/LMCache")
    )
    ap.add_argument("--tag", default=os.environ.get("GITHUB_REF_NAME", ""))
    ap.add_argument("--sha", default=os.environ.get("GITHUB_SHA", ""))
    ap.add_argument("--out", default=STACK_ASSET)
    ap.add_argument("--max-walk-back", type=int, default=5)
    ap.add_argument(
        "--no-freeze",
        action="store_true",
        help="always resolve, never reuse an RC stack",
    )
    ap.add_argument(
        "--cu129-for-torch",
        metavar="TORCH",
        help="only report cu129=true|false for this torch",
    )
    a = ap.parse_args()

    if a.cu129_for_torch:
        miss = torch_on_index(a.cu129_for_torch, "cu129")
        if miss:
            print(
                f"torch {a.cu129_for_torch}+cu129 missing for {', '.join(miss)}",
                file=sys.stderr,
            )
        emit({"CU129": "false" if miss else "true"}, None)
        return

    stack = None if a.no_freeze else frozen_from_rc(a.repo, a.tag, a.sha)
    if stack is None:
        if re.fullmatch(r"v\d+\.\d+\.\d+", a.tag):
            warn(
                f"No RC of {a.tag} on this commit recorded {STACK_ASSET}; "
                "resolving the latest vLLM now"
            )
        stack = resolve(a.max_walk_back)
    # Hard guard, also for a stack reused from an LMCache RC tag: vLLM and torch
    # must be final releases.
    for key in ("VLLM_VERSION", "TORCH_VERSION"):
        if not is_stable(stack.get(key, "")):
            sys.exit(f"::error::{key}={stack.get(key)!r} is not a stable release")
    emit(stack, a.out)


if __name__ == "__main__":
    main()
