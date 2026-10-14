# SPDX-License-Identifier: Apache-2.0
"""Tonight's images: LMCache built FROM a vLLM serving image pinned for a week.

Flow::

    pinned serving image younger than VLLM_REFRESH_S: reuse it   ("pinned")
    else: build serving@nightly                                   ("nightly")
    -> build lmcache FROM it (when a pair of the suite uses LMCache)

A failed build leaves the night with no images; there is no fallback. A new
serving image is pinned (:func:`pin`) only after a valid suite, so a broken
vLLM nightly is retried the next night. State lives in
``<host.root>/prt-images/pinned.json``.
"""

# Standard
from pathlib import Path
from typing import Any
import json
import subprocess
import time

# First Party
from perf_regression_testing.runner import docker
from perf_regression_testing.runner.config import PKG_DIR

BUILD_SH = PKG_DIR / "docker" / "build.sh"
# Rebuilt every night; anything not kept is removed by :func:`prune`.
NIGHTLY_REPOS = ("prt-serving", "prt-lmcache", "prt-mooncake")
# A new vLLM is a long build plus cold compile, and pinning it leaves LMCache
# as the only thing that changes between nights.
VLLM_REFRESH_S = 7 * 86400


def build(args: list[str], log: Path) -> str:
    """Run build.sh, logging to ``log``; return the tag it printed.

    Args:
        args: build.sh arguments (target first).
        log: File that receives the build output.

    Returns:
        The image tag.

    Raises:
        RuntimeError: when the build fails; the message names the log.
    """
    with open(log, "w") as f:
        rc = subprocess.run(
            ["bash", str(BUILD_SH), *args], stdout=f, stderr=subprocess.STDOUT
        ).returncode
    lines = log.read_text(errors="replace").splitlines()
    if rc != 0:
        tail = " | ".join(lines[-5:])
        raise RuntimeError(f"build.sh {args[0]} rc={rc} ({log}): {tail}")
    return lines[-1]


def tonight(
    root: Path, lmcache_src: Path, out: Path, need_lmcache: bool
) -> dict[str, Any]:
    """Produce tonight's images; see the module docstring.

    Args:
        root: ``[host].root``; holds ``prt-images/pinned.json``.
        lmcache_src: Clean LMCache checkout at the commit under test.
        out: Dir for build logs; created.
        need_lmcache: Whether the suite runs an LMCache pair.

    Returns:
        ``usable``, ``time``, ``vllm_source`` (``pinned`` or ``nightly``),
        ``pinned_since`` when pinned, ``error`` when a build failed, and when
        usable ``serving`` and ``vllm_commit``, plus ``lmcache`` and
        ``lmcache_commit`` when an LMCache image was built.
    """
    pin_path = root / "prt-images" / "pinned.json"
    out.mkdir(parents=True)
    pinned = json.loads(pin_path.read_text()) if pin_path.exists() else None
    res: dict[str, Any] = {"usable": False, "time": time.time()}
    try:
        if pinned is not None and res["time"] - pinned["time"] < VLLM_REFRESH_S:
            res["vllm_source"], res["pinned_since"] = "pinned", pinned["time"]
            serving = pinned["serving"]
        else:
            res["vllm_source"] = "nightly"
            serving = build(["serving"], out / "build_serving.log")
        if need_lmcache:
            res["lmcache"] = build(
                ["lmcache", str(lmcache_src), serving], out / "build_lmcache.log"
            )
            res["lmcache_commit"] = docker.image_label(
                res["lmcache"], "prt.lmcache.commit"
            )
    except RuntimeError as e:
        res["error"] = str(e)
        return res
    res["usable"] = True
    res["serving"] = serving
    res["vllm_commit"] = docker.image_label(serving, "prt.vllm.commit")
    return res


def pin(root: Path, images: dict[str, Any]) -> None:
    """Pin tonight's serving image after a valid suite, if it was a new one.

    Args:
        root: ``[host].root``.
        images: :func:`tonight`'s result.
    """
    if images["vllm_source"] != "nightly":
        return
    pin_path = root / "prt-images" / "pinned.json"
    pin_path.parent.mkdir(parents=True, exist_ok=True)
    pin_path.write_text(
        json.dumps(
            {
                "serving": images["serving"],
                "vllm_commit": images["vllm_commit"],
                "time": images["time"],
            },
            indent=1,
        )
    )


def prune(keep: list[str]) -> list[str]:
    """Remove nightly images other than ``keep``; run after the night's runs.

    Args:
        keep: Image tags still needed: tonight's.

    Returns:
        The tags removed; a tag docker refuses to remove (in use) is kept.
    """
    tags = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    old = [t for t in tags if t.split(":")[0] in NIGHTLY_REPOS and t not in keep]
    # One at a time, so an image in use does not block the rest.
    return [
        t
        for t in old
        if subprocess.run(["docker", "rmi", t], capture_output=True).returncode == 0
    ]
