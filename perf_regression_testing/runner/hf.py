# SPDX-License-Identifier: Apache-2.0
"""Download a model snapshot into the HF cache that runs read offline.

The download runs in the serving image, which ships ``huggingface_hub``,
so the host needs nothing beyond docker.
"""

# Standard
from pathlib import Path
from typing import Any
import logging
import shutil
import subprocess
import time

# First Party
from perf_regression_testing.runner import docker
from perf_regression_testing.runner.run import Cancelled

logger = logging.getLogger(__name__)

# Refuses to start when the cache's file system cannot hold what is left to
# fetch: the model's size less the blobs a previous attempt left.
_FETCH = """
import pathlib, shutil, sys
from huggingface_hub import HfApi, snapshot_download
repo, rev, cache = sys.argv[1:]
info = HfApi().model_info(repo, revision=rev, files_metadata=True)
blobs = pathlib.Path(cache, "hub", "models--" + repo.replace("/", "--"), "blobs")
have = sum(f.stat().st_size for f in blobs.glob("*")) if blobs.is_dir() else 0
need = sum(f.size or 0 for f in info.siblings) - have
free = shutil.disk_usage(cache).free
if free < need:
    sys.exit(f"{cache}: {free / 1e9:.0f} GB free, {need / 1e9:.0f} GB more needed")
snapshot_download(repo, revision=rev)
"""


def fetch(
    model: dict[str, Any],
    hf_cache: str,
    image: str,
    container: dict[str, Any],
    log: Path,
) -> None:
    """Download ``model`` at its revision into ``hf_cache``, then write its
    ``done`` marker.

    An interrupted download resumes on the next call.

    Args:
        model: A ``[models.<name>]`` table (``repo``, ``revision``, ``done``).
        hf_cache: The ``HF_HOME`` that runs mount.
        image: An image with ``huggingface_hub``: the serving image.
        container: The ``[container]`` table.
        log: File that receives the download output.

    Raises:
        RuntimeError: the download failed or the disk is too small; the
            message names the log.
    """
    name = f"prt-fetch-{time.strftime('%Y%m%d-%H%M%S')}"
    args = docker.run_args(
        name,
        name,
        image,
        container,
        mounts=[{"path": hf_cache}],
        env={"HF_HOME": hf_cache},
        gpus=None,
        devices=[],
        cap_add=[],
    )
    args += ["python3", "-c", _FETCH, model["repo"], model["revision"], hf_cache]
    logger.info("downloading %s (log: %s)", model["repo"], log)
    t0 = time.monotonic()
    stop_s = docker.STOP_S
    try:
        with open(log, "w") as f:
            rc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT).returncode
    except Cancelled:
        stop_s = 0
        raise
    finally:
        docker.remove([name], stop_s)
    if rc != 0:
        tail = " | ".join(log.read_text(errors="replace").splitlines()[-3:])
        raise RuntimeError(f"download of {model['repo']} rc={rc} ({log}): {tail}")
    Path(model["done"]).parent.mkdir(parents=True, exist_ok=True)
    Path(model["done"]).touch()
    logger.info("downloaded %s in %.0f s", model["repo"], time.monotonic() - t0)


def prune(hf_cache: str, root: str, models: list[dict[str, Any]]) -> list[str]:
    """Delete the snapshots and done markers of models no longer in ``[models]``.

    Only a cache inside PRT's root (symlinks resolved) is pruned: the default
    ``HF_HOME`` may hold other users' models. Weekly-only models are in
    ``[models]`` too, so a nightly keeps them. :func:`fetch` runs as the host
    user, so the host can delete what it downloaded.

    Args:
        hf_cache: The ``HF_HOME`` that runs mount.
        root: PRT's root (``[host].root``).
        models: Every ``[models]`` table.

    Returns:
        The repos removed, as ``models--<org>--<name>`` dirs.

    Raises:
        OSError: a snapshot could not be deleted, e.g. one downloaded as root.
    """
    if not Path(hf_cache).resolve().is_relative_to(Path(root).resolve()):
        return []
    markers = {Path(m["done"]).name for m in models}
    for done in Path(root, "state").glob("model-*.done"):
        if done.name not in markers:
            done.unlink()
    hub = Path(hf_cache, "hub")
    keep = {"models--" + m["repo"].replace("/", "--") for m in models}
    old = sorted(d.name for d in hub.glob("models--*") if d.name not in keep)
    for d in old:
        shutil.rmtree(hub / d)
        shutil.rmtree(hub / ".locks" / d, ignore_errors=True)
    if old:
        logger.info("removed models no longer configured: %s", ", ".join(old))
    return old
