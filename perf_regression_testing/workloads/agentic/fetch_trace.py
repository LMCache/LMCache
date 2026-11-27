# SPDX-License-Identifier: Apache-2.0
"""Build an agentic trace directory, or print its digest.

Each trace is an HF dataset at a pinned revision: one ``traces.jsonl``, one
session tree per line. AIPerf's ``weka_trace`` loader reads a directory with
one ``<id>.json`` per tree, so this script splits the file by ``id``, writing
each line unchanged. The directory's name picks the dataset (``TRACES``):
the same 393 sessions with requests capped at 256k tokens, or uncapped (up to
990,016) for models with a 1M context.

Usage::

    fetch_trace.py build OUT_DIR    # download (HF cache), verify, split
    fetch_trace.py digest TRACE_DIR # print dir_digest (no download)

``build`` verifies ``traces.jsonl`` against the pinned sha256 first.
``digest`` is the byte-level check the runner does before every run without
network or HF cache; config.toml pins its value for the directory ``build``
writes.
"""

# Standard
import hashlib
import json
import os
import shutil
import sys

# Directory name -> (HF dataset, revision, sha256 of traces.jsonl).
TRACES = {
    "weka-062126-256k-dir": (
        "semianalysisai/cc-traces-weka-062126-256k",
        "8fecd2fc56694469f758f0afbbb6335ad3043740",
        "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef",
    ),
    "weka-062126-dir": (
        "semianalysisai/cc-traces-weka-062126",
        "23f152f6f0f9399a85901b89a6458def0ef16729",
        "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276",
    ),
}


def dir_digest(trace_dir: str) -> str:
    """Return sha256 over the sorted (file name, file sha256) pairs of a dir.

    Args:
        trace_dir: Directory of trace files.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    for name in sorted(os.listdir(trace_dir)):
        with open(os.path.join(trace_dir, name), "rb") as f:
            h.update(f"{name}\0{hashlib.sha256(f.read()).hexdigest()}\n".encode())
    return h.hexdigest()


def load_lines(name: str) -> list[str]:
    """Download the pinned ``traces.jsonl`` of one trace and return its lines.

    Args:
        name: A key of ``TRACES``.

    Returns:
        One JSON string per session tree.

    Raises:
        SystemExit: when the file's sha256 differs from the pin.
    """
    # Imported here so `digest` (used by the runner) needs only the stdlib.
    # Third Party
    from huggingface_hub import hf_hub_download

    repo, revision, sha256 = TRACES[name]
    path = hf_hub_download(repo, "traces.jsonl", repo_type="dataset", revision=revision)
    with open(path, "rb") as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        sys.exit(f"traces.jsonl sha256 {digest} != pinned {sha256}")
    return data.decode().splitlines()


def build(out_dir: str) -> None:
    """Write one ``<id>.json`` per tree into a new directory ``out_dir``.

    The files go to ``<out_dir>.partial`` first, renamed when complete, so an
    interrupted build never leaves a directory that looks finished.

    Args:
        out_dir: Directory to create; must not exist. Its name is a key of
            ``TRACES``.

    Raises:
        SystemExit: when the name is not in ``TRACES``.
    """
    name = os.path.basename(os.path.normpath(out_dir))
    if name not in TRACES:
        sys.exit(f"{name}: not one of {', '.join(TRACES)}")
    lines = load_lines(name)
    tmp = out_dir.rstrip("/") + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for line in lines:
        with open(os.path.join(tmp, json.loads(line)["id"] + ".json"), "w") as f:
            f.write(line)
    os.rename(tmp, out_dir)
    print(f"built {out_dir}: {len(lines)} trees")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("build", "digest"):
        sys.exit(__doc__)
    if sys.argv[1] == "digest":
        print(dir_digest(sys.argv[2]))
    else:
        build(sys.argv[2])
