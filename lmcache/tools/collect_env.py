# SPDX-License-Identifier: Apache-2.0
"""
``lmcache.tools.collect_env`` — automatic environment information collector.

Usage::

    python -m lmcache.tools.collect_env
    python -m lmcache.tools.collect_env --json

The script prints a Markdown block (or JSON object) with all relevant LMCache
runtime and dependency information, ready to paste into a GitHub issue.
"""

# Standard
import argparse
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path


def _get_version(mod: str) -> str:
    """Try to import a module and return its version string.

    First attempts ``importlib.import_module`` followed by a ``__version__``
    attribute lookup. Falls back to ``importlib.metadata.version()``.

    Args:
        mod: Dotted module name (e.g. ``"torch"``, ``"lmcache"``).

    Returns:
        The version string, or ``"N/A"`` on failure.
    """
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", None)
        if ver is None:
            # fallback: try pkg_resources / importlib.metadata
            import importlib.metadata
            return importlib.metadata.version(mod.split(".")[0])
        return str(ver)
    except Exception:
        return "N/A"


def _run_cmd(cmd: list[str], timeout: int = 10) -> str:
    """Run a command and return stdout stripped, or error message.

    Args:
        cmd: The command as a list of strings (passed to ``subprocess.run``).
        timeout: Maximum wall-clock time in seconds (default 10).

    Returns:
        The stripped stdout on success, or a descriptive string starting with
        ``"(not found)"``, ``"(timeout)"``, or ``"(error: ...)"`` on failure.
    """
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip()
    except FileNotFoundError:
        return "(not found)"
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(error: {e})"


def _get_git_commit(repo_path: Path | None = None) -> dict[str, str]:
    """Try to get git commit information.

    Args:
        repo_path: Path to the git repository. Defaults to the parent of
            ``lmcache/tools/`` (three levels up from this file).

    Returns:
        A dict with ``"branch"`` and ``"commit"`` keys. Both default to
        ``"N/A"`` if the information cannot be obtained.
    """
    result: dict[str, str] = {"branch": "N/A", "commit": "N/A"}
    try:
        if repo_path is None:
            repo_path = Path(__file__).resolve().parent.parent.parent  # lmcache root
        branch = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if branch.returncode == 0:
            result["branch"] = branch.stdout.strip()

        commit = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if commit.returncode == 0:
            result["commit"] = commit.stdout.strip()
    except Exception:
        pass
    return result


def _get_gpu_info() -> list[dict[str, str]]:
    """Collect GPU info via nvidia-smi.

    Falls back to PyTorch's device API when ``nvidia-smi`` is unavailable.

    Returns:
        A list of GPU dicts, each containing ``"index"``, ``"name"``,
        ``"memory_mib"``, and ``"compute_cap"``. Returns an empty list when
        no GPU is detected.
    """
    gpus: list[dict[str, str]] = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15
        )
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "index": parts[0],
                        "name": parts[1],
                        "memory_mib": parts[2],
                        "compute_cap": parts[3],
                    })
    except Exception:
        pass
    if not gpus:
        # fallback: try torch
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "index": str(i),
                        "name": props.name,
                        "memory_mib": str(props.total_memory // (1024 * 1024)),
                        "compute_cap": f"{props.major}.{props.minor}",
                    })
        except Exception:
            pass
    return gpus


def _get_cuda_info() -> dict[str, str]:
    """Collect CUDA runtime and driver version.

    Returns:
        A dict with ``"driver"`` and ``"runtime"`` keys.
        Both default to ``"N/A"`` if the information cannot be obtained.
    """
    info: dict[str, str] = {"driver": "N/A", "runtime": "N/A"}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0 and out.stdout.strip():
            info["driver"] = out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            info["runtime"] = torch.version.cuda or "N/A"
    except Exception:
        pass
    return info


def _get_package_info() -> dict[str, str]:
    """Get versions of key packages used by LMCache.

    Returns:
        A dict mapping package names (``"lmcache"``, ``"torch"``, ``"vllm"``,
        ``"sglang"``, ``"transformers"``, ``"numpy"``) to their version
        strings. Missing packages show ``"N/A"``.
    """
    return {
        "lmcache": _get_version("lmcache"),
        "torch": _get_version("torch"),
        "vllm": _get_version("vllm"),
        "sglang": _get_version("sglang"),
        "transformers": _get_version("transformers"),
        "numpy": _get_version("numpy"),
    }


def _get_config_content() -> str:
    """Read LMCache config file if ``LMCACHE_CONFIG_FILE`` env var is set.

    Returns:
        The config content inside a YAML fenced code block, or a short status
        string (``"(not set)"`` or ``"(error reading config: ...)"``).
    """
    config_path = os.environ.get("LMCACHE_CONFIG_FILE", "")
    if not config_path:
        return "(not set)"
    try:
        content = Path(config_path).read_text(encoding="utf-8")
        return f"```yaml\n{content.rstrip()}\n```"
    except Exception as e:
        return f"(error reading config: {e})"


def collect() -> dict[str, object]:
    """Collect all environment information into a dictionary.

    The returned dict contains the following top-level keys:

    - ``lmcache_version``
    - ``git_branch`` / ``git_commit``
    - ``python_version`` (full ``sys.version`` string)
    - ``os`` (dict with ``system`` / ``release`` / ``version``)
    - ``packages`` (dict of dependency versions)
    - ``cuda`` (dict with ``driver`` / ``runtime``)
    - ``gpus`` (list of GPU info dicts)
    - ``config_file`` / ``config_content``

    Returns:
        A dictionary suitable for JSON serialisation or Markdown formatting.
    """
    git_info = _get_git_commit()
    gpus = _get_gpu_info()
    cuda = _get_cuda_info()
    packages = _get_package_info()

    info: dict[str, object] = {
        "lmcache_version": packages["lmcache"],
        "git_branch": git_info["branch"],
        "git_commit": git_info["commit"],
        "python_version": sys.version,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "packages": packages,
        "cuda": cuda,
        "gpus": gpus,
        "config_file": os.environ.get("LMCACHE_CONFIG_FILE", "(not set)"),
        "config_content": _get_config_content(),
    }
    return info


def _format_collected(info: dict) -> str:
    """Format the collected info as a GitHub-ready Markdown block.

    Args:
        info: The dictionary returned by :func:`collect`.

    Returns:
        A GitHub-flavoured Markdown string with environment details.
    """
    lines = []
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- **LMCache version**: `{info['lmcache_version']}`")
    lines.append(f"- **Git branch**: `{info['git_branch']}`")
    lines.append(f"- **Git commit**: `{info['git_commit']}`")
    lines.append(f"- **Python**: `{info['python_version'].replace(chr(10), ' ')}`")

    os_info = info["os"]
    lines.append(f"- **OS**: {os_info['system']} {os_info['release']}")
    lines.append(f"- **OS version**: `{os_info['version']}`")
    lines.append("")
    lines.append("### Packages")
    lines.append("")
    for name, ver in info["packages"].items():
        lines.append(f"- `{name}`: `{ver}`")
    lines.append("")
    lines.append(f"- **CUDA Driver**: `{info['cuda']['driver']}`")
    lines.append(f"- **CUDA Runtime**: `{info['cuda']['runtime']}`")
    lines.append("")
    if info["gpus"]:
        lines.append("### GPUs")
        lines.append("")
        for gpu in info["gpus"]:
            lines.append(f"- **GPU {gpu['index']}**: {gpu['name']}")
            lines.append(f"  - Memory: {gpu['memory_mib']} MiB")
            lines.append(f"  - Compute Capability: {gpu['compute_cap']}")
            lines.append("")
    else:
        lines.append("- **GPU**: (none detected)")
        lines.append("")
    lines.append("### Config")
    lines.append("")
    lines.append(f"- **LMCACHE_CONFIG_FILE**: `{info['config_file']}`")
    lines.append("")
    lines.append(info["config_content"])
    return "\n".join(lines)


def main() -> None:
    """Entry point for ``python -m lmcache.tools.collect_env``.

    Parses ``--json`` to switch between Markdown and JSON output, then
    collects and prints the environment information.
    """
    parser = argparse.ArgumentParser(
        description="Collect LMCache environment information for bug reports."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of Markdown.",
    )
    args = parser.parse_args()

    info = collect()
    if args.json:
        import json
        json.dump(info, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(_format_collected(info))


if __name__ == "__main__":
    main()
