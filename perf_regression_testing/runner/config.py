# SPDX-License-Identifier: Apache-2.0
"""Resolve config.toml into the concrete settings of one run.

:func:`resolve` merges the shared base, the deployment's model, the deployment
and the workload, later layers overriding earlier ones key by key. A service
with ``replicas = N`` expands into N services: ``<i>`` becomes ``i`` and
``<B+Si>`` becomes ``B + S * i``; a string that is only a placeholder becomes
an integer.
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import re
import tomllib

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"
PKG_DIR = CONFIG_PATH.parent
DEFAULT_HF_CACHE = "/root/.cache/huggingface"
_REPLICA = re.compile(r"<(?:(\d+)\+)?(\d*)i>")


class HostSetupError(Exception):
    """The host does not provide what the config needs (env var, model)."""


def substitute(value: Any, subst: dict[str, str]) -> Any:
    """Replace each key of ``subst`` in every string inside ``value``.

    Args:
        value: A string, or a dict or list holding strings at any depth;
            other values are returned unchanged.
        subst: Placeholder -> replacement.

    Returns:
        A copy of ``value`` with every placeholder replaced.
    """
    if isinstance(value, str):
        for key, new in subst.items():
            value = value.replace(key, new)
        return value
    if isinstance(value, dict):
        return {k: substitute(v, subst) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, subst) for v in value]
    return value


def _for_replica(value: Any, i: int) -> Any:
    """Resolve the replica placeholders in every string inside ``value``.

    Args:
        value: A config value: string, dict or list, nested freely.
        i: The replica index.

    Returns:
        A copy of ``value``; a string that is only a placeholder becomes an int.
    """

    def num(m: re.Match[str]) -> int:
        return int(m.group(1) or 0) + int(m.group(2) or 1) * i

    if isinstance(value, str):
        whole = _REPLICA.fullmatch(value)
        if whole:
            return num(whole)
        return _REPLICA.sub(lambda m: str(num(m)), value)
    if isinstance(value, dict):
        return {k: _for_replica(v, i) for k, v in value.items()}
    if isinstance(value, list):
        return [_for_replica(v, i) for v in value]
    return value


def load(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the config file and fill in the host-specific placeholders.

    * ``<root>``: ``$PRT_ROOT`` (required), also set as ``[host].root``.
    * ``<hf_cache>``: ``$PRT_HF_CACHE``, default :data:`DEFAULT_HF_CACHE`,
      also set as ``[host].hf_cache``.
    * ``<rdma_devices>``: ``$PRT_RDMA_DEVICES``, left in place when unset.

    Each ``[models.<name>]`` gets ``path``, its snapshot in that cache, and
    ``done``, a marker under ``<root>/state`` written once the snapshot is
    complete (a partial snapshot already holds ``config.json``). The
    ``<model_*>`` placeholders depend on the run's model; :func:`resolve`
    fills them.

    Args:
        path: TOML file to read.

    Returns:
        The parsed config.

    Raises:
        HostSetupError: ``PRT_ROOT`` is unset or empty.
    """
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    root = os.environ.get("PRT_ROOT")
    if not root:
        raise HostSetupError("PRT_ROOT is unset: set it to PRT's state dir")
    hf_cache = os.environ.get("PRT_HF_CACHE") or DEFAULT_HF_CACHE
    subst = {"<root>": root, "<hf_cache>": hf_cache}
    if os.environ.get("PRT_RDMA_DEVICES"):
        subst["<rdma_devices>"] = os.environ["PRT_RDMA_DEVICES"]
    cfg = substitute(cfg, subst)
    cfg["host"]["root"] = root
    cfg["host"]["hf_cache"] = hf_cache
    for m in cfg["models"].values():
        m["path"] = (
            f"{hf_cache}/hub/models--{m['repo'].replace('/', '--')}"
            f"/snapshots/{m['revision']}"
        )
        m["done"] = f"{root}/state/model-{m['revision']}.done"
    return cfg


def check_host(cfg: dict[str, Any], deployments: list[str]) -> None:
    """Check that the host provides what ``deployments`` need.

    Args:
        cfg: Parsed config from :func:`load`.
        deployments: Keys under ``[deployments]`` about to run.

    Raises:
        HostSetupError: A deployment uses ``<rdma_devices>`` while
            ``PRT_RDMA_DEVICES`` is unset.
    """
    for d in deployments:
        if "<rdma_devices>" in json.dumps(cfg["deployments"][d]):
            raise HostSetupError(f"{d} needs PRT_RDMA_DEVICES, which is unset")


def missing_models(cfg: dict[str, Any], deployments: list[str]) -> list[str]:
    """Return the models of ``deployments`` with no complete snapshot.

    Args:
        cfg: Parsed config from :func:`load`.
        deployments: Keys under ``[deployments]`` about to run.

    Returns:
        Keys under ``[models]``, each once, in deployment order.
    """
    names = dict.fromkeys(cfg["deployments"][d]["model"] for d in deployments)
    return [n for n in names if not Path(cfg["models"][n]["done"]).is_file()]


def display(cfg: dict[str, Any], deployment: str, gated: bool) -> dict[str, Any]:
    """Return what the results page shows for a run of ``deployment``.

    Args:
        cfg: Parsed config from :func:`load`.
        deployment: Key under ``[deployments]``.
        gated: Whether the run's suite fails on a regression.

    Returns:
        The deployment's ``label`` and ``desc``, the model's repo name, and
        ``gated``.
    """
    d = cfg["deployments"][deployment]
    return {
        "label": d["label"],
        "desc": d["desc"],
        "model": cfg["models"][d["model"]]["repo"].split("/")[-1],
        "gated": gated,
    }


def cli_args(args: dict[str, Any]) -> list[str]:
    """Turn an args table into command-line arguments.

    ``True`` is a bare flag, a list follows one flag, a table is one JSON value.

    Args:
        args: Mapping of option name (without ``--``) to value.

    Returns:
        The argument list, in the table's order.
    """
    out: list[str] = []
    for key, value in args.items():
        out.append(f"--{key}")
        if value is True:
            continue
        if isinstance(value, list):
            out += [str(v) for v in value]
        elif isinstance(value, dict):
            out.append(json.dumps(value))
        else:
            out.append(str(value))
    return out


@dataclass
class Service:
    """A cache-layer container started before vLLM."""

    name: str
    command: list[str]
    args: dict[str, Any]
    env: dict[str, str]
    ready_url: str
    ready_timeout_s: float
    scratch: list[dict[str, Any]]
    # docker --security-opt seccomp= value, or None for docker's default.
    seccomp: str | None


@dataclass
class RunSpec:
    """Everything one (workload, deployment) run needs, with layers merged."""

    workload: str
    deployment: str
    image_target: str
    gpu_count: int
    gpu_busy_wait_s: float
    container: dict[str, Any]
    model_path: str
    served_name: str
    vllm_port: int
    vllm_startup_timeout_s: float
    vllm_env: dict[str, str]
    vllm_args: dict[str, Any]
    services: list[Service]
    vllm_scratch: list[dict[str, Any]]
    vllm_memory_gb: int | None
    vllm_shm_gb: int | None
    devices: list[str]
    cap_add: list[str]
    files: dict[str, dict[str, Any]]
    aiperf_env: dict[str, str]
    aiperf_args: dict[str, Any]
    aiperf_timeout_s: float
    trace: dict[str, Any]
    trace_dir: Path
    window_s: float
    invalid_log_patterns: list[str]


def resolve(cfg: dict[str, Any], workload: str, deployment: str) -> RunSpec:
    """Merge the config layers for one (workload, deployment) pair.

    ``<model_path>``, ``<model_repo>`` and ``<model_revision>`` in the vLLM
    and AIPerf settings become the deployment's model's snapshot path, repo
    and revision.

    Args:
        cfg: Parsed config from :func:`load`.
        workload: Key under ``[workloads]``.
        deployment: Key under ``[deployments]``.

    Returns:
        The merged run settings.

    Raises:
        KeyError: if ``workload``, ``deployment`` or its model is not in the
            config.
    """
    w = cfg["workloads"][workload]
    d = cfg["deployments"][deployment]
    m = cfg["models"][d["model"]]
    msubst = {
        "<model_path>": m["path"],
        "<model_repo>": m["repo"],
        "<model_revision>": m["revision"],
    }
    host = cfg["host"]
    templates = d.get("services", [])
    raw_services = []
    for s in templates:
        if "replicas" in s:
            one = {k: v for k, v in s.items() if k != "replicas"}
            raw_services += [_for_replica(one, i) for i in range(s["replicas"])]
        else:
            raw_services.append(s)
    services = [
        Service(
            name=s["name"],
            command=s["command"],
            args=s.get("args", {}),
            env=s.get("env", {}),
            ready_url=s["ready_url"],
            ready_timeout_s=s["ready_timeout_s"],
            scratch=s.get("scratch", []),
            seccomp=s.get("seccomp"),
        )
        for s in raw_services
    ]
    trace = w["trace"]
    trace_dir = Path(host["root"]) / trace["dir"]
    aiperf_args = substitute(
        {
            **cfg["aiperf"]["args"],
            "model": m["served_name"],
            **m.get("aiperf_args", {}),
            **w["aiperf"].get("args", {}),
        },
        msubst,
    )
    aiperf_args["input-file"] = str(trace_dir)
    return RunSpec(
        workload=workload,
        deployment=deployment,
        image_target=d["image_target"],
        gpu_count=m.get("gpu_count", host["gpu_count"]),
        gpu_busy_wait_s=host["gpu_busy_wait_s"],
        container=cfg["container"],
        model_path=m["path"],
        served_name=m["served_name"],
        vllm_port=cfg["vllm"]["port"],
        vllm_startup_timeout_s=cfg["vllm"]["startup_timeout_s"],
        vllm_env={**cfg["vllm"]["env"], **d.get("vllm_env", {})},
        vllm_args=substitute(
            {
                **cfg["vllm"]["args"],
                **m.get("vllm_args", {}),
                **d.get("vllm_args", {}),
                **w.get("vllm_args", {}),
            },
            msubst,
        ),
        services=services,
        vllm_scratch=d.get("vllm_scratch", []),
        vllm_memory_gb=d.get("vllm_memory_gb"),
        vllm_shm_gb=d.get("vllm_shm_gb"),
        devices=d.get("devices", []),
        cap_add=d.get("cap_add", []),
        files=d.get("files", {}),
        aiperf_env=substitute(
            {**cfg["aiperf"]["env"], **w["aiperf"].get("env", {})}, msubst
        ),
        aiperf_args=aiperf_args,
        aiperf_timeout_s=cfg["aiperf"]["timeout_s"],
        trace=trace,
        trace_dir=trace_dir,
        window_s=w["scoring"]["window_s"],
        invalid_log_patterns=cfg["validity"]["invalid_log_patterns"],
    )
