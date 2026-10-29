# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import Any

# Third Party
import pytest

# First Party
from perf_regression_testing.runner.config import (
    HostSetupError,
    check_host,
    load,
    missing_models,
    resolve,
)


def test_load_fills_host_values_from_the_env(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(
        '[host]\n[models.m]\nrepo = "org/m"\nrevision = "abc"\n'
        '[a]\nx = "<root>/l2"\n'
        'mounts = [{ path = "<hf_cache>" }, { path = "<artifact_dir>" }]\n'
        'd = "<rdma_devices>"\nn = 3\n'
    )
    cfg = load(path)
    assert cfg["a"]["x"] == f"{tmp_path}/l2"
    assert cfg["a"]["mounts"] == [
        {"path": f"{tmp_path}/hf"},
        {"path": "<artifact_dir>"},
    ]
    assert cfg["a"]["d"] == "mlx5_0,mlx5_1"
    assert cfg["a"]["n"] == 3
    assert cfg["host"]["root"] == str(tmp_path)
    snap = f"{tmp_path}/hf/hub/models--org--m/snapshots/abc"
    assert cfg["models"]["m"]["path"] == snap


def test_load_requires_prt_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRT_ROOT")
    with pytest.raises(HostSetupError, match="PRT_ROOT"):
        load()


def test_hf_cache_defaults_to_roots_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRT_HF_CACHE")
    for m in load()["models"].values():
        assert m["path"].startswith("/root/.cache/huggingface/hub/")


def test_shipped_config_has_no_unresolved_host_placeholder() -> None:
    cfg = load()
    text = repr(cfg)
    for p in ("<root>", "<hf_cache>", "<rdma_devices>"):
        assert p not in text
    for w in cfg["workloads"]:
        for d in cfg["deployments"]:
            assert "<model_" not in repr(resolve(cfg, w, d))


def _fetch(cfg: dict[str, Any], model: str) -> None:
    done = Path(cfg["models"][model]["done"])
    done.parent.mkdir(parents=True, exist_ok=True)
    done.touch()


def test_missing_models_names_each_absent_model_once() -> None:
    cfg = load()
    deps = ["vllm_lmcache", "vllm_lmcache_ssm", "vllm_native"]
    assert missing_models(cfg, deps) == ["dsv41", "qwen38"]
    _fetch(cfg, "dsv41")
    assert missing_models(cfg, deps) == ["qwen38"]
    _fetch(cfg, "qwen38")
    assert missing_models(cfg, deps) == []


def test_a_snapshot_without_its_done_marker_is_still_missing() -> None:
    # An interrupted download leaves config.json but not every weight.
    cfg = load()
    path = Path(cfg["models"]["qwen38"]["path"])
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    assert missing_models(cfg, ["vllm_lmcache_ssm"]) == ["qwen38"]


def test_a_deployment_runs_its_own_model() -> None:
    cfg = load()
    for d, model, served in [
        ("vllm_lmcache", "dsv41", "deepseek-v4.1-flash"),
        ("vllm_lmcache_ssm", "qwen38", "qwen3p8"),
        ("vllm_lmcache_swa_l2", "trinity", "trinity-large-thinking"),
    ]:
        spec = resolve(cfg, "agentic", d)
        m = cfg["models"][model]
        assert spec.model_path == m["path"]
        assert spec.served_name == spec.aiperf_args["model"] == served
        assert spec.aiperf_args["tokenizer"] == m["repo"]
        assert spec.aiperf_args["tokenizer-revision"] == m["revision"]
        assert spec.aiperf_env["PRT_LIVE_HISTORY_TOKENIZER"] == m["path"]
        if "speculative-config" in m["vllm_args"]:
            assert spec.vllm_args["speculative-config"] == (
                m["vllm_args"]["speculative-config"].replace("<model_path>", m["path"])
            )


def test_a_model_s_gpu_count_replaces_the_host_s_and_matches_its_tp() -> None:
    cfg = load()
    assert resolve(cfg, "agentic", "vllm_lmcache_ssm").gpu_count == 4
    assert resolve(cfg, "agentic", "vllm_lmcache").gpu_count == 8
    for d in cfg["deployments"]:
        spec = resolve(cfg, "agentic", d)
        assert spec.vllm_args["tensor-parallel-size"] == spec.gpu_count, d


def test_a_model_with_its_own_trace_replays_it_in_place_of_the_workloads() -> None:
    cfg = load()
    root = Path(cfg["host"]["root"])
    v41 = resolve(cfg, "agentic", "vllm_native")
    assert v41.trace_dir == root / "work" / "weka-062126-dir"
    assert v41.aiperf_args["input-file"] == str(v41.trace_dir)
    assert v41.trace == cfg["models"]["dsv41"]["trace"]
    ssm = resolve(cfg, "agentic", "vllm_lmcache_ssm")
    assert ssm.trace == cfg["workloads"]["agentic"]["trace"]
    assert ssm.trace_dir.name == "weka-062126-256k-dir"


def test_aiperf_keeps_every_session_of_the_uncapped_trace() -> None:
    # AIPerf drops each session whose peak in + out exceeds max-context-length;
    # the longest one in the uncapped trace peaks at 996,579 tokens.
    v41 = resolve(load(), "agentic", "vllm_lmcache")
    assert v41.aiperf_args["max-context-length"] >= 996_579


def test_lmcache_read_locks_outlast_the_queue_wait() -> None:
    # Build 55 waited up to 313 s from lookup to retrieve; locks that expire
    # first fail the retrieve and kill vLLM.
    v41 = resolve(load(), "agentic", "vllm_lmcache")
    assert int(v41.services[0].args.get("l1-read-ttl-seconds", 300)) > 313


def test_check_host_needs_rdma_devices_only_for_mooncake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRT_RDMA_DEVICES")
    cfg = load()
    _fetch(cfg, "trinity")
    check_host(cfg, ["vllm_lmcache", "vllm_native"])
    with pytest.raises(HostSetupError, match="PRT_RDMA_DEVICES"):
        check_host(cfg, ["vllm_mooncake"])


def test_a_replicated_service_expands_per_index() -> None:
    spec = resolve(load(), "agentic", "vllm_mooncake")
    clients = [s for s in spec.services if s.name.startswith("mooncake_client")]
    assert [s.name for s in clients] == [f"mooncake_client{i}" for i in range(4)]
    for i, s in enumerate(clients):
        assert s.args["port"] == 29052 + 10 * i
        assert s.args["host"] == f"127.0.0.1:{29053 + 10 * i}"
        assert s.args["http_port"] == 19301 + i
        assert s.ready_url == f"http://127.0.0.1:{19301 + i}/health"
