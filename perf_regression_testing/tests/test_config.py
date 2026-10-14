# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from perf_regression_testing.runner.config import (
    HostSetupError,
    check_host,
    load,
    resolve,
)


def test_load_fills_host_values_from_the_env(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(
        '[host]\n[model]\nrepo = "org/m"\nrevision = "abc"\n'
        '[a]\nx = "<root>/l2"\n'
        'mounts = [{ path = "<model_path>" }, { path = "<artifact_dir>" }]\n'
        'd = "<rdma_devices>"\nn = 3\n'
    )
    cfg = load(path)
    snap = f"{tmp_path}/hf/hub/models--org--m/snapshots/abc"
    assert cfg["a"]["x"] == f"{tmp_path}/l2"
    assert cfg["a"]["mounts"] == [{"path": snap}, {"path": "<artifact_dir>"}]
    assert cfg["a"]["d"] == "mlx5_0,mlx5_1"
    assert cfg["a"]["n"] == 3
    assert cfg["host"]["root"] == str(tmp_path)
    assert cfg["model"]["path"] == snap


def test_load_requires_prt_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRT_ROOT")
    with pytest.raises(HostSetupError, match="PRT_ROOT"):
        load()


def test_hf_cache_defaults_to_roots_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRT_HF_CACHE")
    assert load()["model"]["path"].startswith("/root/.cache/huggingface/hub/")


def test_shipped_config_has_no_unresolved_host_placeholder() -> None:
    text = repr(load())
    for p in ("<root>", "<hf_cache>", "<model_path>", "<model_", "<rdma_devices>"):
        assert p not in text


def test_check_host_needs_the_model_snapshot(tmp_path: Path) -> None:
    cfg = load()
    with pytest.raises(HostSetupError, match="hf download"):
        check_host(cfg, ["vllm_lmcache"])
    Path(cfg["model"]["path"]).mkdir(parents=True)
    (Path(cfg["model"]["path"]) / "config.json").write_text("{}")
    check_host(cfg, ["vllm_lmcache", "vllm_mooncake"])


def test_check_host_needs_rdma_devices_only_for_mooncake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRT_RDMA_DEVICES")
    cfg = load()
    Path(cfg["model"]["path"]).mkdir(parents=True)
    (Path(cfg["model"]["path"]) / "config.json").write_text("{}")
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
        assert s.scratch[0]["path"].endswith(f"/l2/c{i}")
