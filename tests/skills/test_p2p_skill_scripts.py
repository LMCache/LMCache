# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure-Python helpers bundled with the lmcache-p2p-* skills.

The scripts live under .claude/skills/<skill>/scripts and are imported from there;
nothing here needs a GPU, a server or Prometheus.
"""

# Standard
import importlib.util
import json
import pathlib
import sys

# Third Party
import pytest

SKILLS = pathlib.Path(__file__).resolve().parents[2] / ".claude" / "skills"


def _load(skill: str, name: str):
    path = SKILLS / skill / "scripts" / f"{name}.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _no_env(monkeypatch, tmp_path):
    """Point the env loader at an empty directory so no real env.sh is picked up."""
    monkeypatch.delenv("P2P_ENV", raising=False)
    monkeypatch.setenv("P2P_HOME", str(tmp_path))


# --- lmcache-p2p-launch/gen_env.sh contract via p2penv -------------------------------


def test_p2penv_loads_generated_env(tmp_path):
    env_sh = tmp_path / "env.sh"
    env_sh.write_text(
        'export NODE_IP="10.0.0.2" PEER_IP="10.0.0.3" HTTP_PORT=7700 VLLM_PORT=8300\n'
        'export VLLM_URL="http://$NODE_IP:$VLLM_PORT" PEER_VLLM_URL="http://$PEER_IP:8301"\n'
        'export HTTP_URL="http://$NODE_IP:$HTTP_PORT" PEER_HTTP_URL="http://$PEER_IP:7701"\n'
    )
    p2penv = _load("lmcache-p2p-test", "p2penv")
    env = p2penv.load()
    assert p2penv.url_defaults(env) == (
        "http://10.0.0.2:8300",
        "http://10.0.0.3:8301",
        "http://10.0.0.2:7700",
        "http://10.0.0.3:7701",
    )


def test_p2penv_missing_file_is_empty():
    p2penv = _load("lmcache-p2p-test", "p2penv")
    assert p2penv.load() == {}
    assert p2penv.url_defaults({})[0] == "http://127.0.0.1:8300"


# --- lmcache-p2p-test/tokens_per_gb.py -----------------------------------------------

QWEN3_8B = {
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 4096,
    "torch_dtype": "bfloat16",
}


def test_tokens_per_gb_qwen3_8b():
    m = _load("lmcache-p2p-test", "tokens_per_gb")
    assert m.kv_bytes_per_token(QWEN3_8B) == 36 * 8 * 128 * 2 * 2
    assert m.tokens_per_gb(QWEN3_8B) == 6781


def test_tokens_per_gb_derives_head_dim_and_mha():
    m = _load("lmcache-p2p-test", "tokens_per_gb")
    cfg = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "hidden_size": 64,
        "torch_dtype": "float16",
    }
    assert m.kv_bytes_per_token(cfg) == 2 * 4 * 16 * 2 * 2


def test_tokens_per_gb_finds_cached_config(tmp_path):
    m = _load("lmcache-p2p-test", "tokens_per_gb")
    snap = tmp_path / "hub" / "models--Org--Model" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(QWEN3_8B))
    assert m.find_config("Org/Model", str(tmp_path)).endswith("abc/config.json")
    with pytest.raises(FileNotFoundError):
        m.find_config("Org/Other", str(tmp_path))


# --- lmcache-p2p-test/basic_test.py --------------------------------------------------


def test_basic_prompt_is_deterministic_and_sized():
    bt = _load("lmcache-p2p-test", "basic_test")
    assert bt.make_prompt(100, 7) == bt.make_prompt(100, 7)
    assert bt.make_prompt(100, 7) != bt.make_prompt(100, 8)
    assert len(bt.make_prompt(6000, 1).split()) >= 6000


def test_basic_evaluate_pass_and_fail():
    bt = _load("lmcache-p2p-test", "basic_test")
    cold = {"ttft_s": 0.2}
    peer_hit = {
        "ttft_s": 0.08,
        "prompt_tokens": 7032,
        "stats": {"num_lmcache_cached_tokens": 6912},
    }
    ok, msg = bt.evaluate(cold, peer_hit, "same", "same")
    assert ok and "PASS" in msg and "2.5x" in msg
    assert not bt.evaluate(cold, peer_hit, "same", "different")[0]
    peer_miss = {"ttft_s": 0.2, "prompt_tokens": 7032, "stats": None}
    ok, msg = bt.evaluate(cold, peer_miss, "same", "same")
    assert not ok and "p2p_cached_tokens=0/7032" in msg


# --- lmcache-p2p-test/stress_p2p.py --------------------------------------------------


def test_stress_summary_percentiles_and_hit_ratio():
    st = _load("lmcache-p2p-test", "stress_p2p")
    rows = [
        dict(error=None, ttft_s=t, total_s=t, cached_tokens=6912, prompt_tokens=7000)
        for t in (0.1, 0.2, 0.3, 0.4)
    ]
    rows.append(
        dict(
            error="http 500",
            ttft_s=None,
            total_s=1.0,
            cached_tokens=None,
            prompt_tokens=None,
        )
    )
    s = st.summarize_phase("peer", rows, wall=2.0, start_ts=100.0, end_ts=102.0)
    assert s["requests"] == 5 and s["errors"] == 1
    assert s["ttft_p50"] == 0.3 and s["ttft_max"] == 0.4
    assert s["lmcache_hit_ratio"] == round(6912 / 7000, 3)
    assert s["prompt_tok_per_s"] == 14000 and s["req_per_s"] == 2.0
    assert st.percentile([], 50) != st.percentile([], 50)  # nan


# --- lmcache-p2p-test/promql.py + export_prom.py + summarize.py ----------------------


def test_promql_parsers_drop_nan():
    pq = _load("lmcache-p2p-test", "promql")
    vec = {
        "data": {
            "result": [
                {"metric": {"node": "a"}, "value": [1, "2.5"]},
                {"metric": {"node": "b"}, "value": [1, "NaN"]},
            ]
        }
    }
    assert pq.parse_vector(vec) == {"a": 2.5}
    mat = {
        "data": {
            "result": [
                {
                    "metric": {"node": "a"},
                    "values": [[10, "1"], [15, "+Inf"], [20, "3"]],
                }
            ]
        }
    }
    assert pq.parse_matrix(mat) == {(10.0, "a"): 1.0, (20.0, "a"): 3.0}
    assert all("by (node)" in e for e, _ in pq.EXPRS.values())


def test_export_window_from_json(tmp_path):
    ex = _load("lmcache-p2p-test", "export_prom")
    f = tmp_path / "stress_x.json"
    f.write_text(
        json.dumps(
            {
                "phases": [
                    {"start_ts": 100, "end_ts": 110},
                    {"start_ts": 120, "end_ts": 130},
                ]
            }
        )
    )
    assert ex.window_from_json(str(f)) == (90, 140)


def test_summarize_tables(tmp_path):
    sm = _load("lmcache-p2p-test", "summarize")
    (tmp_path / "stress_s1.json").write_text(
        json.dumps(
            {
                "phases": [
                    {
                        "phase": "peer",
                        "requests": 4,
                        "errors": 0,
                        "ttft_p50": 0.1,
                        "ttft_p99": 0.2,
                        "prompt_tok_per_s": 1000,
                        "lmcache_hit_ratio": 0.98,
                    }
                ]
            }
        )
    )
    b = tmp_path / "bench_x" / "query_B"
    b.mkdir(parents=True)
    (b / "bench_summary.json").write_text(
        json.dumps(
            {
                "results": {
                    "total_requests": 76,
                    "failed_requests": 0,
                    "p50_ttft_ms": 729.4,
                    "p99_ttft_ms": 855.0,
                    "input_throughput": 342186.5,
                }
            }
        )
    )
    (tmp_path / "prom_s1.csv").write_text(
        "ts,node,load_GB/s,hit_l2_tok/s,pf_fail/s,l1_use%,ttft_mean_s\n1,n1,10,500,0,20,0.5\n2,n1,30,900,,25,0.4\n"
    )
    rows = sm.stress_rows(str(tmp_path / "stress_s1.json"))
    assert rows[0]["tag"] == "s1" and rows[0]["hit"] == 0.98
    assert sm.bench_rows(str(tmp_path / "bench_x"))[0]["ttft_p50_ms"] == 729
    peaks = sm.prom_peaks(str(tmp_path / "prom_s1.csv"))
    assert peaks[0]["load_GB_s"] == 30.0 and peaks[0]["hit_l2_tok_s"] == 900
    table = sm.md_table(rows)
    assert table.splitlines()[0].startswith("| tag | phase |") and sm.md_table([]) == ""
