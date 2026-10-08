# SPDX-License-Identifier: Apache-2.0
"""Opt-in native vLLM P/D check against running engines and LMCache servers.

Set LMCACHE_PD_TEST_CONFIG to a JSON file containing ``prefill``, ``decode``,
``prefill_cache``, ``decode_cache`` HTTP URLs; ``model``; ``prompts`` and
matching ``expected_token_ids``; ``l1_tags``; and an ``evidence_dir``. Use
MultiConnector[NixlConnector, LMCacheMPConnector] on each engine and distinct
LMCache servers. Pools should be small enough to reach every configured L1.
"""

# Standard
from pathlib import Path
import json
import os
import re
import time
import urllib.request

# Third Party
import pytest

pytestmark = pytest.mark.no_shared_allocator


def _request(url: str, body: dict[str, object] | None = None) -> str:
    """Read an HTTP endpoint, or POST a JSON body, propagating HTTP failures."""
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode()


def _metric(text: str, name: str, tag: str | None = None) -> float:
    """Sum metric samples, optionally restricting them to an L1 tag."""
    total = 0.0
    for line in text.splitlines():
        match = re.fullmatch(re.escape(name) + r"(?:\{([^}]*)\})?\s+(\S+)", line)
        if match and (tag is None or f'l1_tag="{tag}"' in (match[1] or "")):
            total += float(match[2])
    return total


def _snapshot(cache: str, directory: Path, label: str) -> str:
    """Save status and metrics after stores and ownership locks have drained."""
    deadline = time.monotonic() + 30
    while True:
        status = json.loads(_request(cache + "/status"))
        sm = status["storage_manager"]
        drained = all(
            not s["read_locked_count"] and not s["staging_object_count"]
            for s in sm["l1_managers"].values()
        ) and all(
            not s["in_flight_task_count"] and not s["pending_keys_count"]
            for s in sm["store_controllers"].values()
        )
        if drained or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    metrics = _request(cache + "/metrics")
    (directory / f"{label}-status.json").write_text(json.dumps(status, indent=2))
    (directory / f"{label}-metrics.txt").write_text(metrics)
    assert status["is_healthy"] and drained, label
    assert sm["l1_usage"][0] == sum(
        s["memory_used_bytes"] for s in sm["l1_managers"].values()
    )
    assert sm["l1_usage"][1] == sum(
        s["memory_total_bytes"] for s in sm["l1_managers"].values()
    )
    for tag, state in sm["l1_managers"].items():
        assert (
            _metric(metrics, "lmcache_mp_l1_memory_usage_bytes", tag)
            == state["memory_used_bytes"]
        )
    return metrics


def test_native_vllm_pd_multi_l1() -> None:
    """Check native P/D transfers and independent LMCache reuse on both roles."""
    config_path = os.environ.get("LMCACHE_PD_TEST_CONFIG", "")
    if not config_path:
        pytest.skip("Set LMCACHE_PD_TEST_CONFIG for the running P/D deployment")
    config = json.loads(Path(config_path).read_text())
    directory = Path(config["evidence_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    prompts = config["prompts"]
    expected = config["expected_token_ids"]
    assert len(prompts) == len(expected) == len(config["l1_tags"])
    before = _request(config["decode"] + "/metrics")
    for phase in ("cold", "warm"):
        for i, (prompt, tokens) in enumerate(zip(prompts, expected, strict=True)):
            if phase == "warm":
                for role in ("prefill", "decode"):
                    result = json.loads(
                        _request(
                            config[role] + "/reset_prefix_cache?reset_external=false",
                            {},
                        )
                    )
                    assert result["success"]
            body = {
                "model": config["model"],
                "prompt": prompt,
                "temperature": 0,
                "seed": 42,
                "max_tokens": 1,
                "return_token_ids": True,
                "kv_transfer_params": {"do_remote_decode": True},
            }
            prefill = json.loads(_request(config["prefill"] + "/v1/completions", body))
            params = prefill["kv_transfer_params"]
            assert params["do_remote_prefill"] and any(params["remote_block_ids"])
            body.update(max_tokens=16, kv_transfer_params=params)
            decode = json.loads(_request(config["decode"] + "/v1/completions", body))
            (directory / f"{phase}-{i}.json").write_text(
                json.dumps({"prefill": prefill, "decode": decode}, indent=2)
            )
            assert decode["choices"][0]["token_ids"] == tokens
        for role in ("prefill", "decode"):
            _snapshot(config[role + "_cache"], directory, f"{phase}-{role}")

    deadline = time.monotonic() + 15
    while True:
        after = _request(config["decode"] + "/metrics")
        transferred = _metric(after, "vllm:nixl_bytes_transferred_sum") - _metric(
            before, "vllm:nixl_bytes_transferred_sum"
        )
        if transferred > 0 or time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    (directory / "decode-engine-metrics.txt").write_text(after)
    assert transferred > 0, "No measured native NIXL P/D transfer"

    # NIXL-imported prefixes are not automatically offloaded by the decoder.
    # Separately compute then reload each prefix through its LMCache connector.
    for phase in ("store", "reuse"):
        for i, (prompt, tokens) in enumerate(zip(prompts, expected, strict=True)):
            reset = json.loads(
                _request(
                    config["decode"] + "/reset_prefix_cache?reset_external=false", {}
                )
            )
            assert reset["success"]
            body = {
                "model": config["model"],
                "prompt": prompt,
                "temperature": 0,
                "seed": 42,
                "max_tokens": 16,
                "return_token_ids": True,
            }
            result = json.loads(_request(config["decode"] + "/v1/completions", body))
            (directory / f"decode-{phase}-{i}.json").write_text(
                json.dumps(result, indent=2)
            )
            assert result["choices"][0]["token_ids"] == tokens
        _snapshot(config["decode_cache"], directory, f"decode-{phase}")

    for role in ("prefill", "decode"):
        metrics = _snapshot(config[role + "_cache"], directory, f"final-{role}")
        assert _metric(metrics, "lmcache_mp_num_chunks_loaded_total") > 0
        for tag in config["l1_tags"]:
            assert _metric(metrics, "lmcache_mp_l1_write_chunks_total", tag) > 0
            assert _metric(metrics, "lmcache_mp_l1_read_chunks_total", tag) > 0
