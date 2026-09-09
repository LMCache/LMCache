# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache query coordinator``.

The renderers are pure functions of a parsed reply, so they are tested
against reply shapes directly. The table rendering itself is tested through
the formatter, since that is what "formalized output" means here: one report
object, two renderings.
"""

# Standard
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.query._coordinator import APIS, get_text, normalize_url
from lmcache.cli.commands.query.coordinator_command import CoordinatorQueryCommand
from lmcache.cli.metrics import Metrics, get_formatter
from lmcache.cli.metrics.section import sections_to_dict

GIB = 1 << 30


def _render(api: str, body: object) -> Metrics:
    """Run one API's renderer over ``body``."""
    metrics = Metrics(f"Coordinator: {api}")
    APIS[api].render(body, metrics)
    return metrics


def _terminal(metrics: Metrics) -> str:
    return get_formatter("terminal").format(metrics._title, metrics._sections)


def _json(metrics: Metrics) -> dict:
    return sections_to_dict(metrics._title, metrics._sections)


def _args(**overrides: object) -> argparse.Namespace:
    """Parsed arguments with every option defaulted."""
    defaults: dict[str, object] = {
        "api": "usage",
        "url": "http://127.0.0.1:9300",
        "instance": None,
        "cache_salt": None,
        "request_id": None,
        "limit": 20,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


_FLEET = {
    "instances": [
        {
            "instance_id": "mp-1",
            "registered": True,
            "declared_capacity": True,
            "modules": [
                {
                    "tier": "l1",
                    "backend": "dram",
                    "shared": False,
                    "used_bytes": 48 * GIB,
                    "capacity_bytes": 64 * GIB,
                    "usage_ratio": 0.75,
                },
                {
                    "tier": "l2",
                    "backend": "fs",
                    "shared": False,
                    "used_bytes": 12 * GIB,
                    "capacity_bytes": 0,
                    "usage_ratio": None,
                },
            ],
        },
        {
            "instance_id": "mp-2",
            "registered": True,
            "declared_capacity": True,
            "modules": [
                {
                    "tier": "l1",
                    "backend": "dram",
                    "shared": False,
                    "used_bytes": 2 * GIB,
                    "capacity_bytes": 64 * GIB,
                    "usage_ratio": 0.03125,
                }
            ],
        },
    ],
    "shared_modules": [
        {
            "tier": "l2",
            "backend": "s3",
            "shared": True,
            "used_bytes": 7 * GIB,
            "capacity_bytes": 0,
            "usage_ratio": None,
        }
    ],
}


class TestUsage:
    def test_rows_are_ordered_busiest_first(self) -> None:
        # The point of the table: which server to avoid, at a glance.
        rendered = _json(_render("usage", _FLEET))["metrics"]["usage"]
        assert [row["ratio"] for row in rendered] == [
            "75.0%",
            "3.1%",
            "unknown",
            "unknown",
        ]

    def test_undeclared_capacity_is_not_shown_as_zero(self) -> None:
        # A "0" here would read as an empty tier rather than an unknown one.
        rows = _json(_render("usage", _FLEET))["metrics"]["usage"]
        fs = next(r for r in rows if r["compartment"] == "l2/fs")
        assert fs["capacity"] == "--"
        assert fs["ratio"] == "unknown"

    def test_shared_pool_is_attributed_to_the_fleet(self) -> None:
        rows = _json(_render("usage", _FLEET))["metrics"]["usage"]
        shared = next(r for r in rows if r["compartment"] == "l2/s3")
        assert shared["instance"] == "(fleet-shared)"

    def test_a_single_instance_reply_renders_as_one_server(self) -> None:
        body = _FLEET["instances"][0]
        rows = _json(_render("usage", body))["metrics"]["usage"]
        assert {r["instance"] for r in rows} == {"mp-1"}

    def test_numeric_columns_are_right_aligned(self) -> None:
        # Sizes must line up on the decimal point to be comparable by eye.
        lines = _terminal(_render("usage", _FLEET)).splitlines()
        body = [ln for ln in lines if "l1/dram" in ln]
        assert len(body) == 2
        assert all(line.index("GB") == body[0].index("GB") for line in body)

    def test_an_empty_fleet_says_so(self) -> None:
        out = _terminal(_render("usage", {"instances": [], "shared_modules": []}))
        assert "(none)" in out


class TestOtherApis:
    def test_instances_lists_addresses(self) -> None:
        body = {
            "instances": [
                {
                    "instance_id": "mp-1",
                    "ip": "10.0.0.1",
                    "http_port": 8101,
                    "mq_port": 0,
                    "p2p_advertised_url": "",
                }
            ]
        }
        row = _json(_render("instances", body))["metrics"]["instances"][0]
        assert row["address"] == "10.0.0.1:8101"
        # Absent optional values read as absent, not as 0 / "".
        assert row["mq_port"] == "--"
        assert row["p2p"] == "--"

    def test_an_address_column_is_not_treated_as_numeric(self) -> None:
        # "10.0.0.1:8000" starts with a digit but is text; right-aligning it
        # makes the column look broken next to a left-aligned header.
        body = {
            "instances": [
                {
                    "instance_id": "mp-1",
                    "ip": "10.0.0.1",
                    "http_port": 8000,
                    "mq_port": 0,
                    "p2p_advertised_url": "",
                }
            ]
        }
        lines = _terminal(_render("instances", body)).splitlines()
        header = next(ln for ln in lines if ln.startswith("instance"))
        row = next(ln for ln in lines if ln.startswith("mp-1"))
        assert header.index("address") == row.index("10.0.0.1:8000")

    def test_health(self) -> None:
        assert (
            _json(_render("health", {"status": "healthy"}))["metrics"]["status"]
            == "healthy"
        )

    def test_quota_list_and_single_salt_both_render(self) -> None:
        listing = {
            "total_gb": 19.0,
            "by_cache_salt": [
                {
                    "cache_salt": "",
                    "usage_gb": 19.0,
                    "quota_limit_gb": 0.0,
                    "quota_exists": False,
                }
            ],
        }
        rows = _json(_render("quota", listing))["metrics"]["by_cache_salt"]
        assert rows[0]["cache_salt"] == "(default)"
        assert rows[0]["quota_exists"] == "no"

        single = {
            "cache_salt": "t",
            "usage_gb": 1.0,
            "quota_limit_gb": 0.0,
            "quota_exists": False,
        }
        assert _json(_render("quota", single))["metrics"]["cache_salt"] == "t"

    def test_quota_config_distinguishes_exempt_from_zero(self) -> None:
        exempt = _json(_render("quota-config", {"default_limit_gb": None}))
        assert exempt["metrics"]["default_limit_gb"] == "none (exempt)"
        zero = _json(_render("quota-config", {"default_limit_gb": 0.0}))
        assert zero["metrics"]["default_limit_gb"] == 0.0

    def test_keys_truncates_the_chunk_hash(self) -> None:
        # 64 hex chars per row would push the placements column off-screen.
        body = {
            "total": 1,
            "keys": [
                {
                    "key": {
                        "chunk_hash_hex": "ab" * 32,
                        "model_name": "m",
                        "kv_rank": 0,
                        "cache_salt": "",
                    },
                    "placements": [
                        {"instance_id": "mp-1", "tier": "l1", "backend": "dram"}
                    ],
                }
            ],
        }
        row = _json(_render("keys", body))["metrics"]["keys"][0]
        assert row["chunk"] == "ab" * 6
        assert row["placements"] == "mp-1:l1/dram"
        assert row["cache_salt"] == "(default)"

    def test_directory_stats_nests_the_blend_index(self) -> None:
        body = {
            "num_keys": 3,
            "num_placements": 4,
            "blend": {"num_contents": 1, "num_chunks": 2, "table_size": 1024},
        }
        metrics = _json(_render("directory", body))["metrics"]
        assert metrics["num_keys"] == 3
        assert metrics["blend"]["table_size"] == 1024


class TestPaths:
    def test_usage_narrows_to_one_instance(self) -> None:
        assert APIS["usage"].path(_args()) == "/instances/usage"
        assert APIS["usage"].path(_args(instance="mp-1")) == "/instances/mp-1/usage"

    def test_quota_narrows_to_one_salt(self) -> None:
        assert APIS["quota"].path(_args()) == "/quota"
        # An empty salt is a real tenant, not "unset".
        assert APIS["quota"].path(_args(cache_salt="")) == "/quota/"

    def test_every_api_builds_a_path(self) -> None:
        args = _args(instance="mp-1", request_id="r1", cache_salt="s")
        for name, api in APIS.items():
            assert APIS[name].path(args).startswith("/"), name

    @pytest.mark.parametrize("url", ["127.0.0.1:9300", "http://127.0.0.1:9300/"])
    def test_url_is_normalized(self, url: str) -> None:
        assert normalize_url(url) == "http://127.0.0.1:9300"


class TestRequiredArguments:
    def test_prefetch_needs_an_instance_and_a_request(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            CoordinatorQueryCommand().execute(_args(api="prefetch"))
        # 2 is the conventional exit for a usage error, distinct from a
        # failed request (1).
        assert excinfo.value.code == 2

    def test_only_prefetch_requires_extra_arguments(self) -> None:
        assert {name for name, api in APIS.items() if api.requires} == {"prefetch"}

    def test_metrics_is_the_only_non_json_api(self) -> None:
        assert {name for name, api in APIS.items() if api.raw} == {"metrics"}


class TestFetchErrors:
    def test_timeout_exits_without_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A coordinator timeout is reported as a normal CLI failure."""
        errors: list[tuple[object, ...]] = []

        def raise_timeout(_url: str, timeout: int) -> None:
            raise TimeoutError(f"timed out after {timeout}s")

        monkeypatch.setattr(
            "lmcache.cli.commands.query._coordinator.urllib.request.urlopen",
            raise_timeout,
        )
        monkeypatch.setattr(
            "lmcache.cli.commands.query._coordinator.logger.error",
            lambda *args: errors.append(args),
        )

        with pytest.raises(SystemExit, match="1"):
            get_text("http://127.0.0.1:9300/health")

        assert errors[0][:3] == (
            "Timed out contacting %s after %ds (%s)",
            "http://127.0.0.1:9300/health",
            10,
        )
        assert str(errors[0][3]) == "timed out after 10s"
