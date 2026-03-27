# SPDX-License-Identifier: Apache-2.0
"""Tests for the workload factory."""

# Standard
from unittest.mock import MagicMock
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.config import EngineBenchConfig
from lmcache.cli.commands.bench.engine_bench.workloads import (
    BaseWorkload,
    create_workload,
)
from lmcache.cli.commands.bench.engine_bench.workloads.long_doc_qa import (
    LongDocQAWorkload,
)


def _make_config(**overrides) -> EngineBenchConfig:
    defaults = dict(
        engine_url="http://localhost:8000",
        model="test-model",
        workload="long-doc-qa",
        kv_cache_volume_gb=100.0,
        tokens_per_gb_kvcache=50000,
        seed=42,
        output_dir=".",
        export_csv=True,
        export_json=False,
        quiet=False,
    )
    defaults.update(overrides)
    return EngineBenchConfig(**defaults)  # type: ignore[arg-type]


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        document_length=10000,
        query_per_document=2,
        shuffle_policy="random",
        num_inflight_requests=3,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_deps() -> tuple[MagicMock, MagicMock, MagicMock]:
    sender = MagicMock()
    sender._on_finished = []
    collector = MagicMock()
    monitor = MagicMock()
    return sender, collector, monitor


class TestCreateWorkload:
    def test_long_doc_qa(self) -> None:
        config = _make_config(workload="long-doc-qa")
        args = _make_args()
        sender, collector, monitor = _make_deps()
        result = create_workload(
            config,
            args,
            sender,
            collector,
            monitor,
        )
        assert isinstance(result, BaseWorkload)
        assert isinstance(result, LongDocQAWorkload)
        # 100 * 50000 / 10000 = 500
        assert result._config.num_documents == 500

    def test_long_doc_qa_custom_args(self) -> None:
        config = _make_config(
            workload="long-doc-qa",
            kv_cache_volume_gb=10.0,
            tokens_per_gb_kvcache=10000,
        )
        args = _make_args(
            document_length=5000,
            query_per_document=4,
            shuffle_policy="tile",
        )
        sender, collector, monitor = _make_deps()
        result = create_workload(
            config,
            args,
            sender,
            collector,
            monitor,
        )
        assert isinstance(result, LongDocQAWorkload)
        assert result._config.document_length == 5000
        assert result._config.query_per_document == 4
        assert result._config.shuffle_policy == "tile"
        assert result._config.num_documents == 20  # 10 * 10000 / 5000

    def test_unknown_workload_raises(self) -> None:
        config = _make_config(workload="unknown-workload")
        args = _make_args()
        sender, collector, monitor = _make_deps()
        with pytest.raises(ValueError, match="Unknown workload"):
            create_workload(
                config,
                args,
                sender,
                collector,
                monitor,
            )
