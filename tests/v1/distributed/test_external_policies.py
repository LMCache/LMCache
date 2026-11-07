# SPDX-License-Identifier: Apache-2.0
"""External selector loading through factories and real MP CLI configuration."""

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
import argparse
import importlib
import sys
import textwrap
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.cli.commands.server import ServerCommand
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
    PrefetchPolicy,
    RetainPrefetchPolicy,
    create_prefetch_policy,
    get_registered_prefetch_policies,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    BufferOnlyStorePolicy,
    DefaultStorePolicy,
    StorePolicy,
    create_store_policy,
    get_registered_store_policies,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.utils import single_row_spec
import lmcache.v1.memory_management as memory_management

PolicyFactory = Callable[[str], StorePolicy | PrefetchPolicy]
FACTORIES: list[tuple[PolicyFactory, str]] = [
    (create_store_policy, "ExternalStore"),
    (create_prefetch_policy, "ExternalPrefetch"),
]


@pytest.fixture
def external_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Install an importable policy module outside the LMCache package tree."""
    name = "external_mp_policies_fixture"
    module = tmp_path / f"{name}.py"
    module.write_text(
        textwrap.dedent(
            """\
            from lmcache.v1.distributed.storage_controllers.store_policy import (
                DefaultStorePolicy, StorePolicy,
            )
            from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
                RetainPrefetchPolicy, PrefetchPolicy,
            )

            class ExternalStore(DefaultStorePolicy):
                calls = 0

                def select_store_targets(self, keys, adapters):
                    type(self).calls += 1
                    return {}

            class ExternalStoreAll(DefaultStorePolicy):
                calls = 0

                def select_store_targets(self, keys, adapters):
                    type(self).calls += 1
                    return super().select_store_targets(keys, adapters)

            class ExternalPrefetch(RetainPrefetchPolicy):
                calls = 0

                def plan_load(self, *args, **kwargs):
                    type(self).calls += 1
                    return super().plan_load(*args, **kwargs)

            class BrokenStore(DefaultStorePolicy):
                def __init__(self):
                    raise RuntimeError("external constructor failed")

            class BrokenPrefetch(RetainPrefetchPolicy):
                def __init__(self):
                    raise RuntimeError("external constructor failed")

            class NotAPolicy:
                pass

            not_a_class = 42
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield name
    monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize("factory, class_name", FACTORIES)
def test_external_package_loads_without_registration(
    external_package: str, factory: PolicyFactory, class_name: str
) -> None:
    before = (get_registered_store_policies(), get_registered_prefetch_policies())
    policy = factory(f"{external_package}:{class_name}")
    assert type(policy).__name__ == class_name
    assert type(policy).__module__ == external_package
    assert (
        get_registered_store_policies(),
        get_registered_prefetch_policies(),
    ) == before
    if isinstance(policy, StorePolicy):
        assert policy.select_store_targets([], []) == {}
        assert policy.select_l1_deletions([]) == []


@pytest.mark.parametrize(
    "factory, selection, expected_type",
    [
        (create_store_policy, "default", DefaultStorePolicy),
        (create_store_policy, "skip_l1", BufferOnlyStorePolicy),
        (create_prefetch_policy, "default", DefaultPrefetchPolicy),
        (create_prefetch_policy, "retain", RetainPrefetchPolicy),
    ],
)
def test_builtin_selection_does_not_import_external_packages(
    monkeypatch: pytest.MonkeyPatch,
    factory: PolicyFactory,
    selection: str,
    expected_type: type[StorePolicy] | type[PrefetchPolicy],
) -> None:
    def reject_import(name: str) -> None:
        pytest.fail(f"Built-in selection unexpectedly imported {name}")

    monkeypatch.setattr(importlib, "import_module", reject_import)
    assert type(factory(selection)) is expected_type


@pytest.mark.parametrize("factory, class_name", FACTORIES)
@pytest.mark.parametrize("selection", ["unknown", ":Class", "module:", "m:C:Extra"])
def test_invalid_selection_fails(
    factory: PolicyFactory, class_name: str, selection: str
) -> None:
    with pytest.raises(ValueError, match="policy"):
        factory(selection)


@pytest.mark.parametrize("factory, class_name", FACTORIES)
def test_missing_module_propagates(factory: PolicyFactory, class_name: str) -> None:
    with pytest.raises(ImportError):
        factory("missing_external_mp_package_731.policy:Class")


@pytest.mark.parametrize("factory, class_name", FACTORIES)
def test_missing_attribute_propagates(
    external_package: str, factory: PolicyFactory, class_name: str
) -> None:
    with pytest.raises(AttributeError, match="MissingPolicy"):
        factory(f"{external_package}:MissingPolicy")


@pytest.mark.parametrize("factory, class_name", FACTORIES)
@pytest.mark.parametrize("invalid_class", ["NotAPolicy", "not_a_class"])
def test_wrong_type_rejected(
    external_package: str,
    factory: PolicyFactory,
    class_name: str,
    invalid_class: str,
) -> None:
    with pytest.raises(ValueError, match="concrete"):
        factory(f"{external_package}:{invalid_class}")


@pytest.mark.parametrize(
    "factory, invalid_class",
    [
        (create_store_policy, "StorePolicy"),
        (create_prefetch_policy, "PrefetchPolicy"),
        (create_store_policy, "ExternalPrefetch"),
        (create_prefetch_policy, "ExternalStore"),
    ],
)
def test_abstract_and_other_policy_types_rejected(
    external_package: str, factory: PolicyFactory, invalid_class: str
) -> None:
    with pytest.raises(ValueError, match="concrete"):
        factory(f"{external_package}:{invalid_class}")


@pytest.mark.parametrize(
    "factory, broken_class",
    [(create_store_policy, "BrokenStore"), (create_prefetch_policy, "BrokenPrefetch")],
)
def test_constructor_error_is_not_silently_replaced(
    external_package: str, factory: PolicyFactory, broken_class: str
) -> None:
    with pytest.raises(RuntimeError, match="external constructor failed"):
        factory(f"{external_package}:{broken_class}")


def test_cli_preserves_external_selections_through_config(
    external_package: str,
) -> None:
    parser = argparse.ArgumentParser()
    add_storage_manager_args(parser)
    args = parser.parse_args(
        [
            "--l1-size-gb",
            "0.0625",
            "--eviction-policy",
            "noop",
            "--l2-store-policy",
            f"{external_package}:ExternalStore",
            "--l2-prefetch-policy",
            f"{external_package}:ExternalPrefetch",
        ]
    )
    config = parse_args_to_config(args)
    assert type(create_store_policy(config.store_policy)).__name__ == "ExternalStore"
    assert (
        type(create_prefetch_policy(config.prefetch_policy)).__name__
        == "ExternalPrefetch"
    )


def test_documented_server_command_loads_example_policies() -> None:
    """The complete documented server arguments produce usable selectors."""
    parser = argparse.ArgumentParser()
    ServerCommand().add_arguments(parser)
    config = parse_args_to_config(
        parser.parse_args(
            [
                "--l1-size-gb",
                "1",
                "--eviction-policy",
                "LRU",
                "--l2-store-policy",
                "examples.external_policies.policies:KeepL1Only",
                "--l2-prefetch-policy",
                "examples.external_policies.policies:RetainLoaded",
            ]
        )
    )
    store = create_store_policy(config.store_policy)
    assert store.select_store_targets([], []) == {}
    assert store.select_l1_deletions([]) == []
    assert isinstance(
        create_prefetch_policy(config.prefetch_policy), RetainPrefetchPolicy
    )


@pytest.mark.no_shared_allocator
def test_storage_manager_executes_external_selectors(
    external_package: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real controllers invoke the selected classes and keep readable L1 data."""
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    parser = argparse.ArgumentParser()
    add_storage_manager_args(parser)
    config = parse_args_to_config(
        parser.parse_args(
            [
                "--l1-size-gb",
                "0.0625",
                "--no-l1-use-lazy",
                "--eviction-policy",
                "noop",
                "--l2-store-policy",
                f"{external_package}:ExternalStore",
                "--l2-prefetch-policy",
                f"{external_package}:ExternalPrefetch",
                "--l2-adapter",
                '{"type":"mock","max_size_gb":0.0625,"mock_bandwidth_gb":10}',
            ]
        )
    )
    module = importlib.import_module(external_package)
    manager = StorageManager(config)
    try:
        key = ObjectKey(b"external-policy-key", "external-policy-test", 0)
        layout = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
        objects = manager.reserve_write([key], layout)
        assert key in objects
        tensor = objects[key].tensor
        assert tensor is not None
        tensor.fill_(7)
        manager.finish_write([key])
        deadline = time.monotonic() + 5
        while module.ExternalStore.calls == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert module.ExternalStore.calls > 0
        handle = manager.submit_prefetch_task(single_row_spec([key], layout))
        result = None
        while result is None and time.monotonic() < deadline:
            result = manager.query_prefetch_status(handle)
            if result is None:
                time.sleep(0.01)
        assert result is not None
        assert result.hit_cells[0].popcount() == 1
        assert module.ExternalPrefetch.calls > 0
        with manager.read_prefetched_results([key], result.l1_owners) as read_objects:
            assert read_objects is not None
            read_tensor = read_objects[0].tensor
            assert read_tensor is not None
            torch.testing.assert_close(read_tensor, torch.full_like(read_tensor, 7))
        manager.finish_read_prefetched([key], l1_owners=result.l1_owners)
        assert manager.delete_l1_keys([key]) == (1, 0)
        assert manager.report_status()["is_healthy"]
    finally:
        manager.close()
