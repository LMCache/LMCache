# SPDX-License-Identifier: Apache-2.0
"""
Tests for the full_attention_only wiring on the multiprocess side.

Two things are covered:

* the registration hook -- registering a KV cache through the MP layer feeds
  the storage manager's ObjectGroupClassifier, and the last unregistration
  clears it again;
* the start-up validation that rejects ``--l2-store-policy full_attention_only``
  without ``--separate-object-groups``.

The registration tests build a real MPCacheServerContext (and therefore a real
L1 allocation), so they are gated on an available torch runtime like the other
storage-backed modules.
"""

# Standard
from collections.abc import Iterator
import argparse
import logging

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.object_group_classifier import (
    ObjectGroupClass,
    ObjectGroupClassifier,
)
from lmcache.v1.multiprocess.config import (
    add_mp_server_args,
    parse_args_to_mp_server_config,
    validate_server_config,
)
from lmcache.v1.multiprocess.engine_context import (
    LayoutDescRegistry,
    MPCacheServerContext,
)

requires_torch_runtime = pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"Requires available {torch_device_type} runtime",
)

MODEL_NAME = "hybrid_model"

# Group 0 is full attention, group 1 is a 4-chunk sliding window.
HYBRID_DESC = AttnWindowDesc(num_chunks_in_sw=[-1, 4])


# =============================================================================
# Helpers
# =============================================================================


def make_object_key(object_group_id: int, model_name: str = MODEL_NAME) -> ObjectKey:
    """Create an ObjectKey in the given model and object group."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(0),
        model_name=model_name,
        kv_rank=0,
        object_group_id=object_group_id,
    )


def make_layout() -> MemoryLayoutDesc:
    """Create a minimal memory layout descriptor for registration."""
    return MemoryLayoutDesc(shapes=[torch.Size([2, 4])], dtypes=[torch.float16])


def make_storage_manager_config() -> StorageManagerConfig:
    """Create a small full_attention_only storage manager config with no L2."""
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=32 * 1024 * 1024,
                use_lazy=False,
                init_size_in_bytes=32 * 1024 * 1024,
                align_bytes=0x1000,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
        store_policy="full_attention_only",
    )


@pytest.fixture
def server_context() -> Iterator[MPCacheServerContext]:
    """A real MP cache server context using the full_attention_only store policy."""
    ctx = MPCacheServerContext(
        storage_manager_config=make_storage_manager_config(),
        chunk_size=16,
        separate_object_groups=True,
    )
    try:
        yield ctx
    finally:
        ctx.close()


# =============================================================================
# Registration hook
# =============================================================================


@requires_torch_runtime
class TestRegistrationHook:
    """Test that KV-cache registration reaches the storage manager."""

    def test_registration_feeds_the_classifier(self, server_context):
        """The registered attention layout classifies the model's groups."""
        classifier = server_context.storage_manager.object_group_classifier
        assert classifier.classify(make_object_key(0)) is ObjectGroupClass.UNKNOWN

        server_context.layout_desc_registry.register(
            MODEL_NAME, 1, make_layout(), attn_desc=HYBRID_DESC
        )

        assert (
            classifier.classify(make_object_key(0)) is ObjectGroupClass.FULL_ATTENTION
        )
        assert (
            classifier.classify(make_object_key(1)) is ObjectGroupClass.SLIDING_WINDOW
        )

    def test_last_unregister_clears_the_classifier(self, server_context):
        """The layout survives until the last worker unregisters."""
        classifier = server_context.storage_manager.object_group_classifier
        registry = server_context.layout_desc_registry
        registry.register(MODEL_NAME, 1, make_layout(), attn_desc=HYBRID_DESC)
        registry.register(MODEL_NAME, 1, make_layout(), attn_desc=HYBRID_DESC)

        registry.unregister(MODEL_NAME, 1)
        assert (
            classifier.classify(make_object_key(1)) is ObjectGroupClass.SLIDING_WINDOW
        )

        registry.unregister(MODEL_NAME, 1)
        assert classifier.classify(make_object_key(1)) is ObjectGroupClass.UNKNOWN

    def test_world_sizes_share_one_classifier_entry(self, server_context):
        """Attention windows do not depend on the tensor-parallel world size."""
        classifier = server_context.storage_manager.object_group_classifier
        registry = server_context.layout_desc_registry
        registry.register(MODEL_NAME, 1, make_layout(), attn_desc=HYBRID_DESC)
        registry.register(MODEL_NAME, 8, make_layout(), attn_desc=HYBRID_DESC)

        registry.unregister(MODEL_NAME, 8)
        assert (
            classifier.classify(make_object_key(1)) is ObjectGroupClass.SLIDING_WINDOW
        )

        registry.unregister(MODEL_NAME, 1)
        assert classifier.classify(make_object_key(1)) is ObjectGroupClass.UNKNOWN


class TestConflictingLayoutForwarding:
    """Test how the registry handles a layout the classifier rejects."""

    def test_conflicting_layout_is_logged_not_raised(self):
        """A KV-cache registration never fails over a classifier conflict."""
        classifier = ObjectGroupClassifier()
        registry = LayoutDescRegistry(classifier)
        registry.register(MODEL_NAME, 1, make_layout(), attn_desc=HYBRID_DESC)

        # lmcache's ``init_logger`` sets ``propagate = False``, so pytest's
        # root-logger ``caplog`` cannot see the records. Attach a local
        # handler to the named logger instead (established pattern, see
        # tests/v1/multiprocess/test_config.py).
        records: list[logging.LogRecord] = []

        class _ListHandler(logging.Handler):
            """Collect the records emitted while the conflict is forwarded."""

            def emit(self, record: logging.LogRecord) -> None:
                """Append one record to the enclosing list."""
                records.append(record)

        handler = _ListHandler(level=logging.ERROR)
        context_logger = logging.getLogger("lmcache.v1.multiprocess.engine_context")
        context_logger.addHandler(handler)
        try:
            registry.register(
                MODEL_NAME,
                1,
                make_layout(),
                attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1]),
            )
        finally:
            context_logger.removeHandler(handler)

        assert any(MODEL_NAME in r.getMessage() for r in records)
        # The layout registry still took the new descriptor...
        assert registry.find_attn_desc(MODEL_NAME, 1).num_chunks_in_sw == [-1]
        # ...while the classifier kept the first one.
        assert (
            classifier.classify(make_object_key(1)) is ObjectGroupClass.SLIDING_WINDOW
        )

    def test_rejected_registration_is_not_unregistered_twice(self):
        """A rejected forward does not decrement the classifier later."""
        classifier = ObjectGroupClassifier()
        registry = LayoutDescRegistry(classifier)
        registry.register(MODEL_NAME, 1, make_layout(), attn_desc=HYBRID_DESC)
        registry.register(MODEL_NAME, 1, make_layout(), attn_desc=AttnWindowDesc([-1]))

        registry.unregister(MODEL_NAME, 1)

        # Only one registration ever reached the classifier, so the first
        # unregistration is also the last one.
        assert classifier.classify(make_object_key(1)) is ObjectGroupClass.UNKNOWN

    def test_standalone_registry_still_works(self):
        """A registry built without a classifier keeps its old behavior."""
        registry = LayoutDescRegistry()
        registry.register(MODEL_NAME, 1, make_layout(), attn_desc=HYBRID_DESC)

        assert registry.find_attn_desc(MODEL_NAME, 1).num_chunks_in_sw == [-1, 4]


# =============================================================================
# Start-up validation
# =============================================================================


def parse_server_args(argv: list[str]) -> argparse.Namespace:
    """Parse a cache-server command line into a namespace."""
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    return parser.parse_args(["--l1-size-gb", "1", "--eviction-policy", "LRU", *argv])


class TestStorePolicyValidation:
    """Test validate_server_config for the full_attention_only store policy."""

    def test_full_attention_only_without_separate_object_groups_raises(self):
        """The error names both flags so the operator can fix the command."""
        args = parse_server_args(["--l2-store-policy", "full_attention_only"])
        mp_config = parse_args_to_mp_server_config(args)
        storage_manager_config = parse_args_to_config(args)

        with pytest.raises(ValueError) as excinfo:
            validate_server_config(mp_config, storage_manager_config)

        message = str(excinfo.value)
        assert "--l2-store-policy" in message
        assert "full_attention_only" in message
        assert "--separate-object-groups" in message

    def test_full_attention_only_with_separate_object_groups_is_accepted(self):
        """The supported combination builds and validates."""
        args = parse_server_args(
            ["--l2-store-policy", "full_attention_only", "--separate-object-groups"]
        )
        mp_config = parse_args_to_mp_server_config(args)
        storage_manager_config = parse_args_to_config(args)

        validate_server_config(mp_config, storage_manager_config)

        assert storage_manager_config.store_policy == "full_attention_only"
        assert mp_config.separate_object_groups is True

    def test_default_policy_needs_no_object_group_split(self):
        """Other store policies are unaffected by the check."""
        args = parse_server_args([])
        mp_config = parse_args_to_mp_server_config(args)
        storage_manager_config = parse_args_to_config(args)

        validate_server_config(mp_config, storage_manager_config)

        assert storage_manager_config.store_policy == "default"
