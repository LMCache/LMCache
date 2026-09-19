# SPDX-License-Identifier: Apache-2.0
"""
Unit and integration tests for DeferWindowedStorePolicy.

The policy keeps windowed object groups out of L2 and writes every other
key through, exactly like DefaultStorePolicy. The StoreController tests at the
bottom need a torch runtime and are gated the way the other L1-backed modules
in this directory are.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.object_group_classifier import ObjectGroupClassifier
from lmcache.v1.distributed.storage_controllers.defer_windowed_store_policy import (
    DeferWindowedStorePolicy,
)
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    create_store_policy,
)
from tests.v1.distributed.utils import should_use_lazy_alloc

requires_torch_runtime = pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"Requires available {torch_device_type} runtime",
)

MODEL_NAME = "test_model"

# Group 0 needs the whole prefix, group 1 is windowed with 4 chunks.
HYBRID_DESC = AttnWindowDesc(num_chunks_in_sw=[-1, 4])

WHOLE_PREFIX_GROUP = 0
WINDOWED_GROUP = 1


# =============================================================================
# Helpers
# =============================================================================


def make_object_key(
    chunk_id: int,
    object_group_id: int = WHOLE_PREFIX_GROUP,
    model_name: str = MODEL_NAME,
) -> ObjectKey:
    """Create a test ObjectKey in the given object group."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        object_group_id=object_group_id,
    )


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


def make_hybrid_policy() -> DeferWindowedStorePolicy:
    """Create a policy whose classifier knows the hybrid test model."""
    classifier = ObjectGroupClassifier()
    classifier.register(MODEL_NAME, HYBRID_DESC)
    return DeferWindowedStorePolicy(classifier)


# =============================================================================
# select_store_targets
# =============================================================================


class TestDeferWindowedStoreTargets:
    """Test DeferWindowedStorePolicy.select_store_targets."""

    def test_whole_prefix_keys_go_to_every_adapter(self):
        """Full-attention keys are written through like the default policy."""
        policy = make_hybrid_policy()
        keys = [make_object_key(i, WHOLE_PREFIX_GROUP) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_store_targets(keys, adapters)

        assert result == {0: keys, 1: keys}

    def test_windowed_keys_reach_no_adapter(self):
        """Sliding-window keys appear in no adapter's list."""
        policy = make_hybrid_policy()
        keys = [make_object_key(i, WINDOWED_GROUP) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_store_targets(keys, adapters)

        for adapter_keys in result.values():
            assert adapter_keys == []

    def test_unknown_model_keys_go_to_every_adapter(self):
        """An unclassifiable key is stored, never silently dropped."""
        policy = make_hybrid_policy()
        keys = [make_object_key(0, WINDOWED_GROUP, model_name="other_model")]
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets(keys, adapters)

        assert result == {0: keys}

    def test_out_of_range_object_group_goes_to_every_adapter(self):
        """A group beyond the registered layout is unknown, so it is stored."""
        policy = make_hybrid_policy()
        keys = [make_object_key(0, object_group_id=7)]
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets(keys, adapters)

        assert result == {0: keys}

    def test_mixed_batch_keeps_key_order(self):
        """Full-attention keys keep their relative order in every list."""
        policy = make_hybrid_policy()
        full_keys = [make_object_key(i, WHOLE_PREFIX_GROUP) for i in range(4)]
        window_keys = [make_object_key(i, WINDOWED_GROUP) for i in range(4)]
        # Interleave: fa0, sw0, fa1, sw1, ...
        keys = [k for pair in zip(full_keys, window_keys, strict=True) for k in pair]
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_store_targets(keys, adapters)

        assert result[0] == full_keys
        assert result[1] == full_keys

    def test_no_adapters_yields_empty_plan(self):
        """With no adapters attached the plan is empty."""
        policy = make_hybrid_policy()
        keys = [make_object_key(0, WHOLE_PREFIX_GROUP)]

        assert policy.select_store_targets(keys, []) == {}

    def test_adapter_lists_are_independent_copies(self):
        """Each adapter gets its own list, safe for callers to mutate."""
        policy = make_hybrid_policy()
        keys = [make_object_key(0, WHOLE_PREFIX_GROUP)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_store_targets(keys, adapters)
        result[0].clear()

        assert result[1] == keys

    def test_classification_follows_late_registration(self):
        """A model registered after construction is classified from then on."""
        classifier = ObjectGroupClassifier()
        policy = DeferWindowedStorePolicy(classifier)
        keys = [make_object_key(0, WINDOWED_GROUP)]
        adapters = [make_descriptor(0)]

        assert policy.select_store_targets(keys, adapters) == {0: keys}

        classifier.register(MODEL_NAME, HYBRID_DESC)
        assert policy.select_store_targets(keys, adapters) == {0: []}


class TestDeferWindowedL1Deletions:
    """Test DeferWindowedStorePolicy.select_l1_deletions."""

    def test_never_deletes_from_l1(self):
        """The clean copy stays in L1 for every class of key."""
        policy = make_hybrid_policy()
        keys = [
            make_object_key(0, WHOLE_PREFIX_GROUP),
            make_object_key(1, WINDOWED_GROUP),
            make_object_key(2, model_name="other_model"),
        ]

        assert policy.select_l1_deletions(keys) == []

    def test_empty_input(self):
        """An empty batch yields an empty deletion list."""
        policy = make_hybrid_policy()

        assert policy.select_l1_deletions([]) == []


class TestDeferWindowedRegistration:
    """Test the policy's registration in the store policy registry."""

    def test_created_by_name_with_injected_classifier(self):
        """create_store_policy wires the classifier into the policy."""
        classifier = ObjectGroupClassifier()
        classifier.register(MODEL_NAME, HYBRID_DESC)

        policy = create_store_policy("defer_windowed", classifier)

        assert isinstance(policy, DeferWindowedStorePolicy)
        keys = [make_object_key(0, WINDOWED_GROUP)]
        assert policy.select_store_targets(keys, [make_descriptor(0)]) == {0: []}

    def test_default_policy_ignores_the_classifier(self):
        """Policies that do not classify keep their behavior and constructor."""
        classifier = ObjectGroupClassifier()
        classifier.register(MODEL_NAME, HYBRID_DESC)

        policy = create_store_policy("default", classifier)

        keys = [
            make_object_key(0, WHOLE_PREFIX_GROUP),
            make_object_key(1, WINDOWED_GROUP),
        ]
        assert policy.select_store_targets(keys, [make_descriptor(0)]) == {0: keys}


# =============================================================================
# StoreController integration
# =============================================================================


def make_layout() -> MemoryLayoutDesc:
    """Create a small MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def wait_for_condition(
    predicate,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    """
    Poll until a predicate returns True or timeout.

    Args:
        predicate: Callable returning bool.
        timeout: Max wait time in seconds.
        poll_interval: Time between polls in seconds.

    Returns:
        True if predicate was satisfied, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


@pytest.fixture
def l1_manager():
    """Create an L1Manager with a reasonable memory config."""
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=should_use_lazy_alloc(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(config)
    yield mgr
    mgr.close()


@requires_torch_runtime
class TestDeferWindowedStoreController:
    """Test defer_windowed end to end through the StoreController."""

    def test_only_whole_prefix_reaches_l2_both_stay_in_l1(self, l1_manager):
        """A mixed L1 write stores only the whole-prefix key to L2."""
        classifier = ObjectGroupClassifier()
        classifier.register(MODEL_NAME, HYBRID_DESC)
        adapter = MockL2Adapter(
            MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
        )
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=create_store_policy("defer_windowed", classifier),
        )
        ctrl.start()

        full_key = make_object_key(0, WHOLE_PREFIX_GROUP)
        window_key = make_object_key(1, WINDOWED_GROUP)
        keys = [full_key, window_key]

        layout = make_layout()
        results = l1_manager.reserve_write(
            keys=keys,
            is_temporary=[False] * len(keys),
            layout_desc=layout,
            mode="new",
        )
        written = [k for k, (_e, m) in results.items() if m is not None]
        assert set(written) == set(keys)
        l1_manager.finish_write(written)

        assert wait_for_condition(lambda: adapter.debug_has_key(full_key)), (
            "The whole-prefix key should be stored to L2"
        )
        # Give the controller a chance to do the wrong thing before asserting.
        time.sleep(0.5)
        assert not adapter.debug_has_key(window_key), (
            "The windowed key must not be stored to L2"
        )
        assert adapter.debug_get_stored_object_count() == 1

        # Both keys keep their L1 copy: defer_windowed never deletes from L1.
        assert l1_manager.get_object_state(full_key) is not None
        assert l1_manager.get_object_state(window_key) is not None

        ctrl.stop()
        adapter.close()
