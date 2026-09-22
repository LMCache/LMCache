# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ObjectGroupClassifier.

The classifier maps an ObjectKey to the cross-chunk attention class of its
object group, from the attention layout registered for the key's model.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, ObjectKey
from lmcache.v1.distributed.object_group_classifier import (
    ObjectGroupClass,
    ObjectGroupClassifier,
)

# =============================================================================
# Helpers
# =============================================================================

# Group 0 needs the whole prefix, groups 1 and 2 are windowed.
HYBRID_DESC = AttnWindowDesc(num_chunks_in_sw=[-1, 4, 1])


def make_object_key(
    object_group_id: int,
    model_name: str = "test_model",
    chunk_id: int = 0,
) -> ObjectKey:
    """Create a test ObjectKey in the given model and object group."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        object_group_id=object_group_id,
    )


# =============================================================================
# classify()
# =============================================================================


class TestClassify:
    """Test ObjectGroupClassifier.classify."""

    def test_whole_prefix_group(self):
        """A group with window -1 classifies as whole prefix."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", HYBRID_DESC)

        assert classifier.classify(make_object_key(0)) is ObjectGroupClass.WHOLE_PREFIX

    def test_windowed_groups(self):
        """Groups with a bounded window classify as windowed."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", HYBRID_DESC)

        for group_id in (1, 2):
            assert (
                classifier.classify(make_object_key(group_id))
                is ObjectGroupClass.WINDOWED
            )

    def test_unknown_model(self):
        """A key of a model that was never registered classifies as unknown."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", HYBRID_DESC)

        key = make_object_key(0, model_name="other_model")
        assert classifier.classify(key) is ObjectGroupClass.UNKNOWN

    def test_object_group_out_of_range(self):
        """A group beyond the registered descriptor classifies as unknown."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", HYBRID_DESC)

        assert classifier.classify(make_object_key(3)) is ObjectGroupClass.UNKNOWN

    def test_empty_classifier(self):
        """Without any registration every key classifies as unknown."""
        classifier = ObjectGroupClassifier()

        assert classifier.classify(make_object_key(0)) is ObjectGroupClass.UNKNOWN

    def test_per_model_layouts_are_independent(self):
        """Two models registered at once keep their own layouts."""
        classifier = ObjectGroupClassifier()
        classifier.register("hybrid", HYBRID_DESC)
        classifier.register("dense", AttnWindowDesc(num_chunks_in_sw=[-1]))

        hybrid_key = make_object_key(1, model_name="hybrid")
        dense_key = make_object_key(0, model_name="dense")
        assert classifier.classify(hybrid_key) is ObjectGroupClass.WINDOWED
        assert classifier.classify(dense_key) is ObjectGroupClass.WHOLE_PREFIX


# =============================================================================
# register() / unregister()
# =============================================================================


class TestRegistrationLifetime:
    """Test the reference-counted registration lifetime."""

    def test_unregister_forgets_the_model(self):
        """After the only registration is dropped, keys classify as unknown."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", HYBRID_DESC)
        classifier.unregister("test_model")

        assert classifier.classify(make_object_key(1)) is ObjectGroupClass.UNKNOWN

    def test_entry_survives_until_last_unregister(self):
        """Several workers share one entry; only the last drop clears it."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", HYBRID_DESC)
        classifier.register("test_model", HYBRID_DESC)

        classifier.unregister("test_model")
        assert classifier.classify(make_object_key(1)) is ObjectGroupClass.WINDOWED

        classifier.unregister("test_model")
        assert classifier.classify(make_object_key(1)) is ObjectGroupClass.UNKNOWN

    def test_unregister_unknown_model_is_noop(self):
        """Unregistering a model that was never registered does nothing."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", HYBRID_DESC)

        classifier.unregister("never_registered")

        assert classifier.classify(make_object_key(0)) is ObjectGroupClass.WHOLE_PREFIX

    def test_re_register_after_unregister(self):
        """A model can be registered again with a new layout once dropped."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", HYBRID_DESC)
        classifier.unregister("test_model")
        classifier.register("test_model", AttnWindowDesc(num_chunks_in_sw=[2]))

        assert classifier.classify(make_object_key(0)) is ObjectGroupClass.WINDOWED


class TestConflictingRegistration:
    """Test rejection of incompatible layouts under one model name."""

    def test_conflicting_windows_raise(self):
        """A different num_chunks_in_sw for a live model is an error."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", HYBRID_DESC)

        with pytest.raises(ValueError, match="already registered"):
            classifier.register("test_model", AttnWindowDesc(num_chunks_in_sw=[-1, 8]))

    def test_rejected_registration_keeps_the_first_layout(self):
        """A rejected registration neither replaces nor counts the entry."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", HYBRID_DESC)

        with pytest.raises(ValueError):
            classifier.register("test_model", AttnWindowDesc(num_chunks_in_sw=[-1]))

        assert classifier.classify(make_object_key(1)) is ObjectGroupClass.WINDOWED
        classifier.unregister("test_model")
        assert classifier.classify(make_object_key(1)) is ObjectGroupClass.UNKNOWN

    def test_same_windows_different_world_size_accepted(self):
        """world_size does not take part in the compatibility check."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", AttnWindowDesc([-1, 4, 1], world_size=1))
        classifier.register("test_model", AttnWindowDesc([-1, 4, 1], world_size=8))

        assert classifier.classify(make_object_key(1)) is ObjectGroupClass.WINDOWED

    def test_same_windows_different_group_kinds_accepted(self):
        """group_kinds does not take part in the compatibility check either."""
        classifier = ObjectGroupClassifier()
        classifier.register("test_model", AttnWindowDesc([-1, 4]))
        classifier.register(
            "test_model",
            AttnWindowDesc([-1, 4], group_kinds=("attention", "recurrent")),
        )

        assert classifier.classify(make_object_key(1)) is ObjectGroupClass.WINDOWED
