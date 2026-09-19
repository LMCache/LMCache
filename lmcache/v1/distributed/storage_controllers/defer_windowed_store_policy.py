# SPDX-License-Identifier: Apache-2.0
"""
``defer_windowed`` store policy for hybrid models.

Whole-prefix object groups are written through to L2 exactly like the
``default`` policy and keep their clean copy in L1. Windowed object groups --
those whose reuse needs only a bounded number of trailing chunks, which covers
sliding-window attention as well as align/all-mode Mamba and linear-attention
groups -- are never written to L2 by the store path: their chunks stay in L1,
and the only way one reaches L2 is the commit path at a turn boundary (see
``docs/design/v1/distributed/storage_controllers/defer_windowed_store_policy.md``).

Requires the server to run with ``--separate-object-groups`` so that windowed
layers actually have object groups of their own; the MP server config
validation enforces that.
"""

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.object_group_classifier import (
    ObjectGroupClass,
    ObjectGroupClassifier,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    StorePolicy,
    register_store_policy_factory,
)

logger = init_logger(__name__)


class DeferWindowedStorePolicy(StorePolicy):
    """Store whole-prefix keys to L2 and keep windowed keys in L1.

    The windowed/whole-prefix split comes from the attention layout the
    workers register at KV-cache registration time, not from configuration, so
    one server can serve several models with different layouts.

    Keys whose class is :attr:`ObjectGroupClass.UNKNOWN` (model not registered,
    or an object group outside the registered descriptor) are stored like
    whole-prefix keys on purpose: an extra L2 write is far cheaper than losing
    a window that no other tier holds. Each such key is logged at debug
    level.

    Thread safety: the policy holds no mutable state. It is called from the
    store controller thread and delegates classification to
    :class:`ObjectGroupClassifier`, which is itself thread-safe.

    Args:
        classifier: Registry that maps a key's model and object group to its
            cross-chunk reuse class.
    """

    def __init__(self, classifier: ObjectGroupClassifier) -> None:
        self._classifier = classifier

    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """
        Store every key that is not windowed to every adapter.

        Args:
            keys: Keys that were just written to L1.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from every adapter index to the keys classified
            ``WHOLE_PREFIX`` or ``UNKNOWN``, in the order they appear in
            ``keys``. Windowed keys appear in no list and are therefore
            not stored to L2.
        """
        selected: list[ObjectKey] = []
        for key in keys:
            group_class = self._classifier.classify(key)
            if group_class is ObjectGroupClass.WINDOWED:
                continue
            if group_class is ObjectGroupClass.UNKNOWN:
                logger.debug(
                    "defer_windowed: no attention layout for model %r "
                    "object group %d; storing to L2 as whole prefix",
                    key.model_name,
                    key.object_group_id,
                )
            selected.append(key)

        return {ad.index: list(selected) for ad in adapters}

    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """
        Never delete from L1.

        Whole-prefix chunks keep their clean copy in L1 after the L2 write,
        so a later read is an L1 hit and eviction can discard them for free.

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            Empty list (keep all keys in L1).
        """
        return []


register_store_policy_factory("defer_windowed", DeferWindowedStorePolicy)
