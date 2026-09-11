# SPDX-License-Identifier: Apache-2.0
"""
Object-group classification for storage policies.

Hybrid models mix layers that need the whole prefix with layers that need
only a bounded number of trailing chunks -- sliding-window attention, and also
align/all-mode Mamba and linear-attention layers, which behave like a window of
one block. When the server runs with ``--separate-object-groups`` each kind
gets its own object group, and storage policies want to treat the two
differently (see
``docs/design/v1/distributed/storage_controllers/defer_windowed_store_policy.md``).

The attention layout is only known once a worker registers its KV cache, so
this module keeps a small runtime registry that maps a model name to the
:class:`AttnWindowDesc` registered for it. The registry lives in the
``distributed`` layer (it must not import from ``lmcache.v1.multiprocess``);
the ``multiprocess`` KV-cache registration path pushes descriptors into it
through :class:`lmcache.v1.multiprocess.engine_context.LayoutDescRegistry`,
which receives the classifier owned by the ``StorageManager``.
"""

# Standard
from dataclasses import dataclass
import enum
import threading

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, ObjectKey


class ObjectGroupClass(enum.Enum):
    """How much of the prefix an object group needs to serve a hit.

    The class is derived from ``AttnWindowDesc.num_chunks_in_sw``, so it says
    nothing about the layer's mechanism: a recurrent group with a one-block
    window is :attr:`WINDOWED` exactly like a sliding-window attention group.

    ``UNKNOWN`` means the classifier cannot answer: the model was never
    registered (or is already unregistered), or the key names an object group
    outside the registered descriptor. Callers decide what to do with it;
    storage policies treat it as whole-prefix, because an extra L2 write is
    cheaper than a lost window.
    """

    WHOLE_PREFIX = enum.auto()
    """The group needs every chunk of the prefix (``num_chunks_in_sw`` -1)."""

    WINDOWED = enum.auto()
    """The group needs only a bounded number of trailing chunks (``w >= 1``)."""

    UNKNOWN = enum.auto()
    """No attention layout is known for this key."""


@dataclass
class _ClassifierEntry:
    """A registered attention layout and its active registration count."""

    attn_desc: AttnWindowDesc
    ref_count: int


class ObjectGroupClassifier:
    """Thread-safe registry mapping a model name to its attention layout.

    Keying by ``model_name`` alone is deliberate: :class:`ObjectKey` carries no
    world size, and per-object-group attention windows do not depend on tensor
    parallelism. ``AttnWindowDesc.world_size`` is therefore ignored when
    descriptors are compared.

    Several workers of the same model register the same layout, so entries are
    reference counted the way
    :class:`lmcache.v1.multiprocess.engine_context.LayoutDescRegistry` counts
    layout descriptors: the entry survives until the last registration is
    dropped.

    Thread safety: every public method takes ``self._lock`` for the whole of
    its body; no lock is held across a call into another component, and no
    other component's lock is held while calling in. Instances are shared
    between the KV-cache registration threads and the store controller thread.
    """

    def __init__(self) -> None:
        self._registry: dict[str, _ClassifierEntry] = {}
        self._lock = threading.Lock()

    def register(self, model_name: str, attn_desc: AttnWindowDesc) -> None:
        """Register the attention layout of a model.

        Re-registering the same model with a compatible descriptor increments
        the registration count and keeps the newest descriptor.

        Args:
            model_name: Name of the model, as it appears in
                :attr:`ObjectKey.model_name`.
            attn_desc: The attention-window descriptor registered for the
                model. Only ``num_chunks_in_sw`` participates in the
                compatibility check; ``world_size`` and ``group_kinds`` do not
                affect classification.

        Raises:
            ValueError: If the model is already registered with a different
                ``num_chunks_in_sw``. One server cannot serve two incompatible
                attention layouts under a single model name.
        """
        with self._lock:
            entry = self._registry.get(model_name)
            if entry is None:
                self._registry[model_name] = _ClassifierEntry(
                    attn_desc=attn_desc, ref_count=1
                )
                return

            if entry.attn_desc.num_chunks_in_sw != attn_desc.num_chunks_in_sw:
                raise ValueError(
                    f"Model {model_name!r} is already registered with "
                    f"attention windows {entry.attn_desc.num_chunks_in_sw}; "
                    f"cannot re-register it with "
                    f"{attn_desc.num_chunks_in_sw}. One model name must map "
                    f"to one attention layout."
                )

            entry.attn_desc = attn_desc
            entry.ref_count += 1

    def unregister(self, model_name: str) -> None:
        """Drop one registration of a model's attention layout.

        The layout is forgotten only when the last active registration is
        dropped. Unregistering a model that is not registered is a no-op.

        Args:
            model_name: Name of the model to unregister.
        """
        with self._lock:
            entry = self._registry.get(model_name)
            if entry is None:
                return

            if entry.ref_count <= 1:
                self._registry.pop(model_name)
                return

            entry.ref_count -= 1

    def classify(self, key: ObjectKey) -> ObjectGroupClass:
        """Classify the object group a key belongs to.

        Args:
            key: The object key to classify.

        Returns:
            :attr:`ObjectGroupClass.WHOLE_PREFIX` or
            :attr:`ObjectGroupClass.WINDOWED` when the key's model is
            registered and its ``object_group_id`` is covered by the
            registered descriptor, otherwise
            :attr:`ObjectGroupClass.UNKNOWN`.
        """
        with self._lock:
            entry = self._registry.get(key.model_name)
            if entry is None:
                return ObjectGroupClass.UNKNOWN

            attn_desc = entry.attn_desc
            if not 0 <= key.object_group_id < attn_desc.num_object_groups:
                return ObjectGroupClass.UNKNOWN

            if attn_desc.is_full_attention(key.object_group_id):
                return ObjectGroupClass.WHOLE_PREFIX
            return ObjectGroupClass.WINDOWED
