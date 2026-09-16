# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LMCache project
"""The layer-major KV staging layout, and which geometries it accepts.

The layout follows the deployment rather than the model: once layer-wise
staging is enabled, every object group is ordered by model depth. A geometry
whose model layers each own a single kernel group gets the identity placement,
where depth order and kernel-group-major order coincide and not one byte moves.
A geometry whose layers span several kernel groups gets a placement that
interleaves the groups, which is what stops it being addressable a whole kernel
group at a time.

Geometries the layout cannot express are refused rather than quietly staged the
other way round; the refusal below proves that a refusal reaches the caller
through the staging buffer. Which geometries are refused, and why, is pinned on
the planner itself in ``test_layer_major_plan.py``.

These tests drive the placement helpers directly with a stub groups manager, so
they need neither a GPU nor a real registration.
"""

# Standard
from unittest.mock import Mock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.layer_major_plan import (
    layer_major_placement,
    placement_keeps_kernel_groups_contiguous,
    set_layer_major_staging_enabled,
    uniform_stride_runs,
)
from lmcache.v1.platform.cuda.cache_context import (
    _TempGPUBuffer,
    _TempLayerMajorGPUBuffer,
)
import lmcache.lmcache_native as lmcache_native


@pytest.fixture(autouse=True)
def _layer_major_opt_in():
    """Detection is gated on the deployment opting in; these tests exercise
    the detection itself, so they opt in and restore the default after."""
    set_layer_major_staging_enabled(True)
    yield
    set_layer_major_staging_enabled(False)


def _placement(buf, object_group_idx: int = 0):
    """Placement of one object group, off a real staging buffer's geometry."""
    return layer_major_placement(
        buf._kv_groups_manager, object_group_idx, buf._get_size_for_kernel_group
    )


def _buffer(groups, object_groups):
    """A ``_TempGPUBuffer`` with just enough state for placement resolution.

    Built without ``__init__`` so no GPU allocation happens; the per-kernel-
    group size, normally derived from the cache shape, is stubbed per group.
    """
    kernel_groups = []
    sizes = {}
    for idx, (layer_indices, model_depths, per_layer) in enumerate(groups):
        group = Mock()
        group.layer_indices = list(layer_indices)
        group.model_depths = None if model_depths is None else list(model_depths)
        group.num_layers = len(layer_indices)
        # A bare Mock would auto-create this attribute, and the placement
        # rule hands it to a pybind function that only accepts the real
        # enum.  Default every group to a per-layer format; the tests that
        # care about the fused one override it explicitly.
        group.engine_kv_format = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
        kernel_groups.append(group)
        sizes[idx] = per_layer * len(layer_indices)

    manager = Mock()
    manager.kernel_groups = kernel_groups
    manager.num_object_groups = len(object_groups)
    manager.object_groups = [
        Mock(kernel_group_indices=list(indices)) for indices in object_groups
    ]

    buf = object.__new__(_TempGPUBuffer)
    buf._kv_groups_manager = manager
    buf._get_size_for_kernel_group = lambda kernel_group_idx: sizes[kernel_group_idx]
    return buf


def _v4_flash_groups():
    """DeepSeek-V4-Flash: 167 caches, 8 kernel groups, type-major registration.

    ``layer_indices`` are registration ordinals -- positions in the flat
    ``kv_caches`` sequence -- while ``model_depths`` carry the transformer
    block each cache belongs to.  The two disagree here, which is the whole
    reason the depths had to be plumbed through.
    """
    even = list(range(2, 43, 2))
    odd = list(range(3, 42, 2))
    depths_per_group = [even, even, odd, [0] + even, [1] + odd, even, even, odd]
    per_layer = [4096, 9344, 1168, 2336, 584, 1024, 2048, 512]

    groups = []
    ordinal = 0
    for depths, size in zip(depths_per_group, per_layer, strict=True):
        indices = list(range(ordinal, ordinal + len(depths)))
        ordinal += len(depths)
        groups.append((indices, depths, size))
    return groups


def test_layer_major_is_selected_for_a_multi_cache_layer_model():
    """V4-Flash: layers own up to five caches, so the layout must change."""
    groups = _v4_flash_groups()
    buf = _buffer(groups, [list(range(len(groups)))])

    placement = _placement(buf)
    assert placement is not None
    assert len(placement) == 167


def test_placement_is_ordered_by_depth_then_kernel_group():
    """The object becomes layer 0\'s caches, then layer 1\'s, and so on."""
    groups = _v4_flash_groups()
    buf = _buffer(groups, [list(range(len(groups)))])

    placement = _placement(buf)
    depth_of = {
        (kg, local): groups[kg][1][local]
        for kg in range(len(groups))
        for local in range(len(groups[kg][1]))
    }
    keys = [(depth_of[(kg, local)], kg) for kg, local, _size in placement]
    assert keys == sorted(keys)

    # Layers 0 and 1 own a single cache each; every even layer owns five.
    assert [kg for depth, kg in keys if depth == 0] == [3]
    assert [kg for depth, kg in keys if depth == 1] == [4]
    assert [kg for depth, kg in keys if depth == 2] == [0, 1, 3, 5, 6]


def test_placement_covers_every_cache_exactly_once():
    """Layer-major moves bytes around; it must not lose or duplicate any."""
    groups = _v4_flash_groups()
    buf = _buffer(groups, [list(range(len(groups)))])

    placement = _placement(buf)
    assert len({(kg, local) for kg, local, _size in placement}) == len(placement)

    total = sum(size for _kg, _local, size in placement)
    assert total == sum(len(depths) * size for _idx, depths, size in groups)


def test_single_kernel_group_model_keeps_its_bytes():
    """Qwen-style: one cache per layer, so the two orders coincide.

    It is placed like every other layer-wise geometry, but the placement
    is the identity, so the staged bytes are exactly what the legacy
    layout produced and the group stays contiguous.
    """
    depths = list(range(64))
    buf = _buffer([(list(range(64)), depths, 8192)], [[0]])

    placement = _placement(buf)
    assert placement == [(0, local, 8192) for local in range(64)]
    assert placement_keeps_kernel_groups_contiguous(placement)


def test_multi_group_model_with_one_cache_per_layer_keeps_legacy_layout():
    """Several kernel groups still coincide when they partition the layers.

    Detection keys on layers spanning groups, not on the group count, so a
    model that merely splits its layers across groups is left alone.
    """
    groups = [
        (list(range(0, 32)), list(range(0, 32)), 8192),
        (list(range(32, 64)), list(range(32, 64)), 4096),
    ]
    buf = _buffer(groups, [[0, 1]])

    assert placement_keeps_kernel_groups_contiguous(_placement(buf))


def test_missing_depths_degrade_to_the_legacy_layout():
    """Without depths the ordinals are all we have, and they are kg-major.

    An engine that reports no depths therefore keeps today\'s behaviour
    rather than getting a layout built on a wrong ordering.
    """
    groups = [(indices, None, size) for indices, _depths, size in _v4_flash_groups()]
    buf = _buffer(groups, [list(range(len(groups)))])

    assert placement_keeps_kernel_groups_contiguous(_placement(buf))


def _interleaved_groups(sizes):
    """Two groups alternating over the layers, which needs the new layout."""
    return [
        (list(range(0, 3)), [0, 2, 4], sizes[0]),
        (list(range(3, 6)), [1, 3, 5], sizes[1]),
    ]


def test_interleaved_groups_break_kernel_group_contiguity():
    """Baseline for the refusal tests below: this geometry interleaves."""
    buf = _buffer(_interleaved_groups([4096, 2048]), [[0, 1]])

    assert not placement_keeps_kernel_groups_contiguous(_placement(buf))


def test_empty_kernel_group_is_refused_instead_of_misaddressed():
    """A group with no layers has no per-layer size to divide out."""
    groups = _interleaved_groups([4096, 2048])
    groups.append(([], [], 4096))
    buf = _buffer(groups, [[0, 1, 2]])

    with pytest.raises(ValueError, match="whole number of staging bytes"):
        _placement(buf)


def test_detection_is_off_until_the_deployment_opts_in():
    """A per-chunk deployment keeps the kernel-group-major layout even for a
    geometry that would otherwise select layer-major."""
    buffer = _buffer(_v4_flash_groups(), [list(range(8))])
    assert _placement(buffer) is not None

    set_layer_major_staging_enabled(False)
    assert _placement(buffer) is None


def _placement_offsets(placement):
    """Byte offset of every ``(kernel group, local layer)`` in the object."""
    offsets = {}
    running = 0
    for kernel_group_idx, local_layer_idx, size in placement:
        offsets[(kernel_group_idx, local_layer_idx)] = running
        running += size
    return offsets


def test_a_kernel_groups_layers_are_evenly_spread_under_layer_major():
    """Most groups stay mergeable: their layers sit a constant distance apart,
    so one launch with a layer stride covers a whole batch."""
    groups = _v4_flash_groups()
    buffer = _buffer(groups, [list(range(len(groups)))])
    offsets = _placement_offsets(_placement(buffer))

    runs_per_group = {}
    for kernel_group_idx, (layer_indices, _, _) in enumerate(groups):
        ordered = [
            offsets[(kernel_group_idx, local)] for local in range(len(layer_indices))
        ]
        runs_per_group[kernel_group_idx] = uniform_stride_runs(ordered)

    # The groups that own an evenly spaced set of layers merge completely.
    for kernel_group_idx in (0, 1, 2, 5, 6, 7):
        assert len(runs_per_group[kernel_group_idx]) == 1, kernel_group_idx

    # The two that also own a leading layer change spacing exactly once, so
    # they cost one extra launch each rather than falling back per layer.
    for kernel_group_idx in (3, 4):
        assert len(runs_per_group[kernel_group_idx]) == 2, kernel_group_idx


def test_merged_runs_use_one_stride_that_actually_reaches_every_layer():
    """The stride a run reports must land on that run's own offsets."""
    groups = _v4_flash_groups()
    buffer = _buffer(groups, [list(range(len(groups)))])
    offsets = _placement_offsets(_placement(buffer))

    for kernel_group_idx, (layer_indices, _, _) in enumerate(groups):
        ordered = [
            offsets[(kernel_group_idx, local)] for local in range(len(layer_indices))
        ]
        for start, length, stride in uniform_stride_runs(ordered):
            for step in range(length):
                assert ordered[start + step] == ordered[start] + step * stride


def test_depth_batches_tile_a_contiguous_byte_range():
    """A batch of consecutive depths is one contiguous copy -- the property
    the layer-major H2D path depends on."""
    groups = _v4_flash_groups()
    buffer = _buffer(groups, [list(range(len(groups)))])
    placement = _placement(buffer)
    offsets = _placement_offsets(placement)
    sizes = {
        (kernel_group_idx, local): size for kernel_group_idx, local, size in placement
    }

    entries = []
    for kernel_group_idx, (_, model_depths, _) in enumerate(groups):
        for local, depth in enumerate(model_depths):
            entries.append((depth, kernel_group_idx, local))
    entries.sort(key=lambda e: (e[0], e[1]))

    depths = sorted({depth for depth, _, _ in entries})
    for start in range(0, len(depths), 4):
        window = set(depths[start : start + 4])
        batch = [e for e in entries if e[0] in window]
        first = offsets[(batch[0][1], batch[0][2])]
        total = sum(sizes[(kg, local)] for _, kg, local in batch)
        last = batch[-1]
        assert offsets[(last[1], last[2])] + sizes[(last[1], last[2])] == first + total


def _carved_layer_major_buffer(groups, object_groups, max_batch_size=2):
    """A ``_TempLayerMajorGPUBuffer`` carved over stubbed geometry.

    Built without ``__init__`` so no GPU allocation happens, then re-carved
    exactly as the subclass does after its base has laid the buffer out in
    kernel-group-major order.
    """
    buf = _buffer(groups, object_groups)
    buf.__class__ = _TempLayerMajorGPUBuffer
    buf._max_batch_size = max_batch_size
    buf._offset_map = {}
    buf._offset_map_kernel_group_only = {}
    buf._offset_map_object_group_only = {}
    buf._offset_map_layer = {}
    buf._recarve_in_depth_order()
    return buf


def test_the_subclass_carves_every_layer_in_depth_order():
    """The moved carve must still place each layer where the placement says.

    ``_placement_offsets`` derives the expected order straight from the
    placement, so this pins the subclass against the layout rule rather
    than against its own previous output.
    """
    groups = _v4_flash_groups()
    buf = _carved_layer_major_buffer(groups, [list(range(8))])

    placement = layer_major_placement(
        buf._kv_groups_manager, 0, buf._get_size_for_kernel_group
    )
    expected = _placement_offsets(placement)

    assert expected
    for (kernel_group_idx, local_layer_idx), offset in expected.items():
        assert (
            buf._offset_map_layer[(0, kernel_group_idx, local_layer_idx)][0] == offset
        )


def test_every_batch_slot_repeats_the_same_depth_order():
    """Slot N is slot 0 shifted by one object group, as the base carve does."""
    groups = _v4_flash_groups()
    buf = _carved_layer_major_buffer(groups, [list(range(8))])

    _, object_group_size = buf._offset_map_object_group_only[(0, 0)]
    assert object_group_size > 0
    for key, (offset, size) in list(buf._offset_map_layer.items()):
        batch_idx, kernel_group_idx, local_layer_idx = key
        if batch_idx != 1:
            continue
        base = buf._offset_map_layer[(0, kernel_group_idx, local_layer_idx)]
        assert offset == base[0] + object_group_size
        assert size == base[1]


def test_the_subclass_reports_the_layer_major_layout():
    groups = _v4_flash_groups()
    buf = _carved_layer_major_buffer(groups, [list(range(8))])

    assert buf.layer_major is True
    assert buf.kernel_groups_contiguous is False


def test_the_base_buffer_never_carves_in_depth_order():
    """The per-chunk class must stay unaware of the layer-wise layout.

    Guards the split itself: if the carve leaked back into the base, this
    attribute would reappear on it.
    """
    assert not hasattr(_TempGPUBuffer, "_recarve_in_depth_order")
    assert (
        "resolve_layer_major_placements"
        not in _TempGPUBuffer.__init__.__code__.co_names
    )
