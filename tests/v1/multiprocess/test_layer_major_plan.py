# SPDX-License-Identifier: Apache-2.0
"""Host-side tests for the layer-major staging plan.

These cover the half of the feature no GPU test can reach.  The GPU test
(``test_layer_major_stride_gpu.py``) is *handed* a stride and proves the
kernel walks it correctly; nothing there checks that the host computed that
stride from the registered geometry in the first place.  A wrong stride would
make both the store and the load agree on the same wrong bytes, so an
end-to-end generation check cannot see it either -- it only shows up as a
cache miss or as silently mixed-up layers.

Two units are exercised:

* :func:`uniform_stride_runs` -- turns a kernel group's layer offsets into
  the constant-stride runs that each become one launch.
* ``_TempGPUBuffer._layer_major_placement`` -- decides the depth order, and
  decides when *not* to use it.  Its ``None`` returns are the safety net that
  keeps every unrecognised geometry on the legacy layout, so they are tested
  as behaviour rather than treated as defensive code.

Everything here is pure Python: no CUDA, no registered model.  The placement
method is called unbound against a stub that supplies only the three things
it reads.
"""

# Standard
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.layer_major_plan import (
    layer_major_placement,
    layer_major_staging_enabled,
    placement_keeps_kernel_groups_contiguous,
    set_layer_major_staging_enabled,
    uniform_stride_runs,
)
import lmcache.lmcache_native as lmcache_native

# A per-layer format: the layer is selected by the paged pointer array, so
# the engine-side offset does not depend on the layer axis.  Layer-major
# requires this; the fused NB_NL_TWO_BS_NH_HS is the counter-example.
_PER_LAYER_FMT = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
_FUSED_FMT = lmcache_native.EngineKVFormat.NB_NL_TWO_BS_NH_HS


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _Group:
    """The five attributes ``_layer_major_placement`` reads off a group."""

    num_layers: int
    model_depths: list[int]
    engine_kv_format: Any = _PER_LAYER_FMT
    layer_indices: list[int] = field(default_factory=list)


@dataclass
class _ObjectGroup:
    kernel_group_indices: list[int]


class _Manager:
    def __init__(self, groups: list[_Group]) -> None:
        self.kernel_groups = groups
        self.object_groups = [_ObjectGroup(list(range(len(groups))))]


class _Ctx:
    """Minimal stand-in for the inputs ``layer_major_placement`` reads.

    Calling the real function rather than reimplementing it is the point:
    the test fails if the production ordering changes.
    """

    def _layer_major_placement(
        self, object_group_idx: int
    ) -> list[tuple[int, int, int]] | None:
        return layer_major_placement(
            self._kv_groups_manager,
            object_group_idx,
            self._get_size_for_kernel_group,
        )

    def __init__(self, groups: list[_Group], sizes: list[int]) -> None:
        self._kv_groups_manager = _Manager(groups)
        self._sizes = sizes

    def _get_size_for_kernel_group(self, kernel_group_idx: int) -> int:
        return self._sizes[kernel_group_idx]


@pytest.fixture
def layer_major_on():
    """Enable the gate for one test and put it back afterwards."""
    previous = layer_major_staging_enabled()
    set_layer_major_staging_enabled(True)
    try:
        yield
    finally:
        set_layer_major_staging_enabled(previous)


def _offsets_by_group(
    placement: list[tuple[int, int, int]],
) -> dict[int, list[int]]:
    """Byte offsets per kernel group, keyed by kernel group index.

    Mirrors the one-line accumulation ``_TempGPUBuffer.__init__`` performs
    over the placement (``offset += per_layer``), which is what
    ``get_layer_offset_in_object`` later reports relative to the object.
    """
    sizes = [per_layer for _, _, per_layer in placement]
    starts = list(accumulate(sizes, initial=0))[:-1]
    offsets: dict[int, list[int]] = {}
    for (kernel_group_idx, local, _), start in zip(placement, starts, strict=True):
        group = offsets.setdefault(kernel_group_idx, [])
        assert local == len(group), "local index must rise with model depth"
        group.append(start)
    return offsets


# ---------------------------------------------------------------------------
# uniform_stride_runs
# ---------------------------------------------------------------------------


def test_even_spread_is_one_run():
    """The case the whole design rests on: one launch per kernel group."""
    assert uniform_stride_runs([0, 100, 200, 300]) == [(0, 4, 100)]


def test_single_layer_run_has_no_stride():
    """A stride is meaningless for one layer, and 0 tells the kernel so."""
    assert uniform_stride_runs([512]) == [(0, 1, 0)]


def test_no_layers_is_no_runs():
    assert uniform_stride_runs([]) == []


def test_run_breaks_where_the_spread_changes():
    """A group owning a leading layer costs one extra launch, not many.

    The split is greedy: the odd first gap is absorbed into a two-layer run
    rather than isolated, which is the same launch count either way.
    """
    assert uniform_stride_runs([0, 5, 15, 25, 35]) == [(0, 2, 5), (2, 3, 10)]


def test_every_layer_at_a_different_spacing_degrades_to_pairs():
    """Worst case stays correct, not merely small."""
    assert uniform_stride_runs([0, 1, 3, 6, 10]) == [
        (0, 2, 1),
        (2, 2, 3),
        (4, 1, 0),
    ]


@pytest.mark.parametrize(
    "offsets",
    [
        [0, 100, 200, 300],
        [0, 5, 15, 25, 35],
        [0, 1, 3, 6, 10],
        [7],
        [0, 64, 128, 192, 1024, 1088, 1152],
        list(range(0, 43 * 256, 256)),
    ],
)
def test_runs_cover_every_layer_exactly_once_at_the_stated_stride(offsets):
    """The property that makes a run safe to launch as one strided copy."""
    covered = 0
    for start, length, stride in uniform_stride_runs(offsets):
        assert start == covered, "runs must be consecutive and non-overlapping"
        assert length >= 1
        for k in range(length):
            expected = offsets[start] + k * stride
            assert offsets[start + k] == expected
        if length == 1:
            assert stride == 0
        covered += length
    assert covered == len(offsets)


@pytest.mark.parametrize(
    "offsets",
    [
        [0, 100, 200, 300],
        [0, 5, 15, 25, 35],
        [0, 1, 3, 6, 10],
        [0, 64, 128, 192, 1024, 1088, 1152],
    ],
)
def test_runs_are_maximal(offsets):
    """No two adjacent runs could have been merged into one launch.

    Without this the function could return one run per layer and still
    satisfy the coverage property above, which would be correct but would
    give back every launch the feature exists to remove.
    """
    runs = uniform_stride_runs(offsets)
    for (start, length, stride), (next_start, _, _) in zip(
        runs, runs[1:], strict=False
    ):
        assert length >= 1
        joined_stride = offsets[next_start] - offsets[next_start - 1]
        if length > 1:
            assert joined_stride != stride, "adjacent runs share a stride"


# ---------------------------------------------------------------------------
# _layer_major_placement
# ---------------------------------------------------------------------------


def test_placement_is_none_when_the_gate_is_off():
    """Default build must be byte-identical to before the feature."""
    set_layer_major_staging_enabled(False)
    groups = [
        _Group(num_layers=2, model_depths=[0, 2]),
        _Group(num_layers=2, model_depths=[1, 3]),
    ]
    ctx = _Ctx(groups, sizes=[2048, 2048])
    assert ctx._layer_major_placement(0) is None


def test_placement_is_the_identity_for_one_cache_per_layer(layer_major_on):
    """Qwen-shaped model: depth order already *is* kernel-group order.

    It is still placed rather than declined, so the layer-wise path has
    one layout to carry.  The bytes are unchanged because the placement
    is the identity, which :func:`placement_keeps_kernel_groups_contiguous`
    is what reports.
    """
    groups = [_Group(num_layers=64, model_depths=list(range(64)))]
    ctx = _Ctx(groups, sizes=[64 * 512])
    placement = ctx._layer_major_placement(0)
    assert placement == [(0, local, 512) for local in range(64)]
    assert placement_keeps_kernel_groups_contiguous(placement)


def test_placement_orders_by_depth(layer_major_on):
    """Two groups interleaved by depth must interleave in the object."""
    groups = [
        _Group(num_layers=3, model_depths=[0, 2, 4]),
        _Group(num_layers=3, model_depths=[1, 3, 5]),
    ]
    ctx = _Ctx(groups, sizes=[3 * 100, 3 * 60])
    assert ctx._layer_major_placement(0) == [
        (0, 0, 100),
        (1, 0, 60),
        (0, 1, 100),
        (1, 1, 60),
        (0, 2, 100),
        (1, 2, 60),
    ]


def test_placement_refuses_a_size_that_is_not_a_multiple_of_layers(layer_major_on):
    """Cannot name a per-layer slice, so refuse rather than pretend to."""
    groups = [
        _Group(num_layers=3, model_depths=[0, 2, 4]),
        _Group(num_layers=3, model_depths=[1, 3, 5]),
    ]
    ctx = _Ctx(groups, sizes=[301, 180])
    with pytest.raises(ValueError, match="whole number of staging bytes"):
        ctx._layer_major_placement(0)


def test_placement_refuses_depths_that_do_not_match_the_layer_count(layer_major_on):
    """A group registering more caches than depths cannot be depth-ordered."""
    groups = [
        _Group(num_layers=4, model_depths=[0, 2, 4]),
        _Group(num_layers=3, model_depths=[1, 3, 5]),
    ]
    ctx = _Ctx(groups, sizes=[4 * 100, 3 * 60])
    with pytest.raises(ValueError, match="one model depth per layer"):
        ctx._layer_major_placement(0)


def test_placement_refuses_the_fused_cross_layer_format(layer_major_on):
    """Its engine offset is computed from an absolute layer index, so a
    per-layer launch would address the wrong place in the engine tensor.

    The engines producing this format load every layer in one kernel
    through their own adapter, so the layer-wise path refuses it outright
    rather than quietly mis-addressing.
    """
    groups = [
        _Group(num_layers=3, model_depths=[0, 2, 4], engine_kv_format=_FUSED_FMT),
        _Group(num_layers=3, model_depths=[1, 3, 5]),
    ]
    ctx = _Ctx(groups, sizes=[3 * 100, 3 * 60])
    with pytest.raises(ValueError, match="fused cross-layer KV format"):
        ctx._layer_major_placement(0)


def test_placement_refuses_the_fused_cross_layer_format_in_hnd(layer_major_on):
    """The HND twin is the same tensor with heads before block tokens, so
    it is refused on the same grounds rather than slipping through."""
    groups = [
        _Group(
            num_layers=3,
            model_depths=[0, 2, 4],
            engine_kv_format=lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS,
        ),
        _Group(num_layers=3, model_depths=[1, 3, 5]),
    ]
    ctx = _Ctx(groups, sizes=[3 * 100, 3 * 60])
    with pytest.raises(ValueError, match="fused cross-layer KV format"):
        ctx._layer_major_placement(0)


def test_placement_falls_back_to_registration_order_without_depths(layer_major_on):
    """``model_depths`` empty -> ``layer_indices`` is the stand-in."""
    groups = [
        _Group(num_layers=2, model_depths=[], layer_indices=[0, 2]),
        _Group(num_layers=2, model_depths=[], layer_indices=[1, 3]),
    ]
    ctx = _Ctx(groups, sizes=[2 * 100, 2 * 60])
    assert ctx._layer_major_placement(0) == [
        (0, 0, 100),
        (1, 0, 60),
        (0, 1, 100),
        (1, 1, 60),
    ]


# ---------------------------------------------------------------------------
# Placement and run splitting together
# ---------------------------------------------------------------------------


def test_interleaved_groups_collapse_to_one_run_each(layer_major_on):
    """Evenly spread groups: 43 layers become two launches, not 43."""
    even = list(range(0, 43, 2))
    odd = list(range(1, 43, 2))
    per_even, per_odd = 1024, 768
    groups = [
        _Group(num_layers=len(even), model_depths=even),
        _Group(num_layers=len(odd), model_depths=odd),
    ]
    ctx = _Ctx(groups, sizes=[len(even) * per_even, len(odd) * per_odd])

    placement = ctx._layer_major_placement(0)
    assert placement is not None
    offsets = _offsets_by_group(placement)

    stride = per_even + per_odd
    assert uniform_stride_runs(offsets[0]) == [(0, len(even), stride)]
    assert uniform_stride_runs(offsets[1]) == [(0, len(odd), stride)]


def test_a_group_owning_a_leading_layer_costs_one_extra_run(layer_major_on):
    """The uneven case the run splitter exists for.

    Group 0 owns depth 0 *and* every odd depth, so its first gap differs
    from the rest.  That is one extra launch, and the boundary is where the
    spread changes -- not a collapse to one launch per layer.
    """
    leading = [0] + list(range(1, 43, 2))
    rest = list(range(2, 43, 2))
    assert len(leading) + len(rest) == 43
    per_a, per_b = 1024, 768
    groups = [
        _Group(num_layers=len(leading), model_depths=leading),
        _Group(num_layers=len(rest), model_depths=rest),
    ]
    ctx = _Ctx(groups, sizes=[len(leading) * per_a, len(rest) * per_b])

    placement = ctx._layer_major_placement(0)
    assert placement is not None
    offsets = _offsets_by_group(placement)

    assert uniform_stride_runs(offsets[0]) == [
        (0, 2, per_a),
        (2, len(leading) - 2, per_a + per_b),
    ]
    assert uniform_stride_runs(offsets[1]) == [(0, len(rest), per_a + per_b)]


def test_the_object_is_packed_in_depth_order_without_gaps(layer_major_on):
    """Depth order is what makes a range of depths one contiguous copy.

    Read back as a flat byte range, the object must be exactly the layers
    of depth 0, then depth 1, and so on, with nothing between them.
    """
    even = list(range(0, 43, 2))
    odd = list(range(1, 43, 2))
    per_even, per_odd = 1024, 768
    groups = [
        _Group(num_layers=len(even), model_depths=even),
        _Group(num_layers=len(odd), model_depths=odd),
    ]
    ctx = _Ctx(groups, sizes=[len(even) * per_even, len(odd) * per_odd])

    placement = ctx._layer_major_placement(0)
    assert placement is not None

    depth_of = {}
    for kernel_group_idx, depths in ((0, even), (1, odd)):
        for local, depth in enumerate(depths):
            depth_of[(kernel_group_idx, local)] = depth

    seen = [depth_of[(kg, local)] for kg, local, _ in placement]
    assert seen == list(range(43))

    total = len(even) * per_even + len(odd) * per_odd
    assert sum(per_layer for _, _, per_layer in placement) == total
