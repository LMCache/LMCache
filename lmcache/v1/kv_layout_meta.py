# SPDX-License-Identifier: Apache-2.0
"""Shared metadata helpers for discovered KV layouts.

This is the single Python-side source for parsing a layout name into axis
groups and rendering those groups back into symbolic or concrete shape
strings. Both ``lmcache.kv_layout`` and the ``KVFormatSpec`` layer consume
these helpers so the same layout metadata is not re-declared in two places.
"""

# Standard
from collections.abc import Callable

AXIS_LABELS = {
    "ONE": "1",
    "TWO": "2",
    "NP": "NP",
    "NBBS": "PBS",
    "NB": "NB",
    "NL": "NL",
    "BS": "BS",
    "NH": "NH",
    "HS": "HS",
    "CS": "CS",
    "BSV": "BSxVALS",
    "BSS": "BSxSCALES",
}

AXIS_ACCESSORS = {
    "NB": "num_blocks",
    "NL": "num_layers",
    "BS": "block_size",
    "NH": "num_heads",
    "HS": "head_size",
    "CS": "head_size",
    "PBS": "page_buffer_size",
    "BSxVALS": "block_size",
    "BSxSCALES": "block_size",
}


def parse_axis_groups(layout_name: str) -> tuple[tuple[str, ...], ...]:
    """Split ``NL_X_TWO_NB_BS_NH_HS`` into grouped axes."""
    return tuple(tuple(group.split("_")) for group in layout_name.split("_X_"))


def render_shape(
    axis_groups: tuple[tuple[str, ...], ...],
    token: Callable[[str], str],
) -> str:
    """Render grouped axes into a readable shape string."""

    def _render_group(group: tuple[str, ...], *, wrap: bool) -> str:
        body = ", ".join(token(axis) for axis in group)
        return f"[{body}]" if wrap else body

    if len(axis_groups) == 1:
        return _render_group(axis_groups[0], wrap=True)
    *outer, inner = axis_groups
    return " x ".join(
        [_render_group(group, wrap=False) for group in outer]
        + [_render_group(inner, wrap=True)]
    )


def describe_axis_groups(axis_groups: tuple[tuple[str, ...], ...]) -> str:
    """Render grouped axes with symbolic labels."""
    return render_shape(axis_groups, lambda axis: AXIS_LABELS[axis])


def concrete_axis_groups(
    axis_groups: tuple[tuple[str, ...], ...],
    size: Callable[[str], int],
) -> str:
    """Render grouped axes with concrete numeric sizes."""
    return render_shape(
        axis_groups,
        lambda axis: (
            AXIS_LABELS[axis]
            if axis in ("ONE", "TWO", "NP")
            else str(size(AXIS_LABELS[axis]))
        ),
    )
