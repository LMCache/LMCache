#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate RawBlock on-device storage layout diagrams.

Outputs two PNGs + a combined SVG:
  rawblock_layout_overview.png   — whole-device region map
  rawblock_layout_detail.png     — slot + meta-header byte-level detail
  rawblock_layout.svg            — vector version of both
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# Palette (consistent across panels)
C_META_A = "#4C72B0"
C_META_B = "#55A868"
C_DATA = "#C44E52"
C_SLOT_HDR = "#DD8452"
C_SLOT_PAYLOAD = "#8172B3"
C_SLOT_FREE = "#9AA0A6"
C_BG = "#FAFAFA"
C_BORDER = "#222222"
C_TEXT = "#1A1A1A"
C_TEXT_INV = "#FFFFFF"


def _box(ax, x, y, w, h, fc, ec=C_BORDER, lw=1.2, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.0,rounding_size=0.02",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(p)
    return p


def _label(ax, x, y, s, *, size=10, color=C_TEXT, weight="bold", ha="center", va="center", zorder=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, zorder=zorder, family="DejaVu Sans")


# ----------------------------------------------------------------------
# Panel 1 — whole-device overview
# ----------------------------------------------------------------------
def draw_overview() -> FigureLike:
    fig, ax = plt.subplots(figsize=(15, 6.2))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    W = 14.0
    H = 1.6
    y = 1.0
    # metadata region (≈256MiB) vs data region (rest)
    meta_w = 2.2
    data_w = W - meta_w
    _box(ax, 0, y, meta_w, H, "#E8ECF2")
    _box(ax, meta_w, y, data_w, H, "#F5E9E9")

    # subdivide metadata into A / B mirrored containers
    a_w = meta_w / 2
    _box(ax, 0, y, a_w, H, C_META_A, lw=1.0)
    _box(ax, a_w, y, a_w, H, C_META_B, lw=1.0)
    _label(ax, a_w / 2, y + H / 2, "Meta A", size=11, color=C_TEXT_INV)
    _label(ax, a_w + a_w / 2, y + H / 2, "Meta B", size=11, color=C_TEXT_INV)

    # data region: several slots
    n_slots = 14
    sw = data_w / n_slots
    for i in range(n_slots):
        fc = C_DATA if i % 3 else C_SLOT_FREE
        _box(ax, meta_w + i * sw, y, sw, H, fc, lw=0.8)
    _label(ax, meta_w + 2 * sw, y + H / 2, "…", size=14)
    _label(ax, meta_w + 5 * sw, y + H / 2, "…", size=14)
    _label(ax, meta_w + 8 * sw, y + H / 2, "…", size=14)
    _label(ax, meta_w + 11 * sw, y + H / 2, "…", size=14)

    # top-level brackets / region labels
    _label(ax, meta_w / 2, y + H + 0.55,
           "Metadata Region  (meta_total_bytes, default 256 MiB)",
           size=12)
    _label(ax, meta_w + data_w / 2, y + H + 0.55,
           "Data Region  (fixed-size slots, slot_bytes each)",
           size=12)

    # offset annotations
    ax.annotate("", xy=(0, y - 0.25), xytext=(meta_w, y - 0.25),
                arrowprops=dict(arrowstyle="<->", color=C_TEXT, lw=1.0))
    _label(ax, meta_w / 2, y - 0.5,
           "meta_total_bytes / 2  (each container)", size=9, weight="normal")
    ax.annotate("", xy=(meta_w, y - 0.25), xytext=(W, y - 0.25),
                arrowprops=dict(arrowstyle="<->", color=C_TEXT, lw=1.0))
    _label(ax, meta_w + data_w / 2, y - 0.5,
           "effective_capacity − meta_total_bytes", size=9, weight="normal")

    # data_base_offset marker
    ax.plot([meta_w, meta_w], [y - 0.05, y + H + 0.05],
            color=C_BORDER, lw=1.4, ls="--", zorder=3)
    _label(ax, meta_w, y + H + 0.18, "_data_base_offset",
           size=9, color=C_TEXT, weight="normal")

    # offset 0
    _label(ax, 0, y - 0.5, "0", size=9, weight="normal", ha="center")

    # legend
    legend = [
        Rectangle((0, 0), 1, 1, fc=C_META_A),
        Rectangle((0, 0), 1, 1, fc=C_META_B),
        Rectangle((0, 0), 1, 1, fc=C_DATA),
        Rectangle((0, 0), 1, 1, fc=C_SLOT_FREE),
    ]
    ax.legend(legend, ["Meta checkpoint A", "Meta checkpoint B",
                       "Occupied data slot", "Free data slot"],
              loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, frameon=False, fontsize=10)

    ax.set_xlim(-0.4, W + 0.4)
    ax.set_ylim(-0.1, y + H + 1.0)
    ax.axis("off")
    ax.set_title("RawBlock on-device storage layout — overview",
                 fontsize=15, fontweight="bold", pad=14)
    fig.tight_layout()
    return fig


# type alias for the two draw_* returns
FigureLike = plt.Figure


# ----------------------------------------------------------------------
# Panel 2 — byte-level detail (one meta container + one slot)
# ----------------------------------------------------------------------
def draw_detail() -> FigureLike:
    fig, (ax_meta, ax_slot) = plt.subplots(2, 1, figsize=(15, 10.5),
                                            gridspec_kw={"height_ratios": [1, 1.15]})
    fig.patch.set_facecolor(C_BG)
    for a in (ax_meta, ax_slot):
        a.set_facecolor(C_BG)

    # ---------- Meta container ----------
    y = 1.0
    H = 1.5
    W = 13.0
    hdr_w = 1.2
    _box(ax_meta, 0, y, hdr_w, H, "#3F4A66")
    _box(ax_meta, hdr_w, y, W - hdr_w, H, C_META_A, lw=1.0)
    _label(ax_meta, hdr_w / 2, y + H / 2, "Meta\nHeader\n(block_align)",
           size=10, color=C_TEXT_INV)
    _label(ax_meta, hdr_w + (W - hdr_w) / 2, y + H / 2,
           "Meta Payload  (JSON index snapshot, CRC32-protected)",
           size=11, color=C_TEXT_INV)

    # header field breakdown (blown up below)
    fy = y - 1.35
    fh = 0.8
    fields = [
        ("magic\nLMCIDX01", 8, "#2C3E50"),
        ("version", 4, "#5D6D7E"),
        ("seq (u64 LE)", 8, "#1A5276"),
        ("payload_len (u64 LE)", 8, "#117A65"),
        ("crc32 (u32 LE)", 4, "#7D3C98"),
    ]
    # scale widths for display
    total_units = sum(f[1] for f in fields)
    x = 0
    scale = W / total_units
    for name, u, col in fields:
        w = u * scale
        _box(ax_meta, x, fy, w, fh, col, lw=0.8)
        _label(ax_meta, x + w / 2, fy + fh / 2, name,
               size=8.5, color=C_TEXT_INV)
        x += w
    ax_meta.annotate("", xy=(hdr_w / 2, y), xytext=(hdr_w / 2, fy + fh),
                     arrowprops=dict(arrowstyle="->", color=C_BORDER, lw=1.0))

    _label(ax_meta, W / 2, y + H + 0.45,
           "One mirrored metadata checkpoint container  "
           "(2 copies: A & B, alternating writes → crash-safe)",
           size=12)
    _label(ax_meta, W / 2, fy - 0.35,
           "struct.Struct(\"<8s I Q Q I\")  =  8 + 4 + 8 + 8 + 4  =  32 bytes  "
           "(padded to block_align)",
           size=9.5, color=C_TEXT, weight="normal")

    ax_meta.set_xlim(-0.3, W + 0.3)
    ax_meta.set_ylim(fy - 0.7, y + H + 0.8)
    ax_meta.axis("off")
    ax_meta.set_title("Metadata checkpoint container — byte layout",
                      fontsize=13, fontweight="bold", pad=10)

    # ---------- Data slot ----------
    y = 1.0
    H = 1.8
    hdr_w = 2.0
    _box(ax_slot, 0, y, hdr_w, H, C_SLOT_HDR)
    _box(ax_slot, hdr_w, y, W - hdr_w, H, C_SLOT_PAYLOAD, lw=1.0)
    _label(ax_slot, hdr_w / 2, y + H / 2,
           "Slot Header\n(header_bytes,\ndefault 4096)",
           size=10.5, color=C_TEXT_INV)
    _label(ax_slot, hdr_w + (W - hdr_w) / 2, y + H / 2,
           "KV Payload  (raw tensor bytes; may be quantized / encrypted by serde layer)",
           size=11, color=C_TEXT_INV)

    # slot header field breakdown
    fy = y - 1.55
    fh = 0.9
    sfields = [
        ("magic\nLMCBLK01", 8, "#922B21"),
        ("slot_identity (u64 LE)", 8, "#7E5109"),
        ("payload_len (u64 LE)", 8, "#145A32"),
        ("padding → header_bytes", 4096 - 24, "#5D6D7E"),
    ]
    total_units = sum(f[1] for f in sfields)
    x = 0
    scale = W / total_units
    for name, u, col in sfields:
        w = u * scale
        _box(ax_slot, x, fy, w, fh, col, lw=0.8)
        _label(ax_slot, x + w / 2, fy + fh / 2, name,
               size=8.5, color=C_TEXT_INV)
        x += w
    ax_slot.annotate("", xy=(hdr_w / 2, y), xytext=(hdr_w / 2, fy + fh),
                     arrowprops=dict(arrowstyle="->", color=C_BORDER, lw=1.0))

    _label(ax_slot, W / 2, y + H + 0.45,
           "One data slot  "
           "(slot_bytes; _slot_to_offset = data_base + slot * slot_bytes)",
           size=12)
    _label(ax_slot, W / 2, fy - 0.35,
           "24 meaningful bytes  +  padding to header_bytes  "
           "(header_bytes must be multiple of block_align)",
           size=9.5, color=C_TEXT, weight="normal")

    # slot allocation flow (compact)
    _label(ax_slot, -0.05, fy - 0.95,
           "Allocation (_allocate_slot_locked):  "
           "1) pid_affinity pool  →  2) global free pool  →  "
           "3) sequential _next_slot++  →  4) RuntimeError (no auto-evict; "
           "caller calls delete_many)",
           size=9.5, color=C_TEXT, weight="normal", ha="left")

    ax_slot.set_xlim(-0.3, W + 0.3)
    ax_slot.set_ylim(fy - 1.25, y + H + 0.8)
    ax_slot.axis("off")
    ax_slot.set_title("Data slot — byte layout & allocation",
                      fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------
def main() -> None:
    ov = draw_overview()
    ov.savefig(OUT / "rawblock_layout_overview.png", dpi=150,
               facecolor=C_BG, bbox_inches="tight")
    dt = draw_detail()
    dt.savefig(OUT / "rawblock_layout_detail.png", dpi=150,
               facecolor=C_BG, bbox_inches="tight")

    # combined SVG (vector)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 16))
    a1.axis("off"); a2.axis("off")
    # re-render into the combined figure by calling the draw fns again
    # (cheap; keeps a single vector source of truth)
    ov2 = draw_overview()
    dt2 = draw_detail()
    # save each as SVG then we are done — combined is optional
    ov2.savefig(OUT / "rawblock_layout_overview.svg",
                facecolor=C_BG, bbox_inches="tight")
    dt2.savefig(OUT / "rawblock_layout_detail.svg",
                facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:")
    for p in sorted(OUT.glob("rawblock_layout_*")):
        print(" ", p, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
