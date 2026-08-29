#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate RawBlock meta-payload & kv-payload internal structure diagrams.

Outputs:
  rawblock_payload_overview.png  — whole-device region map (context)
  rawblock_payload_meta.png      — meta-payload JSON schema (checkpoint body)
  rawblock_payload_kv.png        — kv-payload slot byte layout + serde layering
  rawblock_payload.svg           — vector versions of meta + kv panels
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# Palette (consistent with gen_rawblock_layout.py)
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
C_ACCENT = "#7D3C98"
C_SERDE = "#B5651D"
C_PAD = "#5D6D7E"


def _box(ax, x, y, w, h, fc, ec=C_BORDER, lw=1.2, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.0,rounding_size=0.02",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(p)
    return p


def _label(ax, x, y, s, *, size=10, color=C_TEXT, weight="bold",
           ha="center", va="center", zorder=5, family="DejaVu Sans",
           style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, zorder=zorder, family=family, style=style)


# ----------------------------------------------------------------------
# Panel 1 — meta-payload JSON schema (checkpoint body)
# ----------------------------------------------------------------------
def draw_meta_payload() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # Outer wrapping: [Meta Header 32B][Meta Payload = JSON]
    y = 7.2
    H = 1.1
    W = 13.0
    hdr_w = 1.6
    _box(ax, 0, y, hdr_w, H, "#3F4A66")
    _box(ax, hdr_w, y, W - hdr_w, H, C_META_A, lw=1.0)
    _label(ax, hdr_w / 2, y + H / 2, "Meta\nHeader\n(32 B)",
           size=10, color=C_TEXT_INV)
    _label(ax, hdr_w + (W - hdr_w) / 2, y + H / 2,
           "Meta Payload  =  compact JSON (UTF-8, ASCII),  CRC32-protected by header",
           size=11, color=C_TEXT_INV)

    # JSON tree (two columns: top-level scalars on left, entries map on right)
    # ---- left column: top-level scalar fields ----
    lx, ly, lw, lh = 0.3, 0.4, 5.4, 6.4
    _box(ax, lx, ly, lw, lh, "#F4F6FA", lw=1.0)
    _label(ax, lx + lw / 2, ly + lh - 0.35,
           "top-level fields  (layout identity + allocator cursor)",
           size=11, weight="bold")

    fields = [
        ("version", "1", "schema tag"),
        ("device_path", "str", "device identity (checked on load)"),
        ("capacity_bytes", "int", "effective device capacity"),
        ("block_align", "int", "power-of-2 alignment (checked)"),
        ("header_bytes", "int", "slot header size (checked)"),
        ("slot_bytes", "int", "fixed slot size (checked)"),
        ("meta_total_bytes", "int", "meta region size (checked)"),
        ("meta_magic", '"LMCIDX01"', "magic (checked)"),
        ("meta_version", "1", "meta schema version (checked)"),
        ("data_base_offset", "int", "= meta_total_bytes"),
        ("next_slot", "int", "sequential alloc cursor"),
    ]
    row_h = 0.46
    fy = ly + lh - 0.75
    for i, (k, v, note) in enumerate(fields):
        ry = fy - i * row_h
        _label(ax, lx + 0.25, ry, k, size=10.5, ha="left",
               color="#1A5276", weight="bold")
        _label(ax, lx + 2.35, ry, v, size=9.5, ha="left",
               color=C_ACCENT, family="DejaVu Sans Mono")
        _label(ax, lx + 3.55, ry, note, size=8.5, ha="left",
               color="#5D6D7E", weight="normal", style="italic")

    # ---- right column: entries map ----
    rx, ry, rw, rh = 6.0, 0.4, 6.9, 6.4
    _box(ax, rx, ry, rw, rh, "#FBF5EF", lw=1.0)
    _label(ax, rx + rw / 2, ry + rh - 0.35,
           'entries: { encoded_key → entry }   (the live index snapshot)',
           size=11, weight="bold")

    # encoded_key example
    _label(ax, rx + 0.25, ry + rh - 0.85,
           '"llama7b@0x0000003c@f3a1@<hash>@rag:user42"',
           size=8.5, ha="left", color=C_DATA,
           family="DejaVu Sans Mono", weight="bold")

    # entry field table
    efields = [
        ("offset", "int", "absolute byte offset on device"),
        ("size", "int", "logical payload bytes (= slot header payload_len)"),
        ("shape", "[int,..] | null", "logical tensor shape"),
        ("dtype", '"half"|"bfloat16"|.. | null', "element dtype name"),
        ("fmt", '"KV_2LTD"|.. | null', "MemoryFormat enum name"),
        ("cached_positions", "[int,..] | null", "token positions tensor"),
    ]
    erow_h = 0.62
    efy = ry + rh - 1.25
    # header row
    _label(ax, rx + 0.25, efy, "field", size=9.5, ha="left",
           color="#5D6D7E", weight="bold")
    _label(ax, rx + 2.15, efy, "value", size=9.5, ha="left",
           color="#5D6D7E", weight="bold")
    _label(ax, rx + 4.55, efy, "meaning", size=9.5, ha="left",
           color="#5D6D7E", weight="bold")
    for i, (k, v, note) in enumerate(efields):
        ey = efy - (i + 1) * erow_h + 0.15
        _label(ax, rx + 0.25, ey, k, size=10, ha="left",
               color="#1A5276", weight="bold")
        _label(ax, rx + 2.15, ey, v, size=8.8, ha="left",
               color=C_ACCENT, family="DejaVu Sans Mono")
        _label(ax, rx + 4.55, ey, note, size=8.5, ha="left",
               color="#5D6D7E", weight="normal", style="italic")

    # note at bottom
    _label(ax, rx + rw / 2, ry + 0.25,
           "dtype recovered via _recover_checkpoint_dtype (legacy ns falls back to key string)",
           size=8.5, color="#5D6D7E", weight="normal", style="italic")

    # arrow from meta payload box down to the two columns
    ax.annotate("", xy=(lx + lw / 2, ly + lh), xytext=(lx + lw / 2, y),
                arrowprops=dict(arrowstyle="->", color=C_BORDER, lw=1.0))
    ax.annotate("", xy=(rx + rw / 2, ry + rh), xytext=(rx + rw / 2, y),
                arrowprops=dict(arrowstyle="->", color=C_BORDER, lw=1.0))

    ax.set_xlim(-0.3, W + 0.3)
    ax.set_ylim(0.1, y + H + 0.6)
    ax.axis("off")
    ax.set_title("Meta Payload — JSON checkpoint body (inside one mirrored container)",
                 fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Panel 2 — kv-payload slot byte layout + serde layering
# ----------------------------------------------------------------------
def draw_kv_payload() -> plt.Figure:
    fig, (ax_slot, ax_serde) = plt.subplots(2, 1, figsize=(15, 11.5),
                                            gridspec_kw={"height_ratios": [1.3, 1.0]})
    for a in (ax_slot, ax_serde):
        a.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)

    # ---------- Slot byte layout ----------
    ax = ax_slot
    y = 5.2
    H = 1.8
    W = 13.0
    hdr_w = 2.4
    _box(ax, 0, y, hdr_w, H, C_SLOT_HDR)
    _box(ax, hdr_w, y, W - hdr_w, H, C_SLOT_PAYLOAD, lw=1.0)
    _label(ax, hdr_w / 2, y + H / 2,
           f"Slot Header\n(header_bytes,\ndefault 4096)",
           size=10.5, color=C_TEXT_INV)
    _label(ax, hdr_w + (W - hdr_w) / 2, y + H / 2,
           "KV Payload  (raw tensor bytes — opaque to RawBlockCore)",
           size=11, color=C_TEXT_INV)

    # slot header field breakdown
    fy = y - 1.45
    fh = 0.85
    sfields = [
        ("magic\nLMCBLK01", 8, "#922B21"),
        ("slot_identity\nu64 LE", 8, "#7E5109"),
        ("payload_len\nu64 LE", 8, "#145A32"),
        ("zero padding\n→ header_bytes", 4096 - 24, C_PAD),
    ]
    total_units = sum(f[1] for f in sfields)
    x = 0
    scale = W / total_units
    for name, u, col in sfields:
        w = u * scale
        _box(ax, x, fy, w, fh, col, lw=0.8)
        _label(ax, x + w / 2, fy + fh / 2, name,
               size=8.5, color=C_TEXT_INV)
        x += w
    ax.annotate("", xy=(hdr_w / 2, y), xytext=(hdr_w / 2, fy + fh),
                arrowprops=dict(arrowstyle="->", color=C_BORDER, lw=1.0))

    # payload region breakdown: [logical payload][O_DIRECT zero-tail padding]
    pfy = y - 2.85
    pfh = 0.85
    # logical payload takes ~70%, padding ~30% for illustration
    log_w = (W - hdr_w) * 0.72
    pad_w = (W - hdr_w) * 0.28
    _box(ax, hdr_w, pfy, log_w, pfh, C_SLOT_PAYLOAD, lw=0.8)
    _box(ax, hdr_w + log_w, pfy, pad_w, pfh, C_PAD, lw=0.8)
    _label(ax, hdr_w + log_w / 2, pfy + pfh / 2,
           "logical payload  (payload_len bytes)",
           size=9.5, color=C_TEXT_INV)
    _label(ax, hdr_w + log_w + pad_w / 2, pfy + pfh / 2,
           "O_DIRECT\nzero-tail\n(padding)",
           size=8, color=C_TEXT_INV)
    ax.annotate("", xy=(hdr_w + (W - hdr_w) / 2, y),
                xytext=(hdr_w + (W - hdr_w) / 2, pfy + pfh),
                arrowprops=dict(arrowstyle="->", color=C_BORDER, lw=1.0))

    # bracket: total_len
    ax.annotate("", xy=(hdr_w, pfy - 0.2),
                xytext=(W, pfy - 0.2),
                arrowprops=dict(arrowstyle="<->", color=C_TEXT, lw=1.0))
    _label(ax, hdr_w + (W - hdr_w) / 2, pfy - 0.45,
           "total_len = round_up(payload_len, block_align)   "
           "(I/O transfer size when use_odirect / io_uring_cmd)",
           size=9, weight="normal")

    _label(ax, W / 2, y + H + 0.45,
           "One data slot  =  [Slot Header][Payload]   "
           "(_slot_to_offset = data_base + slot × slot_bytes)",
           size=12)

    _label(ax, -0.05, pfy - 0.85,
           "Slot header (24 meaningful bytes):  "
           "magic LMCBLK01 | slot_identity (blake2b of encoded key, u64 LE) | "
           "payload_len (logical bytes, u64 LE) | zero pad → header_bytes.  "
           "header_bytes must be multiple of block_align.",
           size=9, color=C_TEXT, weight="normal", ha="left")

    ax.set_xlim(-0.3, W + 0.3)
    ax.set_ylim(pfy - 1.2, y + H + 0.8)
    ax.axis("off")
    ax.set_title("KV Payload — slot byte layout & O_DIRECT padding",
                 fontsize=13, fontweight="bold", pad=10)

    # ---------- Serde layering (what's actually in the payload bytes) ----------
    ax = ax_serde
    # Stack: [Caller KV tensor] → [SerdeL2AdapterWrapper] → [RawBlockCore] → device
    layers = [
        ("Caller-provided MemoryObj\n(KV tensor: shape, dtype, fmt)",
         "#2C3E50", 2.6),
        ("SerdeL2AdapterWrapper\n(only if serde_config set)",
         C_SERDE, 2.6),
        ("RawBlockCore.put_many / load_many_into\n(writes byte_array verbatim)",
         C_SLOT_HDR, 2.6),
        ("Device slot payload\n(opaque bytes from Core's view)",
         C_SLOT_PAYLOAD, 2.6),
    ]
    ly = 1.0
    lh = 1.4
    gap = 0.55
    x = 0.5
    total_w = 12.0
    for i, (name, col, _) in enumerate(layers):
        lw = total_w / len(layers) - gap * 0.3
        bx = x + i * (total_w / len(layers))
        _box(ax, bx, ly, lw, lh, col, lw=1.0)
        _label(ax, bx + lw / 2, ly + lh / 2, name,
               size=9.5, color=C_TEXT_INV)
        if i < len(layers) - 1:
            ax.annotate("", xy=(bx + lw + gap * 0.3 + 0.05, ly + lh / 2),
                        xytext=(bx + lw - 0.02, ly + lh / 2),
                        arrowprops=dict(arrowstyle="->", color=C_BORDER, lw=1.2))

    # annotations under each layer
    notes = [
        "bfloat16 / fp16 KV\nin MemoryFormat\n(KV_2LTD / KV_T2D / ...)",
        "serialize → temp L1 buffer\n(store)  /  deserialize ← temp\n(load)",
        "sees only byte_array;\ndoes NOT know if bytes\nare quantized/encrypted",
        "payload_len = len(byte_array)\n= what serde emitted\n(or raw tensor if no serde)",
    ]
    for i, n in enumerate(notes):
        bx = x + i * (total_w / len(layers))
        lw = total_w / len(layers) - gap * 0.3
        _label(ax, bx + lw / 2, ly - 0.55, n,
               size=8.3, color="#5D6D7E", weight="normal", style="italic")

    # serde type chips
    _label(ax, x + (total_w / len(layers)) + (total_w / len(layers) - gap * 0.3) / 2,
           ly + lh + 0.35,
           "serde types:  fp8  |  aesgcm  |  turboquant   (or None = raw bytes)",
           size=9.5, color=C_SERDE, weight="bold")

    _label(ax, total_w / 2, 0.25,
           "On load: Core reads payload_len bytes into caller buffer → wrapper deserializes → "
           "caller gets original KV tensor.  fmt/dtype/shape in meta entry tell consumer how to "
           "interpret the restored tensor.",
           size=9, color=C_TEXT, weight="normal")

    ax.set_xlim(-0.3, 13.3)
    ax.set_ylim(-0.1, ly + lh + 0.9)
    ax.axis("off")
    ax.set_title("What's in the payload bytes — serde layering above RawBlockCore",
                 fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Overview (context: where payloads live)
# ----------------------------------------------------------------------
def draw_overview() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 4.8))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    W = 14.0
    H = 1.4
    y = 1.0
    meta_w = 2.2
    data_w = W - meta_w
    _box(ax, 0, y, meta_w, H, "#E8ECF2")
    _box(ax, meta_w, y, data_w, H, "#F5E9E9")

    a_w = meta_w / 2
    _box(ax, 0, y, a_w, H, C_META_A, lw=1.0)
    _box(ax, a_w, y, a_w, H, C_META_B, lw=1.0)
    _label(ax, a_w / 2, y + H / 2, "Meta A", size=11, color=C_TEXT_INV)
    _label(ax, a_w + a_w / 2, y + H / 2, "Meta B", size=11, color=C_TEXT_INV)

    n_slots = 14
    sw = data_w / n_slots
    for i in range(n_slots):
        fc = C_DATA if i % 3 else C_SLOT_FREE
        _box(ax, meta_w + i * sw, y, sw, H, fc, lw=0.8)

    _label(ax, meta_w / 2, y + H + 0.5,
           "Meta Payload  (JSON index)", size=11, color=C_META_A, weight="bold")
    _label(ax, meta_w + data_w / 2, y + H + 0.5,
           "KV Payload  (per-slot tensor bytes)", size=11, color=C_DATA, weight="bold")
    _label(ax, meta_w / 2, y - 0.4,
           "← detailed in panel 1", size=8.5, color="#5D6D7E",
           weight="normal", style="italic")
    _label(ax, meta_w + data_w / 2, y - 0.4,
           "← detailed in panel 2", size=8.5, color="#5D6D7E",
           weight="normal", style="italic")

    ax.set_xlim(-0.4, W + 0.4)
    ax.set_ylim(0.2, y + H + 1.0)
    ax.axis("off")
    ax.set_title("Context — where the two payloads live on device",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def main() -> None:
    ov = draw_overview()
    ov.savefig(OUT / "rawblock_payload_overview.png", dpi=150,
               facecolor=C_BG, bbox_inches="tight")

    mp = draw_meta_payload()
    mp.savefig(OUT / "rawblock_payload_meta.png", dpi=150,
               facecolor=C_BG, bbox_inches="tight")

    kv = draw_kv_payload()
    kv.savefig(OUT / "rawblock_payload_kv.png", dpi=150,
               facecolor=C_BG, bbox_inches="tight")

    mp.savefig(OUT / "rawblock_payload_meta.svg",
               facecolor=C_BG, bbox_inches="tight")
    kv.savefig(OUT / "rawblock_payload_kv.svg",
               facecolor=C_BG, bbox_inches="tight")

    print("Wrote:")
    for p in sorted(OUT.glob("rawblock_payload_*")):
        print(" ", p, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
