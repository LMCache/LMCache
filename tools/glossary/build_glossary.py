# SPDX-License-Identifier: Apache-2.0
"""Generate Sphinx RST glossary pages from the single-source glossary markdown.

Reads:  glossary.md   (one H3 per term, preceded by an HTML-comment metadata block)
Writes: <outdir>/<slug>.rst  for every tier 1-4 term
        <outdir>/index.rst   the glossary hub
        <outdir>/../tier5-systems-terms.json

Tier 5 (systems/Kubernetes) terms are NOT emitted as individual pages; they are
handed to the hand-authored kubernetes.rst page as a JSON sidecar so that page
stays a single SEO target rather than eleven thin ones.

The RST carries a machine-readable metadata comment that Sphinx ignores and the
WordPress sync tool reads. Keeping it inside the .rst (rather than a sidecar)
means the docs file stays the single source of truth for both surfaces.

Usage::

    python tools/glossary/build_glossary.py \\
        tools/glossary/glossary.md docs/source/glossary
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
import argparse
import json
import re

SITE = "https://lmcache.ai"
LAST_REVIEWED = "2026-09-10"
PAGE_TIERS = ("1", "2", "3", "4")
SYSTEMS_TIER = "5"
TIER_NAMES = {
    "1": "Inference primitives",
    "2": "Prefill, decode, and the KV cache",
    "3": "Performance metrics",
    "4": "Engine and hardware knobs",
    "5": "Deployment and systems",
}
TIER_BLURBS = {
    "1": "How a model turns text into text. The foundation everything else rests on.",
    "2": "The specific problem LMCache exists to solve.",
    "3": "The vocabulary of any benchmark claim.",
    "4": "The settings that move the metrics, most of which are not LMCache's.",
}

META_RE = re.compile(r"<!--\n(.*?)\n-->\n### (.+?)\n(.*?)(?=\n<!--|\n## |\Z)", re.S)


@dataclass(frozen=True)
class GlossaryTerm:
    """One glossary entry parsed from the markdown source.

    Attributes:
        slug: URL segment and RST document name, e.g. ``kv-cache``.
        title: Display title taken from the H3 heading.
        question: Question-shaped title for SEO; empty if not given.
        aliases: Alternate names used for search and keywords.
        related: Slugs of sibling terms, rendered as ``seealso`` links.
        tier: Tier number as a string, ``"1"`` through ``"5"``.
        publish: Publish flag from the metadata, ``"yes"`` by default.
        body: Markdown body of the entry, stripped.
    """

    slug: str
    title: str
    question: str
    aliases: tuple[str, ...]
    related: tuple[str, ...]
    tier: str
    publish: str
    body: str


def _parse_list(value: str) -> tuple[str, ...]:
    """Parse a ``[a, b, c]`` metadata value into a tuple of strings."""
    return tuple(x.strip() for x in value.strip("[]").split(",") if x.strip())


def parse(md_text: str) -> list[GlossaryTerm]:
    """Parse every term from the glossary markdown.

    Args:
        md_text: Full contents of ``glossary.md``. Everything before the
            ``## Tier 1`` heading is explanatory preamble and is skipped.

    Returns:
        Terms in source order.

    Raises:
        ValueError: If the text has no ``## Tier 1`` heading.
    """
    start = md_text.find("## Tier 1")
    if start < 0:
        raise ValueError("glossary markdown has no '## Tier 1' heading")
    terms = []
    for m in META_RE.finditer(md_text[start:]):
        meta: dict[str, str] = {}
        for line in m.group(1).strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        terms.append(
            GlossaryTerm(
                slug=meta["slug"],
                title=m.group(2).strip(),
                question=meta.get("question", ""),
                aliases=_parse_list(meta.get("aliases", "")),
                related=_parse_list(meta.get("related", "")),
                tier=meta.get("tier", ""),
                publish=meta.get("publish", "yes"),
                body=m.group(3).strip(),
            )
        )
    return terms


def md_to_rst(text: str, redirect: dict[str, str]) -> str:
    """Convert the restricted markdown subset used in the glossary to RST.

    Handles fenced code blocks, inline code, ``/glossary/<slug>`` links,
    external links, and ``> `` blockquotes (rendered as a note directive).

    Args:
        text: Markdown body of one term.
        redirect: Maps slugs that have no page of their own to the page that
            covers them (tier-5 terms live on ``kubernetes``), so no
            cross-reference dangles.

    Returns:
        The RST body, stripped.
    """

    def doc_link(m: re.Match[str]) -> str:
        slug = redirect.get(m.group(2), m.group(2))
        return f":doc:`{m.group(1)} <{slug}>`"

    out, in_fence = [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            if not in_fence:
                out.extend(["::", ""])
                in_fence = True
            else:
                out.append("")
                in_fence = False
            continue
        if in_fence:
            out.append("   " + line)
            continue
        # Inline code must be converted before anything else touches backticks.
        line = re.sub(r"(?<!`)`([^`]+)`(?!`)", r"``\1``", line)
        line = re.sub(r"\[([^\]]+)\]\(/glossary/([a-z0-9-]+)\)", doc_link, line)
        line = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"`\1 <\2>`_", line)
        out.append(line)

    lines, result, i = "\n".join(out).split("\n"), [], 0
    while i < len(lines):
        if lines[i].startswith("> "):
            block = []
            while i < len(lines) and (lines[i].startswith("> ") or lines[i] == ">"):
                block.append(lines[i][2:] if len(lines[i]) > 1 else "")
                i += 1
            result.extend([".. note::", ""])
            result.extend("   " + b if b else "" for b in block)
            result.append("")
            continue
        result.append(lines[i])
        i += 1
    return "\n".join(result).strip()


def first_sentence(body: str) -> str:
    """Return the lead sentence of a term body as plain text.

    Used for the meta description and the hub's one-line summaries.

    Args:
        body: Markdown body of one term.

    Returns:
        The first sentence of the first paragraph with bold, code, and link
        markup removed, truncated on a word boundary to at most 155
        characters (ending in ``...``) when longer.
    """
    para = body.strip().split("\n\n")[0]
    plain = re.sub(r"\*\*(.+?)\*\*", r"\1", para)
    plain = re.sub(r"`([^`]+)`", r"\1", plain)
    plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
    plain = " ".join(plain.split())
    m = re.search(r"^(.+?[.!?])(\s|$)", plain)
    sent = m.group(1) if m else plain
    if len(sent) > 155:
        sent = sent[:152].rsplit(" ", 1)[0] + "..."
    return sent


def _meta_comment(term: GlossaryTerm) -> list[str]:
    """Build the hidden WORDPRESS-SYNC docutils comment for one term."""
    return [
        ".. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the",
        "   glossary sync tool parses it. Keep the keys stable -- they are the",
        "   contract between this file and the published page.",
        "",
        f"   slug: {term.slug}",
        f"   question: {term.question}",
        f"   aliases: {', '.join(term.aliases)}",
        f"   related: {', '.join(term.related)}",
        f"   tier: {term.tier}",
        f"   publish: {term.publish}",
        "   wp_post_type: glossary",
        "   wp_status: draft",
        f"   canonical: {SITE}/glossary/{term.slug}",
        "   schema_type: DefinedTerm",
        f"   seo_title: {term.question or term.title} | LMCache",
        f"   meta_description: {first_sentence(term.body)}",
        f"   last_reviewed: {LAST_REVIEWED}",
        "   review_status: needs-technical-review",
    ]


def render_term(term: GlossaryTerm, redirect: dict[str, str]) -> str:
    """Render one term as a complete RST page.

    Args:
        term: The term to render.
        redirect: Slug remapping applied to body links and ``seealso``
            entries; see :func:`md_to_rst`.

    Returns:
        RST text ending in a single newline.
    """
    # Dedup while preserving order, after following redirects.
    related = list(dict.fromkeys(redirect.get(r, r) for r in term.related))
    keywords = ", ".join((term.title, *term.aliases))

    parts = _meta_comment(term)
    parts.extend(
        [
            "",
            f".. _glossary-{term.slug}:",
            "",
            term.title,
            "=" * max(len(term.title), 3),
            "",
            ".. meta::",
            f"   :description: {first_sentence(term.body)}",
            f"   :keywords: {keywords}",
            "",
            md_to_rst(term.body, redirect),
            "",
        ]
    )
    if related:
        parts.extend(
            [
                ".. seealso::",
                "",
                "   " + " | ".join(f":doc:`{r}`" for r in related),
                "",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def render_index(terms: list[GlossaryTerm]) -> str:
    """Render the glossary hub page.

    Args:
        terms: All parsed terms; only tiers 1-4 are listed, grouped by tier.

    Returns:
        RST text for ``index.rst``, including a hidden toctree over every
        term page plus ``kubernetes``.
    """
    published = [t for t in terms if t.tier in PAGE_TIERS]
    p = [
        ".. WORDPRESS-SYNC metadata block. Sphinx ignores this comment.",
        "",
        "   slug: glossary",
        "   wp_post_type: page",
        "   wp_status: draft",
        f"   canonical: {SITE}/glossary",
        "   schema_type: DefinedTermSet",
        "   seo_title: LLM Inference and KV Cache Glossary | LMCache",
        "   meta_description: Plain-language definitions of the inference, caching,"
        " benchmarking, and deployment terms used across LMCache.",
        f"   last_reviewed: {LAST_REVIEWED}",
        "",
        ".. _glossary:",
        "",
        "Glossary",
        "========",
        "",
        ".. meta::",
        "   :description: Plain-language definitions of the inference, caching, and"
        " benchmarking terms used across LMCache.",
        "   :keywords: KV cache glossary, LLM inference glossary, TTFT, prefill,"
        " decode",
        "",
        "Definitions of the terms that come up when reading, writing, or benchmarking",
        "anything about LMCache. Every entry stands alone: land on one cold,"
        " understand",
        "it without reading anything above it, and follow the links outward from"
        " there.",
        "",
        "For the infrastructure vocabulary that appears in deployment discussions --",
        "pods, nodes, replicas, probes -- see :doc:`kubernetes`.",
        "",
    ]
    for tier in PAGE_TIERS:
        group = [t for t in published if t.tier == tier]
        if not group:
            continue
        heading = TIER_NAMES[tier]
        p.extend([heading, "-" * len(heading), "", TIER_BLURBS[tier], ""])
        for t in group:
            p.append(f":doc:`{t.title} <{t.slug}>`")
            p.extend("   " + chunk for chunk in wrap(first_sentence(t.body), 88))
            p.append("")
    p.extend([".. toctree::", "   :hidden:", "   :maxdepth: 1", ""])
    p.extend(f"   {t.slug}" for t in published)
    p.extend(["   kubernetes", ""])
    return "\n".join(p)


def build(src: Path, outdir: Path) -> None:
    """Generate every glossary output from the markdown source.

    Writes one page per tier 1-4 term and ``index.rst`` into ``outdir``, and
    the tier-5 sidecar ``tier5-systems-terms.json`` into ``outdir.parent``.
    ``kubernetes.rst`` is hand-authored and never written.

    Args:
        src: Path to ``glossary.md``.
        outdir: Directory for the RST pages; created if missing.

    Raises:
        ValueError: If ``src`` is not a valid glossary source; see :func:`parse`.
    """
    terms = parse(src.read_text())
    published = [t for t in terms if t.tier in PAGE_TIERS]
    systems = [t for t in terms if t.tier == SYSTEMS_TIER]
    # Tier-5 terms have no page of their own; every reference to one lands on
    # the single Kubernetes page instead.
    redirect = {t.slug: "kubernetes" for t in systems}

    outdir.mkdir(parents=True, exist_ok=True)
    for t in published:
        (outdir / f"{t.slug}.rst").write_text(render_term(t, redirect))
    (outdir / "index.rst").write_text(render_index(terms))

    sidecar = [
        {
            "slug": t.slug,
            "title": t.title,
            "question": t.question,
            "definition": first_sentence(t.body),
            "body": t.body,
        }
        for t in systems
    ]
    (outdir.parent / "tier5-systems-terms.json").write_text(
        json.dumps(sidecar, indent=2)
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="path to glossary.md")
    parser.add_argument("outdir", type=Path, help="directory for the RST pages")
    args = parser.parse_args()
    build(args.src, args.outdir)


if __name__ == "__main__":
    main()
