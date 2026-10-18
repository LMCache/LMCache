# SPDX-License-Identifier: Apache-2.0
"""Tests for the glossary RST generator in ``tools/glossary/build_glossary.py``."""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "tools" / "glossary" / "build_glossary.py"
SOURCE_MD = REPO_ROOT / "tools" / "glossary" / "glossary.md"
DOCS_GLOSSARY = REPO_ROOT / "docs" / "source" / "glossary"

# kubernetes.rst is hand-authored from the tier-5 sidecar, not generated.
HAND_AUTHORED = {"kubernetes.rst"}


def _load_generator() -> ModuleType:
    """Import the generator script, which lives outside any package."""
    spec = importlib.util.spec_from_file_location("build_glossary", GENERATOR)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_glossary"] = module
    spec.loader.exec_module(module)
    return module


bg = _load_generator()

SAMPLE_MD = """# Preamble that the parser must skip

<!--
slug: ignored
-->
### Not a term

## Tier 1 -- Inference primitives

<!--
slug: token
question: What is a token?
aliases: [tokens, subword]
related: [tokenization, pod]
tier: 1
publish: yes
-->
### Token

**A token is the unit a model reads.** More detail follows.

## Tier 5 -- Systems

<!--
slug: pod
question: What is a pod?
aliases: []
related: [token]
tier: 5
publish: yes
-->
### Pod

**A pod is the smallest deployable unit in Kubernetes.**
"""


class TestParse:
    def test_skips_preamble_and_reads_metadata(self) -> None:
        terms = bg.parse(SAMPLE_MD)
        assert [t.slug for t in terms] == ["token", "pod"]
        token = terms[0]
        assert isinstance(token, bg.GlossaryTerm)
        assert token.title == "Token"
        assert token.question == "What is a token?"
        assert token.aliases == ("tokens", "subword")
        assert token.related == ("tokenization", "pod")
        assert token.tier == "1"
        assert token.body.startswith("**A token is the unit")

    def test_empty_list_metadata(self) -> None:
        pod = bg.parse(SAMPLE_MD)[1]
        assert pod.aliases == ()


class TestMdToRst:
    def test_inline_code(self) -> None:
        assert bg.md_to_rst("set `chunk_size`", {}) == "set ``chunk_size``"

    def test_glossary_link_follows_redirect(self) -> None:
        text = "see [pods](/glossary/pod) and [tokens](/glossary/token)"
        rst = bg.md_to_rst(text, {"pod": "kubernetes"})
        assert rst == "see :doc:`pods <kubernetes>` and :doc:`tokens <token>`"

    def test_external_link(self) -> None:
        rst = bg.md_to_rst("[vLLM](https://vllm.ai)", {})
        assert rst == "`vLLM <https://vllm.ai>`_"

    def test_fenced_block_becomes_literal(self) -> None:
        rst = bg.md_to_rst("```\nlmcache server\n```", {})
        assert rst == "::\n\n   lmcache server"

    def test_blockquote_becomes_note(self) -> None:
        rst = bg.md_to_rst("> Careful here.\n>\n> Second line.", {})
        assert rst == ".. note::\n\n   Careful here.\n\n   Second line."


class TestFirstSentence:
    def test_strips_markup(self) -> None:
        body = "**A `KV` cache** stores [keys](/glossary/qkv). Then more."
        assert bg.first_sentence(body) == "A KV cache stores keys."

    def test_truncates_long_sentences(self) -> None:
        sentence = bg.first_sentence(("word " * 60).strip() + ".")
        assert len(sentence) <= 155
        assert sentence.endswith("...")


class TestRenderTerm:
    def test_metadata_title_and_seealso(self) -> None:
        token = bg.parse(SAMPLE_MD)[0]
        rst = bg.render_term(token, {"pod": "kubernetes"})
        assert "   slug: token" in rst
        assert "   canonical: https://lmcache.ai/glossary/token" in rst
        assert ".. _glossary-token:\n\nToken\n=====\n" in rst
        assert "   :doc:`tokenization` | :doc:`kubernetes`" in rst
        assert rst.endswith("\n")


class TestBuild:
    def test_writes_pages_index_and_sidecar(self, tmp_path: Path) -> None:
        src = tmp_path / "glossary.md"
        src.write_text(SAMPLE_MD)
        outdir = tmp_path / "out" / "glossary"

        bg.build(src, outdir)

        assert sorted(p.name for p in outdir.iterdir()) == ["index.rst", "token.rst"]
        sidecar = json.loads(
            (tmp_path / "out" / "tier5-systems-terms.json").read_text()
        )
        assert [t["slug"] for t in sidecar] == ["pod"]

    def test_checked_in_pages_match_source(self, tmp_path: Path) -> None:
        """The committed RST must be exactly what the generator produces."""
        outdir = tmp_path / "glossary"
        bg.build(SOURCE_MD, outdir)

        generated = {p.name for p in outdir.iterdir()}
        committed = {p.name for p in DOCS_GLOSSARY.iterdir()} - HAND_AUTHORED
        assert generated == committed
        for name in sorted(generated):
            assert (outdir / name).read_text() == (DOCS_GLOSSARY / name).read_text(), (
                f"{name} is stale; rerun build_glossary.py"
            )
