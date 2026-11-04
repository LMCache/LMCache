# `lmcache bench quality` — Design

**Status:** Prototype  |  **Date:** 2026-10-12

## Overview

`lmcache bench quality` answers one question: **does a serving engine change
the answers?** The candidate can differ from the reference by quantization, KV
cache compression, post-training, or another engine or model. It sends the
same QA prompts to a *reference* engine (e.g. the original model) and to a
*candidate* engine (e.g. a quantized version of it), scores both against gold
answers, and prints one verdict — `PASS`, `FAIL`, `UNDERPOWERED`, or `INVALID`
— that CI can gate on. The work is split into two stages that share a results
directory (Section 6): `run` sends prompts to one or more arms and saves their
answers; `analyze` scores the saved answers and reaches the verdict.

`bench engine --workload rag-qa-quality` already scores one stack's answers.
`bench quality` adds what a comparison needs on top of that: a plan shared by
every arm, an explicit description of how the passages got into the cache
(the *reuse manifest*), a noise floor from repeated reference runs,
optionally a check that KV blending actually happened (Section 3), and a
statistical verdict.

```bash
WORKLOAD="--model Qwen/Qwen3-8B --dataset hotpotqa --limit 200"
# One engine at a time, into one results directory. Reference: the original
# model. Candidate: its FP8 version, served under the same name
# (vllm serve Qwen/Qwen3-8B-FP8 --served-model-name Qwen/Qwen3-8B).
lmcache bench quality run --reference-url http://gpu0:8000 $WORKLOAD --output-dir out
lmcache bench quality run --candidate-url http://gpu0:8000 $WORKLOAD --output-dir out
# No engine needed: score and decide
lmcache bench quality analyze out \
    --judge-url http://judge-host:8000 --judge-model openai/gpt-oss-120b
```

`analyze` exits `1` on `FAIL` or `INVALID`, `0` otherwise. `run` exits `1`
when a pass already guarantees `INVALID` (no successful request, or failed
store requests), so CI flags the job that broke. Both exit `2` when the
results directory does not fit (Section 6), so a broken pipeline is never
mistaken for a quality verdict.

### Scope

| In scope | Not yet |
|---|---|
| Three reuse scenarios: `independent`, `reordered`, `concatenated` | Other placements (shared prefix, worst-case position) |
| QA datasets: HotpotQA, MuSiQue, FinanceBench, any local QA file | Non-QA task categories (next step) |
| LLM judge (RAGAS `AnswerAccuracy`) as the reference metric (Section 5); judge-free `match` and token F1 as a screen | Judge-derived loss tags for non-QA tasks |
| Existing endpoints: the user launches each engine | Launching and restarting engines |

Any file in a supported QA schema loads with `--dataset <path>`.

---

## 1. Arms

| Role | Flag | Engine | Stores passages first? |
|---|---|---|---|
| `reference` | `--reference-url` | the stack every arm is judged against, e.g. the original model on vLLM with prefix caching **off** | No |
| `candidate` | `--candidate-url` | the stack under test, e.g. a quantized, compressed or post-trained model, a KV cache optimization, or another engine | Yes |
| `control` | `--control-url` (optional) | a stack expected to match the reference, e.g. the reference with prefix caching on | Yes |

Each `run` takes any subset of these flags (at least one); `analyze` needs a
reference and a candidate arm in the results directory. Every arm is sent
the `--model` name, so a candidate model is served under that name (vLLM:
`--served-model-name`); the plan, and so the prompts, are built with that
model's tokenizer.

The reference runs `--reference-repeats` times (default 2). Its disagreement
with itself is the **flip rate**, the noise floor below which no arm can be
told apart. The optional `control` arm should match the reference; with
vLLM prefix caching as the control it should match exactly, since prefix
caching reuses KV only at its original positions. A gap there points at the
harness or the engine, not at the candidate. It is reported, and does not set
the verdict.

Within a `run`, arms run one after another (all reference passes, then
`control`, then `candidate`) and questions run sequentially, so per-question
counter deltas belong to that question. **Each caching arm must start from an
empty cache**: restart its engine (and any cache server behind it) before
its `run`. Running one arm per engine lifetime, as the staged flow does,
satisfies this; the command does not restart engines itself.

## 2. Plan and reuse scenarios

`build_plan` turns the dataset into a `QualityPlan`:

* **store requests** sent to caching arms before measuring (`max_tokens=1`),
* **one measured prompt per question**, byte-identical across arms,
* **per-passage provenance** — the reuse manifest.

Prompts are built by the shared `quality/prompts.py` and padded by
`quality/alignment.py` (see Section 4) so that each passage starts on a chunk
boundary and fills whole chunks.

### Padding (`--filler`)

At chunk size 256, short passages are mostly padding: HotpotQA's ~130-token
paragraphs each need ~126 tokens of it, about 45% of a measured prompt. What
that padding *is* therefore matters as much as where it ends:

| `--filler` | Block layout |
|---|---|
| `dots` (default) | `Passage <id>:\n<passage>\n\n . . … .\nend of passage <id>` |
| `dashes` | same, with ` -` |
| `words` | `<passage>\n` + random single-token words (`rag-qa-quality`'s padding) |

The prompt around the passages is the same for every filler, so comparing
two fillers isolates the padding.

`words` was the only padding before the flag. Its filler is random
vocabulary words (on a multilingual tokenizer, a mix of languages and code
identifiers), and nothing separates it from the next passage
(`…cantorLeederville Oval.…`). It lowers the reference's own accuracy and
may cost a KV-reuse candidate more, since recompute budget can go to noise
tokens. The framed fillers use a repeated low-information token instead,
inside a label and an end marker that also make passage boundaries explicit.

The framed layout is built so no tokenizer merges a block with its
neighbours and no chat template changes it:

- **It starts with a letter and ends with a digit.** `<id>` is six hex
  characters of the passage digest plus a digit (`block_id`). A final
  letter merges into the next block's first word on Llama-style
  pre-tokenizers. A final punctuation mark glues to the following newlines
  on GPT-4o-style ones. Final whitespace merges with the question's leading
  `\n\n`, and Gemma's and Llama's templates trim it from a store prompt.
  Every tokenizer checked splits a digit from a following letter or
  newline.
- **The pad unit is one token everywhere checked.** ` .` and ` -` were
  checked for gpt-oss-120b/20b, GLM-5.2, MiniMax-M3/M2.5, Gemma 3/4,
  Nemotron 3 Super/Nano, MiMo-V2.5/V2-Flash and Llama 3.1, and the layout
  aligned every passage of HotpotQA and FinanceBench in both scenarios.
- **The system block ends the same way:** `\nend of instructions 0`.

### Answer tags

`bench quality` asks for `<answer>…</answer>` (`ANSWER_TAG_PROMPT`).
`rag-qa-quality`, whose prompt it shared before, asks for
`<final_answer>…</final_answer>`, which trips up gpt-oss. Its response format
opens the answer channel with `<|channel|>final`. Primed by a prompt full of
`final_answer`, it continues `_answer>…` instead of `<|message|>`, so the
reply has no opening tag (5–9 empty answers per 100 HotpotQA questions in
one run). `extract_final_answer` accepts both tags and that bare form, so
`rag-qa-quality` recovers those answers without a prompt change.

`plan_settings` records `filler`, so a results directory never mixes
fillers. The manifest records each passage's `pad_tokens`. `plan.json` and
`summary.json` record `padding`: the filler, and the share of measured
prompt tokens that are padding.

### Placement check

Padding a block standalone does not guarantee it lands on a boundary in a
real prompt. The chat template wraps the content, and a tokenizer may merge
across a block's edges. So `build_plan` renders every store and measured
prompt with the chat template (and `--template-kwargs`), tokenizes it as the
engine will, and checks each passage (`ChunkAligner.place_blocks`):

- it appears verbatim, in order;
- it starts and ends on chunk boundaries, with no token straddling either;
- it has the same token ids in every prompt that holds it.

Any failure refuses the plan (`PlanAlignmentError`, exit 2) before a
request is sent, listing the misplaced passages. A misplaced passage would
otherwise lose its reuse silently, and the run would measure less reuse than
its plan claims. Real tokenizers fail with `words`: 7 of the 12 above, on
HotpotQA or FinanceBench `reordered`. `dots` passed on all of them.

| Scenario | Stored as | Measured as | What it tests |
|---|---|---|---|
| `independent` | each distinct passage alone, once | the sample's passages in dataset order | the basic RAG case: passages cached separately, combined at query time |
| `reordered` | each sample's passages together, in dataset order | the same passages in a seeded permutation (never the identity) | reuse at a different position *and* with different neighbors |
| `concatenated` | questions' contexts concatenated into units of ≥ `--min-unit-tokens`, no system prompt, no padding | the sample's passages in dataset order, unpadded | reuse of a whole context computed after unrelated contexts, at a new position |

Every passage a question uses is stored before the first question is
measured, so `--limit` (measuring a prefix of the dataset) never changes what
a measured question can reuse.

### Concatenated units

`independent` stores each passage with no neighbours, and needs padding
because short passages (HotpotQA's median is 118 tokens) are smaller than a
256-token chunk. `concatenated` stores whole question contexts instead,
which are 1–3K tokens on the sets checked (gpt-oss tokens: HotpotQA mean
1,354, FinanceBench 1,949, LongBench 2Wiki 2,549, MuSiQue 2,905), so it
needs no padding:

1. **Contexts.** A question's context is its passages in request order, each
   framed as `Passage <id>:\n<text>\n\nend of passage <id>` (the `dots`
   frame with no filler).
2. **Units.** Every question in the dataset (not only the `--limit` measured
   ones) is visited in a shuffle seeded by `--seed`. Its context joins the
   current unit until the unit reaches `--min-unit-tokens`. A context
   identical to one already placed is not placed again: the question
   records the owner as `duplicate_of` (12 of FinanceBench's 150 questions).
   The shuffle keeps related questions (FinanceBench orders questions by
   filing, and 113 of its 149 adjacent pairs share a passage) out of the
   same unit by default.
3. **Stores.** Each unit holding a measured question's context is sent as
   one store request, **without the system prompt**. A unit's first context
   would otherwise sit at the same position as in its request, behind the
   same chat template and system prompt, and be served as an exact prefix:
   at sizes up to about one context, that is 80–97% of all reuse. Without
   it, every reuse is a blend at a new position of KV computed after the
   unit's earlier contexts.
4. **Requests.** Each question asks over its own context, unpadded:
   `system prompt + context + question template`.

A stored chunk is reusable wherever all its tokens recur in a request (the
blend server's matcher probes every offset), so the chunks straddling a
context's edges in its unit are recomputed: 18% of HotpotQA context tokens
and 6% of FinanceBench's at `--min-unit-tokens 8192`. The default, 8192,
puts about 3–6 contexts in a unit. A larger unit costs only stores (the
unit must still fit the engine's `--max-model-len`).

This tests something other than `independent`: a question's passages are
stored together and in order, so their cross-attention in the cached KV is
already right. What blending has to undo is the position shift and the
unrelated preceding contexts. Expect smaller gaps than `independent` and
report the two separately.

### Expected reuse

`build_plan` computes, for every scenario, each question's
`expected_reused_tokens`: the tokens of its rendered request covered by a
whole chunk of a rendered store request, found at any offset. This mirrors
the blend server's matcher; chunks the candidate caches from earlier measured
prompts are not counted. `run` prints the total, and `analyze` compares it
with what the server reported (Section 3).

### Reuse manifest (`manifest.jsonl`)

One record per question: the prompt hash, scenario, chunk size, and per
passage: `passage_id` (content hash), `origin` (`alone` or
`with_sample_passages`), the store request id, `stored_position`,
`measured_position`, `is_evidence`, and `measured_earlier_in` (see below).

Evidence comes from the dataset when it marks it (MuSiQue `is_supporting`,
HotpotQA `supporting_facts`, `context_metadata[].gold`), otherwise from
whether the passage contains a gold answer string. `evidence_source` records
which (`dataset`, `answer_string`, `none`), so loss can later be split by
"was the moved passage the one that mattered".

### Reuse from earlier measured questions

For a candidate served through the LMCache blend server, the store request is
not always where its KV for a passage comes from. The candidate caches every
measured prompt too, and the blend server registers each stored chunk in a
fingerprint index keyed by the chunk's **content** (`BlendTokenRangeMatcher`
in `lmcache/v1/multiprocess/modules/blend/matcher.py`). Chunk alignment makes
a passage hash the same in every prompt, so when a passage recurs in a later
question, which copy is reused depends on the server's registration policy:

| Blend server | Policy | Copy reused by a later question |
|---|---|---|
| default (`enable_dedup_content=False`) | newest registration overwrites the content's table slot | the **latest earlier measured prompt** holding the passage |
| `--enable-dedup-content` | content already indexed is skipped | the **first** copy, i.e. the store request, as the manifest assumes |

Under the default, the reused copy is itself a blend product (KV reused and
partly recomputed in another question's context), so a later question can
measure blending of already-blended KV rather than the configured scenario,
and its `store_request_id` is not where its KV came from.

How often this can happen is a property of the dataset, known before any
request: per passage, `measured_earlier_in` lists the earlier measured
questions that also held it, and `summary.json` reports
`measured_earlier_share`, the share of measured passages with any. Datasets
whose questions draw on their own paragraph sets (HotpotQA, MuSiQue) recur
rarely. Those retrieved from a shared corpus (FinanceBench, RAGAS hand-offs
from a real retriever) recur often.

**Potential fix:** run the blend server with `--enable-dedup-content` for
this bench. Every store request runs before the first question, so the store
copy is then the first registration of each passage, and stays the source
until it is evicted (which `stale_chunks` reports). The bench cannot detect
the flag, so this would be a documented requirement.

**Open question:** is dedup mode representative of production serving? If
production runs newest-wins, that is also how production reuses recurring
documents, and requiring dedup mode would measure a configuration nobody
serves.

## 3. Engagement (blend-server candidates)

**Reuse report.** With captured counters, `analyze` also reports the
candidate's reuse as observed against the plan's expected reuse
(`summary.json` `reuse`): observed prefix and blended tokens, and their
total over `expected_reused_tokens`. It classifies each question by which
kind of reuse served more of it (`prefix`, `blended` or `none`, from its
counter deltas) and reports Δ and SE per kind. In `independent` a question
mixes both (its system prompt and first passage are a prefix). In
`concatenated` the stored units carry no system prompt, so reuse from a unit
is blended; a `prefix` group there comes from an earlier measured prompt
that shares the question's leading passages, or from the blend server's
reorder (`--enable-blend-reorder`) serving a match as a prefix copy. A
per-question kind can lag by one question, like the per-row blend share.

A candidate served through the LMCache blend server reuses cached KV at new
positions; one that silently fell back to recompute would score like the
reference and `PASS`. When `--capture-blend-metrics` is given (it needs
`--candidate-url`), the runner reads the blend server's Prometheus counters
before and after each of the candidate's measured questions:

| Counter | Meaning |
|---|---|
| `lmcache_blend_lookup_requested_tokens_total` | prompt tokens looked up |
| `lmcache_blend_lookup_prefix_hit_tokens_total` | served as an exact prefix |
| `lmcache_blend_lookup_non_prefix_hit_tokens_total` | served by blending at a new position |
| `lmcache_blend_lookup_segmented_prefix_hit_tokens_total` | prefix chunks past an L2 load gap, loaded at their original positions (`--enable-segmented-prefix` only) |
| `lmcache_blend_lookup_stale_chunks_total` | fingerprint matched, KV gone |
| `lmcache_blend_fingerprints_registered_total` | chunks indexed (cap 2^20 per server lifetime) |

**Blend share** = non-prefix hit tokens / requested tokens. The verdict is
`INVALID` if the run-wide share, or the share in any window of 10 consecutive
questions, is below `--min-blend-share` (default 0.3). The windowed check
catches a run that blended early and then stopped (e.g. the fingerprint table
filled). Without a metrics URL, engagement is reported as unmeasured and
cannot make the verdict `INVALID`.

**Reuse gate (`concatenated`).** The blend share is the wrong gate for
`concatenated`. Its expected reuse can legitimately arrive as a prefix (see
the reuse report above), and how much of a prompt is reusable at all depends
on the dataset: a FinanceBench run that reused 0.97 of what the plan
expected had a blend share of 0.38 and windows near 0.2. So for that
scenario the gate is *observed over expected reuse*: prefix, segmented-prefix
and blended tokens over the plan's `expected_reused_tokens`, run-wide and in
every 10-question window (windows that expect no reuse are skipped), must
reach `--min-reuse-ratio` (default 0.8). The margin below 1 absorbs counters
that lag a question and a few evicted chunks. A recompute fallback reuses
nothing and lands near 0. Reuse above the expectation is not penalized: an
earlier measured prompt can add prefix hits the plan does not count. A plan
that expects no reuse at all is `INVALID`, since it measures nothing. The
blend share is still reported, marked as not gated, and the `INVALID` reason
names stale chunks when there are any (matched but evicted, which points at
the cache size rather than at blending). `independent` and `reordered` keep
the blend-share gate.

Segmented-prefix tokens stay outside the share's numerator: they are reused
at their original positions, not blended. They still count in the
denominator, so L2 load failures lower the share. Reported separately (and
named in the `INVALID` reason when present), they distinguish "storage lost
part of the prefix" from "blending did not fire". A passage that moves back
to its stored position in `reordered` is unaffected: it follows different
content than when stored, so it is found by the fingerprint index and counts
as a non-prefix (blended) hit.

The server may publish a lookup slightly after its response completes, so
the pass's opening and closing reads each wait a short settle delay. The
opening wait keeps late store-phase lookups, mostly misses, from diluting the
measured share. Per-row shares are read without waiting and can lag by one
question.

A failed counter read aborts the `run`, so the reads are guarded twice. The
endpoint is read once before any arm runs: otherwise a bad
`--capture-blend-metrics` would surface only when the candidate pass starts,
after every other pass in the same `run`. Each read also retries a failed
fetch (3 attempts, 1 s apart), so one flaky scrape does not abort an
hours-long run. Passes that finished before an abort are already saved
(Section 6).

## 4. Shared modules

`bench quality` builds on the answer-quality helpers in
`engine_bench/quality/`, which it shares with `bench engine --workload
rag-qa-quality`. Sharing them means both benchmarks build byte-identical
prompts and score answers the same way, so their results are comparable:

* `alignment.py` — `ChunkAligner`: pads the system block so chat-template
  prefix + system block is a whole number of chunks, and pads each passage to
  whole chunks with the chosen `Filler` (Section 2, "Padding"), and checks
  placements in rendered prompts (`place_blocks`). `rag-qa-quality` keeps
  the `words` filler.
* `prompts.py` — the QA system prompt and question template
  (`ANSWER_TAG_PROMPT` for `bench quality`; `rag-qa-quality` keeps its
  `<final_answer>` version), and message composition for measured and store
  requests.
* `dataset.py` — `load_samples` and the hub dataset registry. A `Sample`
  carries the dataset's evidence markers (`evidence_indices`), and the
  RAGAS hand-off shape (`user_input`, `retrieved_contexts`, `reference`)
  loads directly.
* `scoring.py` — answer extraction, `answer_match`, `is_abstention` and
  token F1 (Section 5).

## 5. Scoring and verdict

**Metrics.** Two kinds, for different jobs.

**The reference metric is an LLM judge**: `--judge-url <endpoint>
--judge-model <name>`, scoring each extracted answer with RAGAS
`AnswerAccuracy` (0–1 in steps of 0.25). It is the default whenever a judge
endpoint is given. Its scores are the numbers to report, because the
lexical scorers understate the size of a loss (below).

`judge.py` reimplements the metric instead of depending on ragas, which
would pull in langchain, `datasets` and a dozen other packages. The pieces
are copied directly from ragas 0.4.3 (`ragas/metrics/_nv_metrics.py`, Apache-2.0):

- **Two prompts.** The answer is rated against the reference, then with the
  roles swapped, each on a 0/2/4 scale. A test pins both templates to
  ragas's by SHA-256.
- **Combining.** The score is the mean of the two ratings divided by 4. If
  one prompt never yields a rating, the other is used alone.
- **Parsing.** The smallest digit 0–4 anywhere in the reply wins.
- **Calls.** Temperature is 0.10 on every call, whatever the judge client
  says. Each prompt is retried up to 5 times while the reply has no rating.

**Operating rules:**

- **Endpoint.** The judge should be a separate deployment from the arms
  under test.
- **Judge sanity check.** Before scoring, the judge rates a right and a
  wrong answer to a sanity-check question on both prompts and must give 4 and 0.
  Reachability alone is not enough: under the ragas parser, a thinking model
  served without a reasoning parser has digits from its reasoning read as
  the rating, almost always 0. That zeroes every arm alike, Δ vanishes, and
  the verdict is a silent `PASS`. The sanity check goes through the same retries
  and parser as real answers and is never cached, so every run re-checks.
- **Request failures.** Transient errors (connection, 408/409/429, 5xx) are
  retried by the OpenAI client, 5 times with backoff, with a 120 s
  per-request timeout. They do not consume the 5 rating retries, which are
  resamples, not recovery. A request that still fails aborts scoring: the
  judge is down, and marking answers unrated would only shrink the
  comparison. The arms' answers are already on disk, and the judge cache
  keeps finished judgments, so rerunning `analyze` judges only the rest.
- **What is judged.** The reference is the question's first gold answer.
  Unparsed answers score 0 without a judge call.
- **Caching.** Identical (question, answer, reference) triples are judged
  once, and judgments are cached in the results directory's
  `judge_cache.jsonl` for later `analyze` runs.
- **Unrated questions.** A question any arm has no rating for is left out
  of the comparison and counted as `unscored_questions`. Dropping it from
  every arm alike costs only power while the gaps are random. Above 5%
  (`MAX_UNSCORED_SHARE`) the judge is malfunctioning, and the gaps are likely
  to track the answers (e.g. long candidate answers failing to parse), so the
  verdict is `INVALID`.
- **Fingerprint.** The judge model and the metric go into the run
  fingerprint, so results from different judges are never compared.

**Parity with ragas.** The port was checked against ragas 0.4.3 itself by
judging the same answers with both, against the same judge endpoint:

- **Port and ragas are interchangeable.** They send byte-identical requests
  (checked offline against a recording server), so their scores are draws
  from the same distribution, and where they differ neither one is the odd
  one out more often than chance predicts.
- **The judge is the noise, not the port.** At temperature 0.10 a judge
  disagrees with itself on a few percent of answers, almost always by one
  0.25 step. Port-vs-ragas disagreement is at that same level.
- **Comparisons carry over.** Arm-vs-arm score differences computed with
  the port match those computed with ragas to within that judge noise.

The check needs a GPU judge endpoint, so it is not part of the unit tests.

**Without a judge, the lexical scorers are a screen.** Use them for CI and
quick checks, not for reported numbers.

`--verdict-metric match` (default without a judge): the answer states a gold
answer — normalized phrase match, or every gold number appears in the answer
within 0.5%, also under a unit rescale (a power of 1000) when either side
names a scale ("million", "5B"). So `$11,588 million` and `$11.588 billion`
match `$11588.00`, while a count of `3` does not match `3,000`. Bare years
are context and never count as figures, digits inside words are not
numbers, and typographic punctuation (curly quotes, non-breaking hyphens) is
folded to ASCII first. An answer that abstains never matches unless the gold
is itself a negative ("There are none"). `--verdict-metric f1`: SQuAD token
F1, kept for continuity with `rag-qa-quality`.

Compared with LLM-judge scores on the same answers, `match` agrees with the
judge on most answers and tends to reach the same PASS/FAIL. But it
understates the size of a loss, and misses many correct answers whose gold
is a sentence rather than a name or figure. Token F1 is weaker still: it can
pass a clear regression outright. Quote magnitudes from the judge.

**Verdict.** Δ = mean(candidate) − mean(reference), where the reference is the
first reference run. SE is the standard error of the paired per-question
differences.

| Verdict | Condition |
|---|---|
| `INVALID` | blend metrics captured and blend share below `--min-blend-share` (overall or any window); for `concatenated`, observed over expected reuse below `--min-reuse-ratio` instead / an arm run with no successful request / a failed store request / more than 5% of questions unscored. `invalid_reasons` lists which |
| `PASS` | \|Δ\| ≤ `--tolerance` (default 0.05) |
| `FAIL` | \|Δ\| > tolerance **and** \|Δ\| > 2·SE |
| `UNDERPOWERED` | otherwise. Reports `samples_needed` = ⌈(2·sd/\|Δ\|)²⌉ |

The comparison is two-sided. A change meant to preserve answers (such as
reusing KV) should not improve them, so a candidate far above the reference points at the setup (e.g. a reference
that is not really recomputing) as much as a loss does, and fails the same
way.

`INVALID` checks run before the scores are compared because `RequestSender`
turns a failed request into an empty answer scored `0`: with both arms
unreachable, every score is `0`, Δ is `0`, and the run would otherwise
`PASS`. A failed store request means the arm did not cache what the
scenario says, so its answers do not measure that scenario.

Later reference runs and the `control` arm, if present, are compared with the
first reference the same way and reported alongside. They do not set the
verdict. The flip rate is the share of questions whose score differs between
the first two reference runs.

**Loss tags.** Each question where the candidate scores below the reference is tagged
`abstention` (the model said the passages lack the answer), `empty` (no
parsed answer), `wrong_value` (it answered with a string that appears in the
prompt, i.e. read the wrong passage), or `wording` (anything else).
`candidate_higher` and `none` cover the other rows.

## 6. Stages and the results directory

**Why two stages.** A single invocation needed every arm's engine, the blend
server's metrics (for a blend-server candidate) and the judge reachable for
the whole run. On a GPU pool that is several GPU servers at once, although the
arms only ever run one after another. `run` and `analyze` are separate
commands that communicate only through files, so each engine (and the judge)
can be up alone, each in its own CI job, and saved answers can be re-analyzed
with another metric or tolerance. A long-lived process that paused between
arms would save the engines' GPUs too, but not the judge's, and would allow
neither re-analysis nor reuse of a saved reference.

**The directory** (`--output-dir`, used as given; omitted, `run` creates
`quality_results/<YYYYmmdd-HHMMSS>/`, with `-2`, `-3`, … on a same-second
collision). Every JSON write except the append-only judge cache is atomic
(temporary file, then `os.replace`) and strict JSON (`allow_nan=False`; an
undefined standard error is written as `null`), so any JSON parser reads the
results. The field-by-field reference -- types, units, sentinels, which keys
are conditional -- is the user docs page
`docs/source/cli/bench_quality_results.rst`; the table below is the overview.

| File | Written by | Contents |
|---|---|---|
| `plan.json` | first `run` | `schema_version`, `plan_fingerprint`, `prompts_digest`, the lmcache version, the workload settings, and the whole plan: store requests and, per question, its manifest entry plus question, gold answers, passages and measured messages |
| `manifest.jsonl` | first `run` | reuse manifest (Section 2) |
| `arm_<label>.json` | `run` | per pass: role, repeat, URL, metrics URL, engine version (best-effort `GET /version`), lmcache version, start and finish times, run-level blend counters (when captured), store failures, `plan_fingerprint` |
| `answers_<label>.jsonl` | `run` | per question: response, parsed answer, TTFT, output tokens, counter delta |
| `rows.jsonl` | `analyze` | per question: gold, score per arm, answers, loss tag, reference flipped, blend share, evidence; with the judge, also each arm's `match` score as a cross-check |
| `summary.json` | `analyze` | verdict and `invalid_reasons`, metric, tolerance, Δ, SE, flip rate, engagement, per-arm counts, each arm's provenance, evidence sources (`dataset` / `answer_string` / `none`), `measured_earlier_share`, `padding`, judge model and unscored questions, `plan_fingerprint`, `run_fingerprint` |
| `judge_cache.jsonl` | `analyze` | judge ratings by (judge model, question, answer, reference), reused by later `analyze` runs. An unparsable line (a partial append from a killed run) is skipped and re-judged, and terminated so the next append starts a fresh line. The key trusts the judge model name; change what serves it and the cache must be deleted |

**What keeps the files consistent.**

- **One plan per directory.** `run` always builds the plan from its own flags
  (it needs the tokenizer anyway) and compares it with `plan.json` before any
  request. Other settings are refused, listing each changed key; the same
  settings with other prompts (a tokenizer, dataset revision or prompt-code
  change) are refused too. `analyze` refuses any arm whose `plan_fingerprint`
  differs, and re-checks `plan.json` against its own digests.
- **Passes are saved as they finish**, answers first and the arm file last,
  so the arm file marks a complete pass; answers without one are ignored.
  A crash in a later pass keeps the earlier ones.
- **A role comes from one `run`.** The first new pass of a role replaces that
  role's earlier files (and the now-stale `rows.jsonl` and `summary.json`),
  so reference repeats never mix engine lifetimes and fewer repeats never
  leave a stale one behind.
- **Schema versions.** `plan.json` and every arm file record
  `schema_version`; another version is refused rather than misread.

Every refusal above, and an `analyze` without `plan.json` or without the
reference or candidate arm, raises `ResultsDirError`. Both commands report it
without a traceback and exit `2`, like an invalid invocation, rather than
`1`: it is a setup problem, not a quality result, and CI can tell the two
apart by exit code alone. A missing-arms message names every missing arm
and one `run` command that supplies them all.

**Reusing a reference.** Copy `plan.json`, `manifest.jsonl` and the
`*reference*` files (optionally `judge_cache.jsonl`) into a fresh directory,
run only the candidate arm into it, and analyze. The plan check refuses the
candidate `run` if anything that shapes the prompts changed. Whether the *reference
stack* is unchanged (engine image, model revision) is the caller's to decide;
the recorded engine and lmcache versions and timestamps make it visible. A
reused arm keeps the answers as its own lmcache version extracted them.

**Fingerprints.** `plan_fingerprint` covers the workload settings and every
prompt; it ties arms to a plan. `run_fingerprint` adds the verdict metric and
the judge model; two summaries are comparable only when it matches.

## 7. Module layout

```text
lmcache/cli/commands/bench/
├── engine_bench/quality/     # shared with rag-qa-quality (Section 4)
│   ├── alignment.py        # ChunkAligner: chunk-aligned padding
│   ├── prompts.py          # QA prompt text and composition
│   ├── dataset.py          # load_samples, Sample (with evidence_indices)
│   └── scoring.py          # answer extraction, answer_match, is_abstention, F1
└── quality_bench/
    ├── __init__.py         # QualityBenchCommand: the "lmcache bench quality" group
    ├── run_command.py      # "run" leaf: defers to core.run_stage
    ├── analyze_command.py  # "analyze" leaf: defers to core.analyze_stage
    └── core/               # runtime, kept out of command discovery's imports
        ├── config.py       # roles, scenarios, QualityRunConfig, QualityAnalyzeConfig
        ├── task.py         # check_task: refuse datasets that are not short-answer QA
        ├── plan.py         # build_plan (all scenarios), expected reuse, manifest, plan records
        ├── blend_counters.py   # Prometheus counter parsing and sources
        ├── runner.py       # ArmRunner: store, measure, per-question counters
        ├── run_files.py    # the results directory: plan.json, arm files, loaders
        ├── run_stage.py    # run: arguments, plan, arms, saving each pass
        ├── judge.py        # AnswerAccuracyJudge: RAGAS AnswerAccuracy port, cache
        ├── verdict.py      # score tables, engagement, decide, rows, loss tags
        ├── report.py       # rows.jsonl, summary.json, terminal summary
        └── analyze_stage.py    # analyze: load, score, decide, report
```

The command wrappers stay tiny because `CompositeCommand` discovery imports
every module directly under `quality_bench/`; the runtime sits in `core/`, so
`lmcache bench quality -h` imports none of it, `run -h` only `run_stage`'s
side and `analyze -h` only `analyze_stage`'s (a routing test checks this).

## 8. Tests

`tests/cli/commands/bench/quality_bench/` runs without a GPU or network:
`FakeSender` stands in for `RequestSender`, `SteppingCounters` for the
Prometheus source, and the fake tokenizer from `engine_bench` for the model's.
`FakeJudgeClient` stands in for the judge endpoint.
- `test_run_stage.py` drives `run`: flag rules, the files a run writes,
  arms joining one directory, refusal of other settings or drifted prompts
  before any request, role replacement, the timestamped default directory,
  and that a crash in a later pass keeps the earlier ones.
- `test_analyze_stage.py` drives `analyze` end to end for `PASS`, `FAIL` and
  `INVALID`, staged and all-at-once, with and without the judge; checks it
  needs no dataset or tokenizer, re-analyzes without requests, refuses
  missing arms and arms from another plan, and reuses a saved reference.
- `test_run_files.py` covers the results-directory contract: plan and arm
  round trips, tamper detection, `ensure_plan`'s refusals, run-directory
  naming.
- `test_judge.py` pins the prompts to ragas's by hash, and covers ragas's
  parsing quirks, retries, the sanity check, caching (including a truncated cache
  file) and deduplication.
- `tests/cli/commands/test_directed_registration.py` checks that each
  `bench quality` route imports only its own stage.

## 9. Extending beyond QA

**Status:** design direction. Implemented so far: the task type with its
refusal (Section 9.2, item 1), the judge scorer (Section 5), and evidence coverage in the
summary (Section 9.4). Everything else here is planned.

The prototype measures QA only. The next step is other task categories
(summarization, code, long-form), and datasets that carry no evidence
markers. Most of the pipeline already doesn't depend on the task. The parts
that do are listed here, with how each should generalize.

### 9.1 What depends on the task

| Piece | Today | Task-specific? |
|---|---|---|
| Dataset loading (`load_samples`) | Any record with passages, a prompt and a reference | No. MultiNews loads unchanged. |
| Alignment, reuse scenarios, reuse manifest | Operate on passages | No |
| Runner, engagement counters | Operate on requests | No |
| Verdict statistics (paired Δ, SE, PASS/FAIL/UNDERPOWERED) | Operate on a per-question score in [0, 1] | No |
| Prompt (`QAPrompt`: system text, question template) | "Answer the question", `<answer>` tags | **Yes** |
| Answer extraction (`extract_final_answer`) | Last `<answer>` / `<final_answer>` region | **Yes** |
| Scorer (`answer_match`, token F1) | Short-answer matching | **Yes** |
| Loss tags (`abstention`, `wrong_value`) | Short-answer heuristics | **Yes** |
| Evidence fallback (gold string in a passage) | Short golds only | **Yes** |

Without a guard, a summarization file would load, be sent the QA prompt,
and be scored with `match` against a multi-paragraph gold that no answer
ever contains. Every arm would score ~0 and the verdict would be a
meaningless PASS. `task.check_task` now refuses such a dataset before any
request is sent (Section 9.2).

### 9.2 Task as an explicit unit

1. **Every run declares a task type** *(implemented)*. `--task` takes
   `qa`, the only type so far. `check_task` refuses a QA run whose
   reference answers have a median of more than 50 words.
   - Measured medians: 2 words for HotpotQA, 2Wiki and MuSiQue, 9 for
     FinanceBench, 200 for MultiNews.
   - So a summarization file fails with an error instead of producing a
     meaningless PASS.
   - When a second task type is added, hub registry entries can declare
     their type and the flag can default to it.
2. **One task object owns the task-specific pieces.** It holds the prompt
   builder, answer extraction, scorer, loss classifier and evidence locator.
   The rest of the pipeline sees only a score in [0, 1] and a tag per
   question. The verdict already works on continuous scores, so judge
   scores, ROUGE-L or pass@1 all fit unchanged.
3. **QA stays the reference implementation.** Today's functions become the
   QA task's members. Abstention and `wrong_value` tags stay inside it.
   `is_abstention` is calibrated on short answers and flags normal news
   summaries as abstentions.

### 9.3 Scoring non-QA tasks needs a judge

Lexical scores don't see regressions on long outputs. On multi-document
summarization (e.g. MultiNews), a judge can register a clear loss of
accuracy and groundedness while token F1 and ROUGE-L stay flat: a long
summary shares most of its tokens with the reference whether or not it is
faithful to the passages.

Non-QA tasks therefore need an LLM-judge scorer from the start. The QA
judge (Section 5) already covers most of what that needs:

- **A fixed judge** *(implemented)*. The model is recorded in the run
  fingerprint.
- **Cached judgments** *(implemented)*. Re-scoring or adding an arm doesn't
  re-judge unchanged answers.
- **A judge noise estimate** *(planned)*. This is the analog of the
  reference flip rate: re-judge a subset, or compare with a second judge
  from a different model family.

A summarization task would add a groundedness-style judge prompt and
judge-derived loss tags: omission (accuracy drops with groundedness intact)
and unsupported content (groundedness drops).

`match` remains useful as a cheap, deterministic QA gate (Section 5).

### 9.4 Datasets without evidence markers

Evidence markers are the exception. Of the common multi-passage datasets
below, only HotpotQA ships them. The hub copy of MuSiQue (`--dataset
musique`) also has them, but the LongBench copy does not. The plan records
each question's `evidence_source` (`dataset`, `answer_string` or `none`), so
missing markers never break a run. But the evidence split quietly covers
fewer questions. Measured on samples of the public datasets (sample size in
parentheses):

| Dataset | Dataset markers | Gold string found in a passage | No evidence |
|---|---|---|---|
| HotpotQA (100) | 100 | – | 0 |
| FinanceBench (150) | 0 | 6 | 144 |
| 2WikiMultihopQA (200) | 0 | 178 | 22 |
| MuSiQue, LongBench copy (200) | 0 | 181 | 19 |
| MultiNews (200) | 0 | 0 | 200 |

- **Coverage is reported** *(implemented)*. `summary.json` and the terminal
  summary show the question count per `evidence_source`. An analysis built
  on 6 of 150 questions then says so, rather than reading as "no losses on
  evidence passages".

**Planned:**

- **Let the task locate evidence.**
  - *QA:* use number-aware matching. Using `answer_match`'s number rules,
    without its abstention check, which misreads passages, finds evidence
    for 62 of 150 FinanceBench questions instead of 6. On HotpotQA, the
    passages it finds overlap the dataset's markers on 94 of 100 questions.
  - *Summarization:* every passage is evidence, so the useful split is by
    how a passage was reused (blended or freshly computed), which the
    manifest already records.
- **Never require markers.** A task without an evidence locator reports
  `none` and skips the split.

### 9.5 Invariants

Arms, reuse scenarios, engagement checks and the verdict rule carry over
unchanged. Multi-document summarization maps directly onto `independent`
(each article cached alone) and `reordered`. A new task adds a task object
and a dataset entry. It doesn't touch the plan, runner or report.

## 10. Next steps

1. Summarization as the second task type, scored by the judge (Section 9.2, Section 9.3).
2. Number-aware evidence location for QA (Section 9.4).
3. More placements: shared prefix + reused passages, worst-case position.
4. Engine lifecycle: clear or restart caches between arms
   (`lmcache kvcache clear`) so a run needs no manual restart.
5. Sweep mode: `recomp_ratio` / `check_layer` grid with one verdict per point.
6. Reuse from earlier measured questions (Section 2): settle the open question of
   whether to require `--enable-dedup-content` on the blend server.
7. `concatenated` with `--enable-blend-reorder`: confirm on a pilot that a
   question's context, a strict part of its unit, is still blended rather
   than copied as a prefix (the reuse report's `prefix` group should be
   empty).
