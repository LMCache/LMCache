---
name: bench-quality
description: Run `lmcache bench quality` to check whether a candidate serving stack (quantization, KV cache reuse such as CacheBlend, KV compression, post-training, another engine) changes answers compared with a reference, then report the verdict to the user in the standard format below. Use when asked to measure, compare or gate answer quality or accuracy loss, or to run, analyze or explain a bench quality results directory.
allowed-tools: Bash, Read, Grep, Glob, Write
argument-hint: "[RESULTS_DIR] | --reference-url URL --candidate-url URL --model M --dataset D [--scenario S] [--judge-url URL]"
---

# bench quality: run, analyze, report

`lmcache bench quality` sends the same QA prompts to a **reference** engine and a
**candidate** engine, scores both against gold answers, and prints one verdict:
`PASS`, `FAIL`, `UNDERPOWERED` or `INVALID`. It has two stages that share a results
directory: `run` (needs the engines) and `analyze` (needs only the files, plus a
judge endpoint if one is used).

This skill covers picking the flags, running both stages, checking that the result
can be trusted, and **reporting it in the standard format (step 6)**. It does not
launch or restart engines. If an engine the run needs is not up, give the user the
launch command (step 1) and stop.

Look up anything not covered here in these files rather than guessing:

- `docs/source/cli/bench.rst`, the "Quality" sections: flags, scenarios, verdict, exit codes.
- `docs/source/cli/bench_quality_results.rst`: every field of every results file.
- `docs/design/cli/commands/bench/quality_bench/bench-quality.md`: why things work as they do.

## 0. Pick the entry point

| What the user gave you | Start at |
|---|---|
| A results directory containing `summary.json` | step 5 (read), then step 6 (report) |
| A results directory with arms but no `summary.json` | step 4 (analyze) |
| Engine URLs, or a description of what to compare | step 1 |

A results directory holds **one plan**, so one scenario, one dataset and one model.
To compare scenarios, use one directory per scenario and write one report for each.

## 1. Decide the comparison

Every arm is sent the same `--model` name. To compare two different models, serve the
candidate under the reference's name (vLLM: `--served-model-name`).

| Candidate under test | Reference | Candidate engine | Extra `run` flags |
|---|---|---|---|
| Quantized / post-trained / other model | Original model, vLLM, `--no-enable-prefix-caching` | Candidate model, `--served-model-name <reference name>` | none |
| LMCache blend server (CacheBlend) | Same model, vLLM, `--no-enable-prefix-caching` | vLLM + `LMCacheMPConnector` to `lmcache server --engine-type blend` | `--capture-blend-metrics http://<server>:<prometheus-port>/metrics`, `--chunk-size <server chunk size>` |
| Another KV cache optimization | Same model, vLLM, prefix caching off | The stack under test | `--chunk-size` if it caches by chunk |

- **Control arm (optional):** `--control-url`, pointing at the reference with prefix
  caching **on**. It should match the reference exactly, so a gap points at the
  harness or the engine. Add it when a surprising result needs ruling out.
- **Scenario** (`--scenario`): `independent` (default; passages cached alone, then
  combined), `reordered` (cached together, measured in a new order), `concatenated`
  (whole contexts cached after unrelated ones). For a blend candidate, run at least
  `independent`. Report scenarios separately; never average them.
- **Size:** `--limit 50` for a smoke test. Use `--limit 200` or more for a number you
  report. Keep `--reference-repeats 2` (the default) so a flip rate is reported.
- **Dataset:** `hotpotqa`, `musique`, or a local `.json`/`.jsonl`/`.parquet` QA file.
  Only short-answer QA is supported; the run refuses summarization sets.
- **Reasoning models:** pass `--template-kwargs reasoning_effort=low` (or the model's
  equivalent), and leave `--max-output-tokens` (default 2048) room for the reasoning.
- **Blend server settings** are not visible to the bench, so record the ones that change
  what is measured: `--enable-dedup-content`, `--enable-blend-reorder`,
  `--enable-segmented-prefix`, and the chunk size. Ask the user if you do not know them.

## 2. Preflight (before spending GPU time)

```bash
python -c "import transformers"                     # `run` needs it; analyze does not
for URL in "$REF_URL" "$CAND_URL"; do                # and $CTRL_URL if used
  curl -sf "$URL/v1/models" | jq -r '.data[].id'     # must list the --model name
done
curl -sf "$BLEND_METRICS_URL" | grep -c '^lmcache_blend_lookup_requested_tokens_total'   # blend candidate: >= 1
```

- **The candidate must start from an empty cache.** Its engine, and any cache server
  behind it, must have been restarted since it last served these prompts. If you did
  not see the restart happen, ask the user. Record the answer: the report has a row
  for it.
- **The judge** should be a separate deployment from the arms. A thinking model must
  be served with a reasoning parser (vLLM `--reasoning-parser`). `analyze` runs a
  sanity check and stops if the judge fails it.
- **Choose the output directory** explicitly, so every stage and the report agree:
  `OUT=quality_results/<candidate-slug>-<scenario>-$(date +%Y%m%d-%H%M)`.

## 3. Run

Put the workload flags in one variable. **Every `run` into a directory must use
identical workload flags** (`--model` through `--seed`); a different plan is refused
with exit 2. Save every command you run. The report's Reproduce section uses them.

```bash
WORKLOAD="--model Qwen/Qwen3-8B --dataset hotpotqa --limit 200 --scenario independent"
mkdir -p "$OUT"

# All engines up at once:
lmcache bench quality run --reference-url "$REF_URL" --candidate-url "$CAND_URL" \
    $WORKLOAD --output-dir "$OUT" 2>&1 | tee -a "$OUT/run.log"

# Or one engine at a time (same $WORKLOAD and $OUT):
lmcache bench quality run --reference-url "$REF_URL" $WORKLOAD --output-dir "$OUT" 2>&1 | tee -a "$OUT/run.log"
lmcache bench quality run --candidate-url "$CAND_URL" $WORKLOAD --output-dir "$OUT" \
    --capture-blend-metrics "$BLEND_METRICS_URL" --chunk-size 256 2>&1 | tee -a "$OUT/run.log"
```

A run can take hours, so start it in the background (`nohup … &`) and poll
`run.log`. Check the exit code (`${PIPESTATUS[0]}` after a `tee` pipeline):

- `0`: the passes are fit to compare.
- `1`: a pass had no successful request or a failed store request. The verdict will
  be `INVALID`. Check the engine and rerun that arm (a rerun replaces the role's files).
- `2`: bad flags, or the plan differs from the directory's `plan.json`. Read the
  listed keys. Fix the flags, or use a new `$OUT`. Never delete `plan.json` to force
  the run through.

To reuse a saved reference, copy `plan.json`, `manifest.jsonl` and `*reference*` from
the old directory into a new `$OUT` and run only the candidate.

## 4. Analyze

```bash
lmcache bench quality analyze "$OUT" \
    --judge-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL" 2>&1 | tee "$OUT/analyze.log"
```

- With a judge, the metric is `judge` (RAGAS `AnswerAccuracy`). **Use it for any number
  you report.** Without one, the metric is `match`. That is a screen for CI and smoke
  tests, and it understates how large a loss is.
- Exit codes: `0` = `PASS` or `UNDERPOWERED`; `1` = `FAIL` or `INVALID`, or an error
  such as the judge being unreachable; `2` = the directory is incomplete or does not
  fit, and the message names what is missing.
- If the judge fails partway, rerun the same command. Finished judgments are cached
  in `judge_cache.jsonl`.

## 5. Read the results: validity first

Read `summary.json`. Do not parse the terminal output. Go through the checks below **in
order, before looking at Δ**, and give each one a status: `OK`, `WARN` or `BAD`.

```bash
jq '{verdict, invalid_reasons, metric: .settings.verdict_metric,
     store_failures, empty: (.arm_counts | map_values(.empty)),
     unscored: .unscored_questions, n: .num_questions,
     engagement, reuse_ratio: .reuse.observed_over_expected,
     flip_rate: .reference_flip_rate, others,
     measured_earlier_share, evidence_sources}' "$OUT/summary.json"
```

| Check | OK | WARN | BAD |
|---|---|---|---|
| `invalid_reasons` | `[]` | | not empty (verdict is `INVALID`) |
| `store_failures` | all 0 | | any > 0 |
| Empty answers (`arm_counts.*.empty`) | ≤ 2% of `num_questions` | > 2%: answers cut off; raise `--max-output-tokens` | an arm mostly empty |
| `unscored_questions` (judge only) | 0 | ≤ 5% | > 5% (verdict is `INVALID`) |
| Engagement, for a caching candidate | `measured` and `engaged` true | `measured` false: a silent fallback to recompute would `PASS` | `engaged` false |
| Candidate started from an empty cache | confirmed | unconfirmed | known not to |
| `reference_flip_rate` | ≤ 0.10 | > 0.10: noisy reference; more questions needed | |
| Control (`others[]`, label `control`) | \|Δ\| ≤ tolerance | \|Δ\| > tolerance: harness or engine issue | |
| `measured_earlier_share` | ≤ 0.10 | > 0.10: later questions may reuse KV that was already blended (see the design doc, §2) | |
| Metric | `judge` | `match` / `f1`: screen only | |

## 6. Report: the standard format

Every report has two parts. **`$OUT/report.md`** is the full report. A **chat summary**
of at most 6 lines points to it. The rules below apply to both.

### Rules

1. **Verdict first, exactly as the tool gave it.** Never upgrade a verdict:
   `UNDERPOWERED` is not "roughly a pass", and an `INVALID` run has no headline Δ.
2. **Every number comes from `summary.json` or `rows.jsonl`,** pulled with the snippets
   below. Do not do arithmetic by hand. Scores are on a 0–1 scale, with 3 decimals.
3. **Sign convention:** Δ = candidate − `reference_1`. Negative is a loss.
4. **Δ never appears alone.** Always put `2·SE` and the tolerance next to it.
5. **Name the metric.** With `match` or `f1`, add "(lexical screen; not a reported number)".
6. **Facts and interpretation stay apart.** Explanations, hypotheses and next steps
   go only in the "Notes" section, worded as such.
7. **One results directory gives one report.** If a role is rerun, `summary.json` is
   deleted and the report goes stale. Regenerate the report after every `analyze`. The
   `run_fingerprint` in the report ties it to one summary.

### Verdict sentence

Use the sentence for the verdict, filled from `summary.json`:

| Verdict | Sentence |
|---|---|
| `PASS` | The candidate answers as well as the reference: Δ = {delta}, within ±{tolerance}. |
| `FAIL`, Δ < 0 | The candidate loses answer quality: Δ = {delta}, beyond both ±{tolerance} and 2·SE = {2se}. |
| `FAIL`, Δ > 0 | The candidate scores above the reference (Δ = {delta}, beyond ±{tolerance} and 2·SE = {2se}). A change meant to preserve answers shouldn't do that, so check the reference setup. |
| `UNDERPOWERED` | Inconclusive: Δ = {delta} is outside ±{tolerance} but within noise (2·SE = {2se}). About {samples_needed} questions would settle it. |
| `INVALID` | No verdict: the runs can't be compared because {invalid_reasons joined with "; "}. |

### `report.md` template

Use this structure exactly. Keep the headings and their order. Leave out the "Reuse"
section when `engagement.measured` is false. In tables, replace `|` and newlines in
answer text with spaces, and cut answers to 80 characters.

````markdown
# bench quality: {VERDICT}: {candidate description} vs {reference description}

{verdict sentence}

| | |
|---|---|
| Verdict | **{VERDICT}** |
| Δ (candidate − reference) | {delta} (2·SE {2se}, tolerance ±{tolerance}) |
| Metric | {judge: <judge model> (ragas-0.4.3/nv_accuracy) \| match (lexical screen; not a reported number)} |
| Workload | {model} · {dataset} · {num_questions} questions · scenario {scenario} · chunk {chunk_size} · filler {filler} |
| Results | `{OUT}` · run_fingerprint `{run_fingerprint}` |

## Validity checks

| Check | Value | Status |
|---|---|---|
| Invalid reasons | {none \| reasons} | {OK/BAD} |
| Store failures | {label: n, …} | |
| Empty answers | {label: n, …} | |
| Unscored questions | {n} / {num_questions} | |
| Engagement (gate: {gate}) | {blend_share or reuse_ratio}, min window {…} (min {threshold}) \| not measured | |
| Candidate started from an empty cache | {confirmed \| unconfirmed} | |
| Reference flip rate | {reference_flip_rate} | |
| Control vs reference | {Δ} \| not run | |
| Reuse from earlier questions | {measured_earlier_share} | |

## Scores

| Pass | Mean score | Δ vs reference_1 | 2·SE | Empty | Abstained |
|---|---|---|---|---|---|
{one row per pass from the scores snippet}

## Reuse

Blend share {blend_share} (min window {min_window_share}); stale chunks {stale_chunks};
segmented-prefix tokens {segmented_prefix_hit_tokens}; fingerprints registered {fingerprints_registered}.
Expected reused tokens {reuse.expected_tokens}; observed {reuse.observed_tokens}
(prefix {observed_prefix_tokens}, blended {observed_blended_tokens}); observed / expected {observed_over_expected}.

| Reuse kind | Questions | Δ | SE |
|---|---|---|---|
{one row per reuse.groups entry}

## Where the candidate lost

| Loss tag | Questions |
|---|---|
{loss_tags counts, largest first; `none` and `candidate_higher` last}

| qid | Tag | Gold | Reference answer | Candidate answer |
|---|---|---|---|---|
{up to 5 lost questions from the losses snippet, largest score drop first}

## Setup and provenance

| Pass | URL | Engine | lmcache | Finished |
|---|---|---|---|---|
{one row per arm}

- Blend server flags: {chunk size, --enable-dedup-content, --enable-blend-reorder, --enable-segmented-prefix \| n/a \| unknown}
- Evidence sources: dataset {n}, answer_string {n}, none {n}
- Padding: {padding.filler}, {padding.share} of measured prompt tokens
- plan_fingerprint `{plan_fingerprint}`

## Reproduce

```bash
{every run and analyze command used, verbatim, with WORKLOAD expanded}
```

## Notes

{Your interpretation, hypotheses, caveats and suggested next steps, worded as such.
"None" if you have none.}
````

### Snippets that fill the template

```bash
S="$OUT/summary.json"; R="$OUT/rows.jsonl"
F3='def f3: if . == null then "—" else (. * 1000 | round + 0) as $n
  | (if $n < 0 then -$n else $n end) as $a
  | (if $n < 0 then "-" else "" end) + (($a / 1000 | floor) | tostring)
    + "." + (("00" + (($a % 1000) | tostring))[-3:]) end;'

# Header values
jq -r "$F3"' "verdict=\(.verdict) delta=\(.candidate.delta | f3)
  2se=\(if .candidate.standard_error == null then null else 2 * .candidate.standard_error end | f3)
  tolerance=\(.settings.tolerance) samples_needed=\(.samples_needed)
  metric=\(.settings.verdict_metric) judge=\(.judge.model // "none")
  invalid_reasons=\(.invalid_reasons | join("; "))"' "$S"

# Scores table rows (reference_1 mean = candidate.mean_score - candidate.delta)
jq -r "$F3"' (.candidate.mean_score - .candidate.delta) as $r1 | .arm_counts as $c
  | ([{label: "reference_1", mean_score: $r1}] + .others + [.candidate])[]
  | "| `\(.label)` | \(.mean_score | f3) | \(.delta | f3) | \(if .standard_error == null then null else 2 * .standard_error end | f3) | \($c[.label].empty) | \($c[.label].abstained) |"' "$S"

# Reuse groups (only when .reuse is not null)
jq -r "$F3"' .reuse.groups[]? | "| \(.kind) | \(.questions) | \(.delta | f3) | \(.standard_error | f3) |"' "$S"

# Loss tag counts
jq -r '.loss_tags | to_entries | sort_by(.key == "none" or .key == "candidate_higher", -.value)[]
  | "| \(.key) | \(.value) |"' "$S"

# Up to 5 lost questions, largest drop first
jq -sr 'def cell: tostring | gsub("[|\n]"; " ") | .[:80];
  map(select(.loss_tag as $t | $t != null and $t != "none" and $t != "candidate_higher")
      | . + {drop: (.scores.reference_1 - .scores.candidate)})
  | sort_by(-.drop)[:5][]
  | "| \(.qid) | \(.loss_tag) | \(.gold[0] | cell) | \(.answers.reference_1 | cell) | \(.answers.candidate | cell) |"' "$R"

# Provenance rows
jq -r '.arms | to_entries[] | "| `\(.key)` | \(.value.url) | \(.value.engine_version // "—") | \(.value.lmcache_version) | \(.value.finished_at) |"' "$S"
```

### Chat summary template

Post this after writing `report.md`. Name every `WARN` or `BAD` check on the Checks
line. Say nothing more about the numbers than the report says.

```text
**bench quality: {VERDICT}** · {candidate} vs {reference} · {dataset} ×{num_questions} · {scenario} · {metric}
{verdict sentence}
Scores: candidate {cand mean} vs reference {ref_1 mean} (Δ {delta}, 2·SE {2se}, tolerance ±{tolerance})
Checks: {all OK | WARN/BAD: <check>: <value>; …} · flip rate {flip} · engagement {value | not measured}
Losses: {top loss tags with counts} ({candidate_higher} questions better)
Report: {OUT}/report.md
```

For an `INVALID` verdict, replace the Scores and Losses lines with
`Reasons: {invalid_reasons}` and the fix you suggest (from Troubleshooting).

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `run` exits 2 with a list of changed keys | Workload flags differ from `plan.json`. Use identical `$WORKLOAD`, or a new `$OUT`. |
| `run` exits 2 with misplaced passages | A passage does not land on chunk boundaries for this tokenizer. Keep `--filler dots` (the default); `words` fails on many tokenizers. |
| `INVALID`: blend share below minimum | Blending did not fire. Check the server is `--engine-type blend`, that `--chunk-size` matches it, and that the candidate started from an empty cache. A low *window* means it stopped partway (fingerprint table full, eviction). |
| `INVALID` naming stale chunks | KV was evicted before reuse. Raise the cache size or use a smaller `--limit`. |
| `INVALID` naming segmented-prefix tokens | L2 load gaps, not blending. Check L2 storage health. |
| `INVALID`: no successful request | An engine was unreachable or rejected the `--model` name (`--served-model-name`). |
| Judge sanity check fails | The judge is a thinking model without `--reasoning-parser`, or it is the wrong model. |
| Many `empty` answers | Reasoning used up the budget. Lower the reasoning effort or raise `--max-output-tokens`. |
| Control differs from the reference | A harness or engine nondeterminism problem. Report it in Notes, and don't attribute the gap to the candidate. |
| `summary.json` missing but `report.md` present | A role was rerun after the report was written. Run `analyze` again and regenerate the report. |
