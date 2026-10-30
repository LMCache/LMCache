.. _bench-quality-results:

bench quality result files
==========================

This page describes every file and field that
:ref:`lmcache bench quality <lmcache-bench-quality>` writes to a results
directory: what each file is for, which stage writes it and when, and the
type and meaning of each field. It is meant for anyone -- a person, a
script, or an AI agent -- reading results without the source at hand.

All files are JSON (``.json``, one object) or JSON Lines (``.jsonl``, one
object per line), UTF-8, and strict JSON: no ``Infinity`` or ``NaN``, so any
JSON parser reads them.

.. code-block:: text

   <results dir>/
       plan.json               the plan: settings, prompts, questions, golds
       manifest.jsonl          the reuse manifest, one record per question
       arm_<label>.json        one arm pass: metadata, written last
       answers_<label>.jsonl   that pass's answers, one record per question
       judge_cache.jsonl       judge ratings, reused by later analyzes
       rows.jsonl              per-question comparison, written by analyze
       summary.json            the verdict, written by analyze


Lifecycle of the files
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 16 30 32

   * - File
     - Written by
     - When
     - Replaced or deleted
   * - ``plan.json``, ``manifest.jsonl``
     - ``run``
     - The first ``run`` into the directory. Later runs only compare their
       plan with ``plan.json``.
     - Never rewritten; ``created_at`` and ``lmcache_version`` are the first
       run's.
   * - ``arm_<label>.json``, ``answers_<label>.jsonl``
     - ``run``
     - As each pass finishes: answers first, the arm file last.
     - The first new pass of a role deletes that role's earlier files (and
       ``rows.jsonl`` and ``summary.json``) before it is saved.
   * - ``rows.jsonl``, ``summary.json``
     - ``analyze``
     - Every ``analyze``.
     - Overwritten by the next ``analyze``; deleted when an arm is rerun.
   * - ``judge_cache.jsonl``
     - ``analyze`` with a judge
     - Appended as each new judgment finishes.
     - Never deleted. Delete it yourself after changing what serves the
       judge model name.

Every file except the append-only judge cache is written atomically: to
``<file>.tmp`` first, then renamed into place, so a reader never sees half a
file. A ``.tmp`` left behind by a killed process is never read. An
``answers_<label>.jsonl`` without its ``arm_<label>.json`` is from a pass
that did not finish; ``analyze`` ignores it with a warning.


Conventions used by every file
------------------------------

**Labels.** Each arm pass has a label, used as the file-name suffix and as
the key in per-arm objects: ``reference_1``, ``reference_2``, … for the
reference passes (``--reference-repeats``), ``control`` and ``candidate``.

**Request ids.** Store requests are ``store0``, ``store1``, …; measured
questions are ``q0``, ``q1``, …, in plan order.

**Digests.** All are lowercase hex prefixes of SHA-256:

.. list-table::
   :header-rows: 1
   :widths: 22 10 68

   * - Digest
     - Length
     - Input
   * - ``prompt_hash``
     - 16
     - The question's measured messages (``json.dumps(..., sort_keys=True)``).
   * - ``passage_id``
     - 12
     - The raw passage text, before chunk padding. Equal text, equal id.
   * - ``prompts_digest``
     - 16
     - Every store and measured message, in order.
   * - ``plan_fingerprint``
     - 16
     - The plan settings (see `plan.json`_) and ``prompts_digest``. Ties
       every arm file to one plan.
   * - ``run_fingerprint``
     - 16
     - The plan settings, the verdict metric, the judge model (with the
       judge metric) and ``prompts_digest``. Two summaries are comparable
       only when it matches. Tolerance, thresholds and URLs are not
       included.
   * - judge cache ``key``
     - 64
     - The judge metric id, judge model, question, answer and reference.

**Times** are ISO 8601, to the second, in local time with the UTC offset
(``2026-10-19T10:41:12-05:00``).

**Versions.** ``lmcache_version`` is ``<version>-<commit>`` (or just the
version, or ``"unknown"`` in a source checkout). ``engine_version`` is what
the engine reported at ``GET /version``, or ``null``.

**Schema version.** ``plan.json`` and every ``arm_<label>.json`` carry
``schema_version`` (currently ``1``). ``run`` and ``analyze`` refuse a
directory written with another version (exit 2) rather than misread it. The
other files have no version of their own: they belong to the plan in the
same directory.

**Special values.** Where a value can be missing, the files use ``null`` or
a sentinel:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Field
     - Meaning of the special value
   * - ``ttft`` (answers)
     - ``-1.0``: the request failed.
   * - ``answer`` (answers, rows)
     - ``""``: no complete answer -- the request failed, or the
       ``<answer>`` tags were missing or cut off.
   * - ``counters`` (arm, answers), ``blend_share`` (rows)
     - ``null``: blend metrics were not captured (any arm but a candidate
       run with ``--capture-blend-metrics``).
   * - ``engine_version`` (arm, summary)
     - ``null``: the engine did not report a version.
   * - ``scores`` values (rows)
     - ``null``: the judge gave no rating for that arm's answer.
   * - ``loss_tag`` (rows)
     - ``null``: the candidate or ``reference_1`` score is ``null``.
   * - ``reference_flipped`` (rows)
     - ``null``: only one reference pass, or one of its scores is ``null``.
   * - ``reference_flip_rate`` (summary)
     - ``-1.0``: only one reference pass.
   * - ``standard_error`` (summary)
     - ``null``: fewer than two questions were scored by every arm.
   * - ``engagement`` (summary)
     - Not measured: ``measured`` is ``false``, every number is ``0.0`` and
       ``engaged`` is ``true`` (an unmeasured run is reported, not
       invalidated).

**Conditional keys.** ``rows.jsonl``'s ``match`` and ``summary.json``'s
``settings.judge_model`` appear only with the judge metric, and
``summary.json``'s ``judge`` is ``null`` without it. ``loss_tags`` holds only
the tags that occur; ``evidence_sources`` always has all three keys.

**Counters** are floats, because they are read from Prometheus, even when
they count tokens or chunks.


plan.json
---------

The plan every arm receives, saved so that ``analyze`` needs no tokenizer
or dataset and so that later runs can check they build the same prompts.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``schema_version``
     - int
     - ``1``.
   * - ``plan_fingerprint``
     - string
     - 16 hex. Every ``arm_<label>.json`` must carry the same value.
   * - ``prompts_digest``
     - string
     - 16 hex, over every store and measured message.
   * - ``lmcache_version``
     - string
     - The lmcache that wrote the plan (the first ``run``).
   * - ``created_at``
     - string
     - When the first ``run`` wrote the plan.
   * - ``settings``
     - object
     - The run settings that shape the prompts (below). Every ``run`` into
       the directory must use the same values.
   * - ``plan``
     - object
     - The plan itself (below).

``settings``:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``model``
     - string
     - ``--model``.
   * - ``dataset``
     - string
     - ``--dataset`` as given: a hub name (``hotpotqa``) or a file path.
   * - ``task``
     - string
     - ``"qa"``.
   * - ``scenario``
     - string
     - ``"independent"``, ``"reordered"`` or ``"concatenated"``.
   * - ``num_samples``
     - int
     - ``--limit``; ``0`` means every question. The actual count is
       ``summary.json``'s ``num_questions``.
   * - ``chunk_size``
     - int
     - ``--chunk-size``.
   * - ``max_output_tokens``
     - int
     - ``--max-output-tokens``.
   * - ``template_kwargs``
     - object
     - ``--template-kwargs`` as ``{key: bool | int | string}``; ``{}`` when
       none.
   * - ``seed``
     - int
     - ``--seed``.
   * - ``filler``
     - string
     - ``--filler``: ``"dots"``, ``"dashes"`` or ``"words"``; ``"none"``
       for ``concatenated``.
   * - ``min_unit_tokens``
     - int
     - ``--min-unit-tokens``; ``concatenated`` only (absent otherwise).

The arm URLs, ``--reference-repeats`` and ``--capture-blend-metrics`` are not
settings: runs of different arms into one directory may differ in them.

``plan``:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``scenario``, ``chunk_size``
     - string, int
     - Same as in ``settings``.
   * - ``store_requests``
     - array
     - In send order: ``{request_id, messages}``, where ``messages`` is
       ``[{"role": "user", "content": <prompt>}]``. Sent with
       ``max_tokens=1`` to caching arms before measuring.
   * - ``samples``
     - array
     - One per question, in dataset order: the question's
       `manifest.jsonl`_ record plus ``question`` (string), ``answers``
       (array of gold answers; more than one means alternate phrasings),
       ``documents`` (array of raw passages, dataset order),
       ``evidence_indices`` (array of ints into ``documents`` that the
       dataset marks as evidence; ``[]`` when it marks none) and
       ``messages`` (the measured prompt, as in ``store_requests``).
   * - ``padding``
     - object
     - ``{filler, pad_tokens, prompt_tokens}``: padding tokens and all
       tokens over every measured prompt, chat template included. Absent in
       plans made before it was recorded.

.. code-block:: json

   {
     "schema_version": 1,
     "plan_fingerprint": "d7428c9bc57348c7",
     "prompts_digest": "ca2bc4ffc3dc39b2",
     "lmcache_version": "0.4.0-1a2b3c4",
     "created_at": "2026-10-19T10:30:05-05:00",
     "settings": {"model": "Qwen/Qwen3-8B", "dataset": "hotpotqa", "task": "qa",
                  "scenario": "independent", "num_samples": 200, "chunk_size": 256,
                  "max_output_tokens": 2048, "template_kwargs": {}, "seed": 42},
     "plan": {"scenario": "independent", "chunk_size": 256,
              "store_requests": [{"request_id": "store0", "messages": ["..."]}],
              "samples": ["..."]}
   }


manifest.jsonl
--------------

The reuse manifest: for every question, how each of its passages got into
the cache. One record per question, in plan order. ``analyze`` does not read
it (``plan.json`` holds the same records); it is there to read and grep.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``qid``
     - string
     - The dataset's question id.
   * - ``scenario``, ``chunk_size``
     - string, int
     - The plan's, repeated on every line.
   * - ``request_id``
     - string
     - ``q<N>``, the measured request.
   * - ``prompt_hash``
     - string
     - 16 hex. Equal on every arm: the prompts are byte-identical.
   * - ``evidence_source``
     - string
     - How the answer-bearing passages were found: ``"dataset"`` (the
       dataset marks them), ``"answer_string"`` (passages containing a gold
       answer) or ``"none"``.
   * - ``evidence_passage_ids``
     - array
     - ``passage_id`` of each evidence passage, in measured order.
   * - ``expected_reused_tokens``
     - int or null
     - Tokens of the rendered request covered by a whole chunk of a
       rendered store request, at any offset: what the blend server's
       matcher can reuse. ``null`` in older plans.
   * - ``unit_position``
     - int or null
     - ``concatenated``: this question's context's index in its unit
       (``0`` = first); ``null`` otherwise.
   * - ``duplicate_of``
     - string or null
     - ``concatenated``: the ``qid`` whose identical context was stored in
       this question's place; ``null`` when its own was stored.
   * - ``passages``
     - array
     - One per passage, in **measured** order (below).

``passages[]``:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``passage_id``
     - string
     - 12 hex.
   * - ``origin``
     - string
     - ``"alone"`` (``independent``: stored by a request holding only this
       passage), ``"with_sample_passages"`` (``reordered``: stored with
       the question's other passages, in dataset order) or ``"in_unit"``
       (``concatenated``: stored in its question's context inside a unit;
       ``stored_position`` counts passages from the unit's start).
   * - ``store_request_id``
     - string
     - ``store<N>``, the store request that cached it.
   * - ``stored_position``
     - int
     - Its index among that store request's passages; always ``0`` for
       ``independent``.
   * - ``measured_position``
     - int
     - Its index in the measured prompt.
   * - ``is_evidence``
     - bool
     - Whether it holds the answer (see ``evidence_source``).
   * - ``measured_earlier_in``
     - array
     - ``q<N>`` ids of earlier questions whose prompts also hold this
       passage. A caching candidate caches measured prompts too, so the passage
       may be reused from the last of these rather than from
       ``store_request_id``. ``[]`` when none.
   * - ``pad_tokens``
     - int
     - Padding tokens added to align it to whole chunks (the filler only,
       not the ``Passage <id>:`` label).

.. code-block:: json

   {"qid": "s1", "scenario": "independent", "chunk_size": 256, "request_id": "q1",
    "prompt_hash": "3ef3226f2db0ad8c", "evidence_source": "answer_string",
    "evidence_passage_ids": ["193996aa3260"],
    "passages": [
      {"passage_id": "2408066aee02", "origin": "alone", "store_request_id": "store1",
       "stored_position": 0, "measured_position": 0, "is_evidence": false,
       "measured_earlier_in": ["q0"], "pad_tokens": 118},
      {"passage_id": "193996aa3260", "origin": "alone", "store_request_id": "store2",
       "stored_position": 0, "measured_position": 1, "is_evidence": true,
       "measured_earlier_in": [], "pad_tokens": 97}]}


arm_<label>.json
----------------

One per arm pass: where and how the pass ran. Written after the pass's
answers, so its presence marks the pass complete.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``schema_version``
     - int
     - ``1``.
   * - ``plan_fingerprint``
     - string
     - Must equal ``plan.json``'s, or ``analyze`` refuses the arm (exit 2).
   * - ``role``
     - string
     - ``"reference"``, ``"control"`` or ``"candidate"``.
   * - ``run_index``
     - int
     - 0-based repeat; above 0 only for reference passes.
   * - ``label``
     - string
     - ``reference_<run_index + 1>``, ``control`` or ``candidate``.
   * - ``url``
     - string
     - The engine URL, as given to ``run``.
   * - ``blend_metrics_url``
     - string
     - ``--capture-blend-metrics`` for a candidate run with it; ``""``
       otherwise.
   * - ``engine_version``
     - string or null
     - The engine's reported version.
   * - ``lmcache_version``
     - string
     - The lmcache that ran the pass.
   * - ``started_at``, ``finished_at``
     - string
     - When the pass started and finished.
   * - ``engagement_measured``
     - bool
     - Whether blend-server counters were read.
   * - ``counters``
     - object or null
     - The pass's blend-server counter deltas, read with a short settle
       delay at each end (below); ``null`` when not measured.
   * - ``store_failures``
     - int
     - Store requests that failed; always ``0`` for reference passes, which
       send none. Any failure makes the verdict INVALID.
   * - ``num_questions``
     - int
     - Questions answered in the pass.

``counters`` (also per question in ``answers_<label>.jsonl``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Description
   * - ``requested_tokens``
     - Chunk-aligned prompt tokens the blend server was asked to look up.
   * - ``prefix_hit_tokens``
     - Of those, served as an exact prefix.
   * - ``non_prefix_hit_tokens``
     - Of those, served by blending reused chunks at new positions. Blend
       share = ``non_prefix_hit_tokens / requested_tokens``.
   * - ``segmented_prefix_hit_tokens``
     - Prefix chunks loaded past an L2 load failure, at their original
       positions (``--enable-segmented-prefix`` only); not blending.
   * - ``stale_chunks``
     - Chunks whose fingerprint matched but whose KV was gone.
   * - ``fingerprints_registered``
     - Chunks added to the blend server's fingerprint index during the pass.

.. code-block:: json

   {"schema_version": 1, "plan_fingerprint": "d7428c9bc57348c7", "role": "candidate",
    "run_index": 0, "label": "candidate", "url": "http://gpu1:8000",
    "blend_metrics_url": "http://gpu1:9090/metrics", "engine_version": "0.11.0",
    "lmcache_version": "0.4.0-1a2b3c4",
    "started_at": "2026-10-19T10:52:40-05:00", "finished_at": "2026-10-19T11:03:17-05:00",
    "engagement_measured": true,
    "counters": {"requested_tokens": 1024.0, "prefix_hit_tokens": 256.0,
                 "non_prefix_hit_tokens": 768.0, "segmented_prefix_hit_tokens": 0.0,
                 "stale_chunks": 0.0, "fingerprints_registered": 8.0},
    "store_failures": 0, "num_questions": 200}


answers_<label>.jsonl
---------------------

The raw outcome of every measured question in one pass, in plan order. This
is where to look when an arm's answer is surprising: ``response`` holds the
full text, reasoning included.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``qid``
     - string
     - The dataset's question id.
   * - ``request_id``
     - string
     - ``q<N>``.
   * - ``successful``
     - bool
     - Whether the request produced output.
   * - ``answer``
     - string
     - The text inside the last ``<answer>`` tags (``<final_answer>`` and
       gpt-oss's bare ``final_answer>…</final_answer>`` are also accepted);
       ``""`` when the request failed or the tags were missing or cut off.
   * - ``response``
     - string
     - The full response, reasoning included.
   * - ``ttft``
     - number
     - Time to first token in seconds, rounded to 4 places; ``-1.0`` on
       failure.
   * - ``num_output_tokens``
     - int
     - Tokens generated.
   * - ``counters``
     - object or null
     - This question's counter deltas (fields as in
       `arm_<label>.json`_). Approximate: a lookup the server records after
       the response lands in the next question's deltas.

.. code-block:: json

   {"qid": "s2", "request_id": "q2", "successful": true, "answer": "Geneva",
    "response": "Bern is the federal city, but ... <answer>Geneva</answer>",
    "ttft": 0.0841, "num_output_tokens": 212,
    "counters": {"requested_tokens": 256.0, "prefix_hit_tokens": 64.0,
                 "non_prefix_hit_tokens": 192.0, "segmented_prefix_hit_tokens": 0.0,
                 "stale_chunks": 0.0, "fingerprints_registered": 2.0}}


rows.jsonl
----------

The per-question comparison, one row per question in plan order -- including
questions some arm could not be scored on. Start here to see which questions
the candidate lost and why.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``qid``
     - string
     - The dataset's question id.
   * - ``gold``
     - array
     - The gold answers (``answers`` in ``plan.json``).
   * - ``scores``
     - object
     - ``{label: number or null}`` for every pass, with the verdict metric:
       ``match`` is ``0.0`` or ``1.0``; ``f1`` is 0 to 1; ``judge`` is 0 to 1
       in steps of 0.25, or ``null`` without a rating (a judge that answers
       off its 0/2/4 scale can produce an in-between value such as
       ``0.375``). A missing answer scores ``0.0`` under every metric.
   * - ``match``
     - object
     - Judge metric only: ``{label: 0.0 or 1.0}``, the judge-free match
       score, as a cross-check of the judge.
   * - ``answers``
     - object
     - ``{label: string}``, each pass's extracted answer.
   * - ``loss_tag``
     - string or null
     - How the candidate's answer compares with ``reference_1``'s: ``"none"``
       (same score), ``"candidate_higher"``, or, for a loss, ``"empty"`` (no
       complete answer), ``"abstention"`` (said the passages lack the
       answer), ``"wrong_value"`` (answered with another string from the
       prompt) or ``"wording"`` (anything else).
   * - ``reference_flipped``
     - bool or null
     - Whether ``reference_2`` scored differently from ``reference_1``.
   * - ``blend_share``
     - number or null
     - This question's blend share (0 to 1).
   * - ``reuse``
     - object
     - ``{expected_tokens, kind, prefix_tokens, blended_tokens}``:
       ``expected_tokens`` from the manifest; ``kind`` (``"prefix"``,
       ``"blended"`` or ``"none"``, whichever served more of the question)
       and the candidate's token counts, ``null`` without captured counters.
   * - ``evidence``
     - array
     - The evidence passages, in measured order:
       ``{passage_id, origin, stored_position, measured_position}``. Join to
       `manifest.jsonl`_ by ``passage_id`` for the rest.

.. code-block:: json

   {"qid": "s2", "gold": ["Bern"],
    "scores": {"reference_1": 1.0, "reference_2": 1.0, "candidate": 0.0},
    "match": {"reference_1": 1.0, "reference_2": 1.0, "candidate": 0.0},
    "answers": {"reference_1": "Bern", "reference_2": "Bern", "candidate": "Geneva"},
    "loss_tag": "wording", "reference_flipped": false, "blend_share": 0.75,
    "evidence": [{"passage_id": "b887df1256ff", "origin": "alone",
                  "stored_position": 0, "measured_position": 0}]}


summary.json
------------

The verdict and everything needed to interpret it.

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Field
     - Type
     - Description
   * - ``run_fingerprint``
     - string
     - 16 hex. Two summaries are comparable only when it matches.
   * - ``plan_fingerprint``
     - string
     - ``plan.json``'s.
   * - ``settings``
     - object
     - ``plan.json``'s settings plus ``verdict_metric`` (``"match"``,
       ``"f1"`` or ``"judge"``), ``judge_model`` (judge metric only),
       ``reference_repeats`` (reference passes analyzed), ``tolerance``,
       ``min_blend_share``, ``min_reuse_ratio`` and ``blend_metrics_url``
       (from the candidate arm).
   * - ``arms``
     - object
     - ``{label: {url, engine_version, lmcache_version, started_at,
       finished_at}}``, copied from each arm file.
   * - ``num_questions``
     - int
     - Questions in the plan.
   * - ``unscored_questions``
     - int
     - Questions left out of every comparison because some pass has no
       score for them (judge only). More than 5% makes the verdict INVALID.
   * - ``judge``
     - object or null
     - ``{model, metric}`` with the judge metric (``metric`` is
       ``"ragas-0.4.3/nv_accuracy"``).
   * - ``evidence_sources``
     - object
     - Question counts by ``evidence_source``: ``{dataset, answer_string,
       none}``.
   * - ``measured_earlier_share``
     - number
     - Share of measured passages that an earlier question also held
       (see ``measured_earlier_in``).
   * - ``padding``
     - object or null
     - ``plan.json``'s ``padding`` plus ``share`` (padding tokens over
       measured prompt tokens); ``null`` for older plans.
   * - ``reuse``
     - object or null
     - ``{expected_tokens, observed_tokens, observed_prefix_tokens,
       observed_blended_tokens, observed_over_expected, groups}``: the
       plan's expected reuse against the candidate pass's counters, and
       ``groups``, one ``{kind, questions, delta, standard_error}`` per reuse
       kind present. ``null`` without captured counters.
   * - ``num_store_requests``
     - int
     - Store requests in the plan.
   * - ``verdict``
     - string
     - ``"PASS"``, ``"FAIL"``, ``"UNDERPOWERED"`` or ``"INVALID"``.
   * - ``invalid_reasons``
     - array
     - Why the verdict is INVALID; ``[]`` otherwise.
   * - ``samples_needed``
     - int
     - For UNDERPOWERED, about how many questions would settle it; ``0``
       otherwise.
   * - ``reference_flip_rate``
     - number
     - Share of questions whose score differs between ``reference_1`` and
       ``reference_2``: the noise floor.
   * - ``candidate``
     - object
     - The candidate pass against ``reference_1`` (below).
   * - ``others``
     - array
     - Every other pass (``reference_2``, …, ``control``) against
       ``reference_1``, same fields. ``reference_1`` itself never appears:
       it is the baseline; its mean is ``candidate.mean_score - candidate.delta``.
   * - ``engagement``
     - object
     - Whether a blend-server candidate actually blended (below).
   * - ``arm_counts``
     - object
     - ``{label: {answered, empty, abstained}}``: complete answers, missing
       or cut-off answers, and complete answers that decline to answer.
   * - ``store_failures``
     - object
     - ``{label: int}``.
   * - ``loss_tags``
     - object
     - Counts of ``rows.jsonl``'s ``loss_tag`` values, ``none`` and
       ``candidate_higher`` included; only tags that occur.

``candidate`` and ``others[]``:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``label``
     - string
     - The pass.
   * - ``mean_score``
     - number
     - Mean score over the questions every pass was scored on.
   * - ``delta``
     - number
     - ``mean_score`` minus ``reference_1``'s, paired by question. Negative
       is a loss.
   * - ``standard_error``
     - number or null
     - Standard error of the paired per-question differences. The verdict
       is FAIL only when ``|delta|`` is beyond both ``tolerance`` and twice
       this.

``engagement``:

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Field
     - Type
     - Description
   * - ``measured``
     - bool
     - Whether the candidate was run with ``--capture-blend-metrics``.
   * - ``blend_share``
     - number
     - Share of looked-up tokens served by blending, over the pass.
   * - ``min_window_share``
     - number
     - Lowest share over consecutive windows of 10 questions (equal to
       ``blend_share`` with fewer than 10).
   * - ``stale_chunks``, ``segmented_prefix_hit_tokens``, ``fingerprints_registered``
     - number
     - The pass's counters (see `arm_<label>.json`_).
   * - ``engaged``
     - bool
     - Whether the gate's run-wide and minimum-window values reach its
       minimum (``min_blend_share``, or ``min_reuse_ratio`` for the reuse
       gate); ``false`` makes the verdict INVALID.
   * - ``gate``
     - string
     - What ``engaged`` was decided on: ``"blend_share"``, or ``"reuse"``
       for the ``concatenated`` scenario (where the blend share is reported
       only).
   * - ``reuse_ratio``
     - number or null
     - Reuse gate only: observed reused tokens (prefix, segmented prefix and
       blended) over the plan's expected reuse, over the pass. ``null``
       otherwise, or when the plan expects no reuse.
   * - ``min_window_reuse_ratio``
     - number or null
     - Reuse gate only: lowest ``reuse_ratio`` over consecutive windows of 10
       questions, skipping windows that expect no reuse.

.. code-block:: json

   {
     "run_fingerprint": "ec1601a6e6537308",
     "plan_fingerprint": "d7428c9bc57348c7",
     "settings": {"model": "Qwen/Qwen3-8B", "dataset": "hotpotqa", "...": "...",
                  "verdict_metric": "judge", "judge_model": "openai/gpt-oss-120b",
                  "reference_repeats": 2, "tolerance": 0.05, "min_blend_share": 0.3,
                  "min_reuse_ratio": 0.8, "blend_metrics_url": "http://gpu1:9090/metrics"},
     "arms": {"candidate": {"url": "http://gpu1:8000", "engine_version": "0.11.0",
                        "lmcache_version": "0.4.0-1a2b3c4",
                        "started_at": "2026-10-19T10:52:40-05:00",
                        "finished_at": "2026-10-19T11:03:17-05:00"}, "...": "..."},
     "num_questions": 200, "unscored_questions": 0,
     "judge": {"model": "openai/gpt-oss-120b", "metric": "ragas-0.4.3/nv_accuracy"},
     "evidence_sources": {"dataset": 200, "answer_string": 0, "none": 0},
     "measured_earlier_share": 0.02, "num_store_requests": 1964,
     "verdict": "FAIL", "invalid_reasons": [], "samples_needed": 0,
     "reference_flip_rate": 0.03,
     "candidate": {"label": "candidate", "mean_score": 0.768, "delta": -0.137,
               "standard_error": 0.032},
     "others": [{"label": "reference_2", "mean_score": 0.9, "delta": -0.005,
                 "standard_error": 0.0095}],
     "engagement": {"measured": true, "blend_share": 0.912, "min_window_share": 0.874,
                    "stale_chunks": 0.0, "segmented_prefix_hit_tokens": 0.0,
                    "fingerprints_registered": 4180.0, "engaged": true,
                    "gate": "blend_share", "reuse_ratio": null,
                    "min_window_reuse_ratio": null},
     "arm_counts": {"candidate": {"answered": 200, "empty": 0, "abstained": 9}, "...": "..."},
     "store_failures": {"reference_1": 0, "reference_2": 0, "candidate": 0},
     "loss_tags": {"none": 168, "candidate_higher": 4, "abstention": 9, "wording": 19}
   }


judge_cache.jsonl
-----------------

Judge ratings, one line per judged (question, answer, gold answer) triple,
so rerunning ``analyze`` judges only what is new.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``key``
     - string
     - 64 hex, over the judge metric id, judge model, question, answer and
       the question's first gold answer. The judge URL is not part of it.
   * - ``score``
     - number
     - The mean of the available ratings, 0 to 1 in steps of 0.25.
   * - ``ratings``
     - array
     - ``[answer_first, reference_first]``: the two prompts' ratings, each
       ``0.0``, ``0.5`` or ``1.0`` (a judge reply of 0, 2 or 4, divided by
       4), or ``null`` if that prompt gave none. A reply off that scale is
       still read, as ragas reads it (1 or 3 gives ``0.25`` or ``0.75``).

Answers the judge could not rate at all are not cached, so a rerun tries
them again; neither are the sanity check and missing answers (those score
``0.0`` without a judge call). An unreadable line, such as a partial last
line from a killed run, is skipped and re-judged.

.. code-block:: json

   {"key": "e1b355a1cfb8035bf28b99ddae617b19739578a4302f584594ac82953c8d6a7f",
    "score": 0.75, "ratings": [1.0, 0.5]}


Working with the results
------------------------

The questions the candidate lost, with both answers:

.. code-block:: bash

   jq -c 'select(.loss_tag != null and .loss_tag != "none" and .loss_tag != "candidate_higher")
          | {qid, gold, loss_tag, reference: .answers.reference_1, candidate: .answers.candidate}' \
      out/rows.jsonl

``reference_1``'s mean score, which ``summary.json`` leaves implicit:

.. code-block:: bash

   jq '.candidate.mean_score - .candidate.delta' out/summary.json

Whether a verdict can be trusted before reading its numbers:

.. code-block:: bash

   jq '{verdict, invalid_reasons, engaged: .engagement.engaged,
        flip_rate: .reference_flip_rate, unscored: .unscored_questions}' out/summary.json

Whether two summaries can be compared (same prompts, metric and judge):

.. code-block:: bash

   [ "$(jq -r .run_fingerprint a/summary.json)" = "$(jq -r .run_fingerprint b/summary.json)" ]

A lost question's evidence passages, and how each was cached:

.. code-block:: bash

   jq -c 'select(.qid == "s2") | .passages[] | select(.is_evidence)' out/manifest.jsonl
