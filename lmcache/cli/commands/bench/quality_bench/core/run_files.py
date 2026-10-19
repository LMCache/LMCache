# SPDX-License-Identifier: Apache-2.0
"""The results directory that ``bench quality run`` and ``analyze`` share.

``run`` and ``analyze`` are separate commands so that each arm, and the
judge, can be served on its own (a GPU server holds one engine at a time)
and so that saved answers can be re-analyzed. They communicate only through
these files::

    <results dir>/
        plan.json               the plan: settings, prompts, questions, golds
        manifest.jsonl          the reuse manifest, one record per question
        arm_<label>.json        one arm pass: metadata, written last
        answers_<label>.jsonl   that pass's answers, one record per question
        judge_cache.jsonl       judge ratings, reused by later analyzes
        rows.jsonl, summary.json    written by analyze

Every ``run`` into a directory must build the same plan as ``plan.json``
records; every arm records the plan's fingerprint, and ``analyze`` refuses
arms from another plan.
"""

# Standard
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, NoReturn
import glob
import hashlib
import json
import os
import sys

# First Party
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import (
    BlendCounters,
)
from lmcache.cli.commands.bench.quality_bench.core.config import ArmRole
from lmcache.cli.commands.bench.quality_bench.core.plan import QualityPlan
from lmcache.cli.commands.bench.quality_bench.core.runner import ArmAnswer, ArmRun
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Bumped when a file's layout changes, so an older or newer lmcache refuses a
# directory it would misread.
SCHEMA_VERSION = 1

PLAN_FILE = "plan.json"
MANIFEST_FILE = "manifest.jsonl"
ROWS_FILE = "rows.jsonl"
SUMMARY_FILE = "summary.json"
JUDGE_CACHE_FILE = "judge_cache.jsonl"

# Name of a results directory created when --output-dir is omitted.
RUN_DIR_FORMAT = "%Y%m%d-%H%M%S"

# Analyze's input order: what decide() and format_summary() assume.
_ROLE_ORDER = (ArmRole.REFERENCE, ArmRole.PREFIX_CACHE, ArmRole.BLEND)

# Roles a verdict needs, with the run flag that produces each.
_REQUIRED_ROLES = (
    (ArmRole.REFERENCE, "--reference-url"),
    (ArmRole.BLEND, "--blend-url"),
)


class ResultsDirError(ValueError):
    """The results directory is missing something, or does not fit.

    A setup problem rather than a quality result: ``run`` and ``analyze``
    exit 2 on it, as for an invalid invocation, so CI can tell it apart from
    a FAIL or INVALID verdict (exit 1).
    """


@dataclass(frozen=True)
class PlanRecord:
    """A results directory's ``plan.json``, loaded.

    Attributes:
        plan: The plan every arm received.
        settings: The run settings that built it (``QualityRunConfig``'s
            ``plan_settings``).
        fingerprint: Digest of ``settings`` and the prompts; every arm
            records it.
        lmcache_version: The lmcache that wrote the plan.
    """

    plan: QualityPlan
    settings: Mapping[str, object]
    fingerprint: str
    lmcache_version: str


@dataclass(frozen=True)
class ArmRecord:
    """One saved arm pass, loaded.

    Attributes:
        run: The pass, as the runner produced it.
        url: The engine it ran against.
        blend_metrics_url: The metrics endpoint read during the pass; empty
            when engagement was not measured.
        engine_version: The engine's reported version, when it reported one.
        lmcache_version: The lmcache that ran the pass.
        started_at: When the pass started (ISO 8601).
        finished_at: When the pass finished (ISO 8601).
    """

    run: ArmRun
    url: str
    blend_metrics_url: str
    engine_version: str | None
    lmcache_version: str
    started_at: str
    finished_at: str

    def metadata(self) -> dict[str, object]:
        """Return the provenance fields, for the summary and the terminal."""
        return {
            "url": self.url,
            "engine_version": self.engine_version,
            "lmcache_version": self.lmcache_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def lmcache_version() -> str:
    """Return the installed lmcache version, with its commit when known.

    Read from the build-generated ``lmcache._version`` rather than
    ``lmcache.utils``, which imports torch (absent from ``lmcache-cli``).
    """
    try:
        # First Party
        from lmcache import _version
    except ImportError:
        return "unknown"
    version = getattr(_version, "__version__", "unknown")
    commit = getattr(_version, "__commit_id__", "")
    return f"{version}-{commit}" if commit else version


def exit_on_results_dir_error(command: str, error: ResultsDirError) -> NoReturn:
    """Report a results-directory problem and exit 2, without a traceback.

    Args:
        command: The subcommand that hit it (``run`` or ``analyze``).
        error: The problem; its message says what to run or fix.

    Raises:
        SystemExit: Always, with status 2.
    """
    print(f"lmcache bench quality {command}: error: {error}", file=sys.stderr)
    raise SystemExit(2) from error


def plan_fingerprint(settings: Mapping[str, object], prompts_digest: str) -> str:
    """Return the digest that identifies one plan.

    Args:
        settings: The run settings that built the plan.
        prompts_digest: ``QualityPlan.prompts_digest``.

    Returns:
        A 16-hex-digit digest.
    """
    material = json.dumps(
        {"settings": dict(settings), "prompts": prompts_digest}, sort_keys=True
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def create_run_dir(output_dir: str, started: datetime) -> str:
    """Create a new, uniquely named results directory under *output_dir*.

    Used when ``--output-dir`` is omitted, so that local runs never land in
    an earlier run's directory.

    Args:
        output_dir: Parent directory -- created if missing.
        started: When the run started. Names the directory
            (``YYYYmmdd-HHMMSS``).

    Returns:
        Path of the new directory. A run that starts in the same second as
        an existing directory gets a ``-2``, ``-3``, … suffix.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, started.strftime(RUN_DIR_FORMAT))
    path, suffix = base, 1
    while True:
        try:
            # mkdir claims the name atomically, even against a concurrent run.
            os.mkdir(path)
            return path
        except FileExistsError:
            suffix += 1
            path = f"{base}-{suffix}"


def ensure_plan(
    results_dir: str, plan: QualityPlan, settings: Mapping[str, object]
) -> str:
    """Record *plan* in *results_dir*, or check it matches the recorded one.

    The first ``run`` into a directory writes ``plan.json`` and
    ``manifest.jsonl``. Later runs (other arms, possibly on other machines
    or days) must have built the same plan, or their answers would not be
    comparable; this is checked before any request is sent.

    Args:
        results_dir: The results directory; created if missing.
        plan: The plan this run built.
        settings: The run settings that built it.

    Returns:
        The plan's fingerprint.

    Raises:
        ResultsDirError: If ``plan.json`` was written by another schema version,
            with other settings, or with other prompts.
    """
    fingerprint = plan_fingerprint(settings, plan.prompts_digest)
    path = os.path.join(results_dir, PLAN_FILE)
    if not os.path.exists(path):
        os.makedirs(results_dir, exist_ok=True)
        _write_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "plan_fingerprint": fingerprint,
                "prompts_digest": plan.prompts_digest,
                "lmcache_version": lmcache_version(),
                "created_at": _now(),
                "settings": dict(settings),
                "plan": plan.to_record(),
            },
        )
        _write_jsonl(
            os.path.join(results_dir, MANIFEST_FILE),
            [s.manifest_entry(plan.scenario, plan.chunk_size) for s in plan.samples],
        )
        return fingerprint

    recorded = _read_json(path)
    _check_schema(path, recorded)
    if recorded.get("plan_fingerprint") == fingerprint:
        return fingerprint
    recorded_settings = recorded.get("settings", {})
    changed = [
        f"{key}: {recorded_settings.get(key)!r} -> {value!r}"
        for key, value in settings.items()
        if recorded_settings.get(key) != value
    ]
    if changed:
        raise ResultsDirError(
            f"{path} was written with other settings ({'; '.join(changed)}). "
            f"Runs into one results directory must share them; use the same "
            f"flags or a new --output-dir."
        )
    raise ResultsDirError(
        f"The prompts differ from {path} although every setting matches: the "
        f"tokenizer, the dataset revision or lmcache's prompt code changed "
        f"since it was written (by lmcache "
        f"{recorded.get('lmcache_version', 'unknown')}). Rerun every arm into "
        f"a new --output-dir."
    )


def remove_role(results_dir: str, role: ArmRole) -> list[str]:
    """Delete a role's saved passes, before a new run of that role.

    A role's passes always come from one ``run`` (one engine lifetime), so
    the reference flip rate never mixes engines and fewer repeats never
    leave a stale pass behind. ``rows.jsonl`` and ``summary.json`` go too:
    they described the old passes.

    Args:
        results_dir: The results directory.
        role: The role about to be rerun.

    Returns:
        The deleted paths.
    """
    patterns = [f"arm_{role.value}*.json", f"answers_{role.value}*.jsonl"]
    paths = [p for pattern in patterns for p in glob.glob(_in(results_dir, pattern))]
    paths += [
        p
        for p in (_in(results_dir, ROWS_FILE), _in(results_dir, SUMMARY_FILE))
        if os.path.exists(p)
    ]
    for path in paths:
        os.remove(path)
    return paths


def write_arm(
    results_dir: str,
    run: ArmRun,
    fingerprint: str,
    url: str,
    blend_metrics_url: str,
    engine_version: str | None,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    """Save one arm pass: its answers, then its metadata.

    The metadata file is written last and marks the pass complete; answers
    without it are ignored by :func:`load_arms`.

    Args:
        results_dir: The results directory.
        run: The finished pass.
        fingerprint: The plan's fingerprint, from :func:`ensure_plan`.
        url: The engine the pass ran against.
        blend_metrics_url: The metrics endpoint read during the pass; empty
            when engagement was not measured.
        engine_version: The engine's reported version, if any.
        started_at: When the pass started.
        finished_at: When the pass finished.
    """
    _write_jsonl(
        _in(results_dir, f"answers_{run.label}.jsonl"),
        [_answer_record(a, run.engagement_measured) for a in run.answers],
    )
    _write_json(
        _in(results_dir, f"arm_{run.label}.json"),
        {
            "schema_version": SCHEMA_VERSION,
            "plan_fingerprint": fingerprint,
            "role": run.role.value,
            "run_index": run.run_index,
            "label": run.label,
            "url": url,
            "blend_metrics_url": blend_metrics_url,
            "engine_version": engine_version,
            "lmcache_version": lmcache_version(),
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "engagement_measured": run.engagement_measured,
            "counters": asdict(run.counters) if run.engagement_measured else None,
            "store_failures": run.store_failures,
            "num_questions": len(run.answers),
        },
    )


def load_plan(results_dir: str) -> PlanRecord:
    """Load and verify a results directory's ``plan.json``.

    Args:
        results_dir: The results directory.

    Returns:
        The plan and its provenance.

    Raises:
        ResultsDirError: If ``plan.json`` is missing, from another schema
            version, malformed, or no longer matches its own digests.
    """
    path = _in(results_dir, PLAN_FILE)
    if not os.path.exists(path):
        raise ResultsDirError(
            f"no {PLAN_FILE} in {results_dir}: run `lmcache bench quality run "
            f"--output-dir {results_dir}` first"
        )
    recorded = _read_json(path)
    _check_schema(path, recorded)
    try:
        plan = QualityPlan.from_record(recorded["plan"])
        settings = dict(recorded["settings"])
        fingerprint = str(recorded["plan_fingerprint"])
        prompts_digest = str(recorded["prompts_digest"])
    except (KeyError, TypeError, ValueError) as e:
        raise ResultsDirError(f"{path}: {e}") from e
    if plan.prompts_digest != prompts_digest or fingerprint != plan_fingerprint(
        settings, prompts_digest
    ):
        raise ResultsDirError(f"{path}: contents do not match its digests (edited?)")
    return PlanRecord(
        plan=plan,
        settings=settings,
        fingerprint=fingerprint,
        lmcache_version=str(recorded.get("lmcache_version", "unknown")),
    )


def load_arms(results_dir: str, plan_record: PlanRecord) -> list[ArmRecord]:
    """Load every completed arm pass, in the order the verdict expects.

    Args:
        results_dir: The results directory.
        plan_record: The directory's plan, from :func:`load_plan`.

    Returns:
        Reference passes by repeat, then the apc pass, then the blend pass.

    Raises:
        ResultsDirError: If a pass is from another plan or schema version, is
            malformed, does not answer the plan's questions in order, or the
            passes lack a blend arm, a reference arm or a reference repeat.
    """
    records = [
        _load_arm(path, plan_record)
        for path in sorted(glob.glob(_in(results_dir, "arm_*.json")))
    ]
    labels = {r.run.label for r in records}
    for path in sorted(glob.glob(_in(results_dir, "answers_*.jsonl"))):
        label = os.path.basename(path)[len("answers_") : -len(".jsonl")]
        if label not in labels:
            logger.warning(
                "Ignoring %s: no arm_%s.json, so its pass did not finish",
                path,
                label,
            )

    roles = {r.run.role for r in records}
    missing = [(role, flag) for role, flag in _REQUIRED_ROLES if role not in roles]
    if missing:
        # Name every missing arm at once, so one fix is enough.
        names = " and ".join(role.value for role, _ in missing)
        flags = " ".join(f"{flag} …" for _, flag in missing)
        raise ResultsDirError(
            f"{results_dir} has no {names} arm{'s' if len(missing) > 1 else ''}: "
            f"run `lmcache bench quality run {flags} --output-dir {results_dir}` "
            f"first (arms can also be run one at a time)"
        )
    records.sort(key=lambda r: (_ROLE_ORDER.index(r.run.role), r.run.run_index))
    indices = [r.run.run_index for r in records if r.run.role is ArmRole.REFERENCE]
    if indices != list(range(len(indices))):
        raise ResultsDirError(
            f"{results_dir}: reference passes {indices} are not consecutive "
            f"from 0; rerun the reference arm"
        )
    return records


def _now() -> str:
    """Return the current local time, ISO 8601, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _in(results_dir: str, name: str) -> str:
    """Return the path of *name* inside *results_dir*."""
    return os.path.join(results_dir, name)


def _check_schema(path: str, record: Mapping[str, Any]) -> None:
    """Refuse a file written with another layout."""
    version = record.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ResultsDirError(
            f"{path} has schema version {version}; this lmcache reads "
            f"{SCHEMA_VERSION}. Use a new --output-dir."
        )


def _load_arm(path: str, plan_record: PlanRecord) -> ArmRecord:
    """Load one ``arm_<label>.json`` and its answers."""
    meta = _read_json(path)
    _check_schema(path, meta)
    if meta.get("plan_fingerprint") != plan_record.fingerprint:
        raise ResultsDirError(
            f"{path} was run with another plan (fingerprint "
            f"{meta.get('plan_fingerprint')}, plan.json has "
            f"{plan_record.fingerprint}); rerun that arm into this directory"
        )
    try:
        role = ArmRole(meta["role"])
        measured = bool(meta["engagement_measured"])
        answers_path = os.path.join(
            os.path.dirname(path), f"answers_{meta['label']}.jsonl"
        )
        answers = [_answer_from_record(r) for r in _read_jsonl(answers_path)]
        run = ArmRun(
            role=role,
            run_index=int(meta["run_index"]),
            answers=answers,
            counters=BlendCounters(**meta["counters"]) if measured else BlendCounters(),
            engagement_measured=measured,
            store_failures=int(meta["store_failures"]),
        )
        record = ArmRecord(
            run=run,
            url=str(meta["url"]),
            blend_metrics_url=str(meta["blend_metrics_url"]),
            engine_version=meta["engine_version"],
            lmcache_version=str(meta["lmcache_version"]),
            started_at=str(meta["started_at"]),
            finished_at=str(meta["finished_at"]),
        )
    except (KeyError, TypeError, ValueError, OSError) as e:
        raise ResultsDirError(f"{path}: {e!r}") from e
    if run.label != meta["label"]:
        raise ResultsDirError(
            f"{path}: label {meta['label']!r} does not match its role"
        )
    qids = [a.sample_id for a in answers]
    if qids != [s.sample.sample_id for s in plan_record.plan.samples]:
        raise ResultsDirError(
            f"{answers_path}: answers do not match the plan's questions"
        )
    return record


def _answer_record(answer: ArmAnswer, measured: bool) -> dict[str, object]:
    """Return one ``answers_<label>.jsonl`` record."""
    return {
        "qid": answer.sample_id,
        "request_id": answer.request_id,
        "successful": answer.successful,
        "answer": answer.answer,
        "response": answer.response,
        "ttft": round(answer.ttft, 4),
        "num_output_tokens": answer.num_output_tokens,
        "counters": asdict(answer.counters) if measured else None,
    }


def _answer_from_record(record: Mapping[str, Any]) -> ArmAnswer:
    """Rebuild one answer from its ``answers_<label>.jsonl`` record."""
    counters = record["counters"]
    return ArmAnswer(
        sample_id=record["qid"],
        request_id=record["request_id"],
        successful=bool(record["successful"]),
        response=record["response"],
        answer=record["answer"],
        ttft=float(record["ttft"]),
        num_output_tokens=int(record["num_output_tokens"]),
        counters=BlendCounters(**counters) if counters else BlendCounters(),
    )


def _read_json(path: str) -> dict[str, Any]:
    """Read one JSON object."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read one JSON object per non-blank line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: str, record: Mapping[str, object]) -> None:
    """Write *record* atomically, so a reader never sees half a file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _write_jsonl(path: str, records: list[dict[str, object]]) -> None:
    """Write *records* atomically, one JSON object per line."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    os.replace(tmp, path)
