# SPDX-License-Identifier: Apache-2.0
"""Check vLLM sleep and wake-up with LMCache in multiprocess mode.

vLLM's extensible KV cache is CUDA VMM memory that the LMCache server imports.
An import keeps the physical memory alive, so sleep frees the KV cache only
after the connector makes the server drop its import, and after wake-up the
server has to import the memory vLLM committed anew. This script drives that
round trip against a running vLLM and LMCache server and checks it from the
outside:

1. The extensible KV cache is on (vLLM ``cache_config_info`` metric).
2. A cold prompt is stored to LMCache. After a reset of vLLM's own prefix
   cache the same prompt is served from LMCache with the same output.
3. Sleep with ``reset_connector=false``: vLLM reports sleeping, the LMCache
   server lists no registered KV cache, and GPU memory in use drops.
4. Wake-up: the KV cache is registered again and GPU memory in use is not
   above the level before sleep.
5. The prompt is served from LMCache again with the same output, so the
   server reads and writes the memory vLLM uses now.
6. Sleep with ``reset_connector=true`` and wake-up: same checks as 3 and 4.
7. The prompt misses LMCache, because the reset cleared it, and is stored
   again. After another prefix-cache reset it hits with the same output.

LMCache stores and returns whole chunks, so the expected counts come from the
prompt's token count and the server's chunk size: a store is complete when
every full chunk of the prompt is written, and a hit covers exactly those
chunks.

In engine-driven transfer the server imports no GPU memory. It registers a
shared-memory staging context per vLLM instance instead, which the same two
connector hooks drop and re-create, so the steps are the same and the
registration checks read that registration.

A failed check does not stop the run, so one run shows every broken step. A
request that vLLM rejects does stop it, because the later steps depend on it.
"""

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

# About 1,900 tokens for Qwen tokenizers: several LMCache chunks at the
# default chunk size of 256 tokens.
PROMPT = (
    "Document A. "
    + "The Turing machine is a fundamental concept in theoretical computer "
    "science that defines an abstract machine capable of manipulating symbols "
    "on a strip of tape according to a table of rules. "
    * 60
    + "\nQuestion: In one sentence, what is a Turing machine?\nAnswer:"
)
MAX_TOKENS = 32

EXTERNAL_HIT_TOKENS = "vllm:external_prefix_cache_hits_total"
L1_WRITE_CHUNKS = "lmcache_mp_l1_write_chunks_total"
L1_READ_CHUNKS = "lmcache_mp_l1_read_chunks_total"

# Key of the LMCache server's /status that lists the registered instance ids,
# per transfer mode. Only the module of the active mode reports its key.
REGISTRATION_STATUS_KEYS = {
    "lmcache_driven": "registered_gpu_ids",
    "engine_driven": "registered_non_cuda_instance_ids",
}

# Level 1 sleep releases the KV cache and moves the weights to host memory, so
# little more than the CUDA context stays on the device. A server that keeps
# its import leaves most of the KV cache allocated, far above this fraction of
# the memory in use before sleep.
SLEEP_MAX_FRACTION = 0.25
# Wake-up commits the KV cache again at its former size. If the old chunks
# were still alive, the new ones would add to them and exceed this fraction.
WAKE_MAX_FRACTION = 1.10

REQUEST_TIMEOUT_SECONDS = 300.0
STATE_TIMEOUT_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 0.5


def _http(method: str, url: str, body: dict[str, object] | None = None) -> str:
    """Send one request and return the response body.

    Raises:
        RuntimeError: The server answered with an error status.
    """
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            return response.read().decode()
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"{method} {url} returned HTTP {error.code}: {error.read().decode()}"
        ) from error


def _read_counter(metrics_text: str, name: str) -> int:
    """Sum every sample of one counter in a Prometheus text exposition."""
    total = 0.0
    for line in metrics_text.splitlines():
        sample, _, value = line.rpartition(" ")
        if sample.split("{", 1)[0] == name:
            total += float(value)
    return int(total)


def _wait_for(condition: Callable[[], bool], timeout: float) -> bool:
    """Poll a condition and return whether it held within the timeout."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_INTERVAL_SECONDS)
    return True


@dataclass(frozen=True)
class Counters:
    """Cache counters of the deployment, or the change between two readings."""

    external_hit_tokens: int
    stored_chunks: int
    read_chunks: int

    def since(self, earlier: "Counters") -> "Counters":
        """Return the change from an earlier reading to this one."""
        return Counters(
            self.external_hit_tokens - earlier.external_hit_tokens,
            self.stored_chunks - earlier.stored_chunks,
            self.read_chunks - earlier.read_chunks,
        )


@dataclass(frozen=True)
class Completion:
    """Generated text of one completion and the size of its prompt."""

    text: str
    prompt_tokens: int


@dataclass(frozen=True)
class CachedPrefix:
    """The part of a prompt that LMCache stores: its full chunks."""

    chunks: int
    tokens: int


@dataclass(frozen=True)
class Generation:
    """Output of one completion and the cache activity it caused."""

    text: str
    prefix: CachedPrefix
    activity: Counters


@dataclass(frozen=True)
class Check:
    """Outcome of one check."""

    name: str
    passed: bool
    detail: str


class Report:
    """Collects check outcomes and prints each one as it is recorded."""

    def __init__(self) -> None:
        self._checks: list[Check] = []

    def check(self, name: str, passed: bool, detail: str) -> None:
        """Record and print one check.

        Args:
            name: What the check verifies.
            passed: Whether it held.
            detail: The measured values behind the outcome.
        """
        self._checks.append(Check(name, passed, detail))
        print(f"{'PASS' if passed else 'FAIL'}: {name} ({detail})", flush=True)

    def failures(self) -> list[Check]:
        """Return the checks that did not hold."""
        return [check for check in self._checks if not check.passed]

    def write(self, path: Path) -> None:
        """Write every recorded check to a JSON file."""
        path.write_text(json.dumps([asdict(check) for check in self._checks], indent=2))


class Deployment:
    """The running vLLM server, LMCache server and the GPU they share."""

    def __init__(
        self,
        engine_url: str,
        lmcache_url: str,
        model: str,
        gpu_index: int,
        transfer_mode: str,
    ) -> None:
        self._engine_url = engine_url
        self._lmcache_url = lmcache_url
        self._model = model
        self._gpu_index = gpu_index
        self._registration_key = REGISTRATION_STATUS_KEYS[transfer_mode]

    def extensible_kv_cache_enabled(self) -> bool:
        """Return whether vLLM reports its extensible KV cache as enabled."""
        metrics = _http("GET", f"{self._engine_url}/metrics")
        return (
            re.search(
                r'^vllm:cache_config_info\{.*enable_extensible_kv_cache="True"',
                metrics,
                re.MULTILINE,
            )
            is not None
        )

    def counters(self) -> Counters:
        """Read the cache counters of vLLM and the LMCache server."""
        engine = _http("GET", f"{self._engine_url}/metrics")
        lmcache = _http("GET", f"{self._lmcache_url}/metrics")
        return Counters(
            external_hit_tokens=_read_counter(engine, EXTERNAL_HIT_TOKENS),
            stored_chunks=_read_counter(lmcache, L1_WRITE_CHUNKS),
            read_chunks=_read_counter(lmcache, L1_READ_CHUNKS),
        )

    def chunk_size(self) -> int:
        """Return the number of tokens in one LMCache chunk."""
        return json.loads(_http("GET", f"{self._lmcache_url}/status"))["chunk_size"]

    def complete(self, prompt: str) -> Completion:
        """Run one greedy completion."""
        response = json.loads(
            _http(
                "POST",
                f"{self._engine_url}/v1/completions",
                {
                    "model": self._model,
                    "prompt": prompt,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0,
                    "seed": 0,
                },
            )
        )
        return Completion(
            response["choices"][0]["text"], response["usage"]["prompt_tokens"]
        )

    def reset_prefix_cache(self) -> None:
        """Clear vLLM's own prefix cache and leave LMCache as it is."""
        _http("POST", f"{self._engine_url}/reset_prefix_cache")

    def sleep(self, reset_connector: bool) -> None:
        """Put vLLM into level 1 sleep.

        Args:
            reset_connector: Whether vLLM also clears the LMCache server.
        """
        flag = "true" if reset_connector else "false"
        _http("POST", f"{self._engine_url}/sleep?level=1&reset_connector={flag}")

    def wake_up(self) -> None:
        """Wake vLLM up, weights and KV cache."""
        _http("POST", f"{self._engine_url}/wake_up")

    def is_sleeping(self) -> bool:
        """Return whether vLLM reports that it is asleep."""
        return json.loads(_http("GET", f"{self._engine_url}/is_sleeping"))[
            "is_sleeping"
        ]

    def registered_kv_caches(self) -> list[int]:
        """Return the instance ids whose KV cache the LMCache server holds."""
        status = json.loads(_http("GET", f"{self._lmcache_url}/status"))
        return status[self._registration_key]

    def gpu_used_mib(self) -> int:
        """Return the memory in use on the GPU, in MiB."""
        output = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(self._gpu_index),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return int(output.strip())


def stored(activity: Counters, prefix: CachedPrefix) -> bool:
    """Tell whether every full chunk of the prompt has been written."""
    return activity.stored_chunks >= prefix.chunks


def hit(activity: Counters, prefix: CachedPrefix) -> bool:
    """Tell whether vLLM took every full chunk of the prompt from LMCache."""
    return activity.external_hit_tokens >= prefix.tokens


def hit_from_l1(activity: Counters, prefix: CachedPrefix) -> bool:
    """Tell whether the hit is complete and the server reports its L1 reads."""
    return hit(activity, prefix) and activity.read_chunks > 0


def run_prompt(
    deployment: Deployment,
    chunk_size: int,
    complete: Callable[[Counters, CachedPrefix], bool],
) -> Generation:
    """Send the prompt and wait for the cache activity it should cause.

    A store finishes after the response and the counters are exported with a
    delay, so they are polled until the awaited activity is complete.

    Args:
        deployment: The servers under test.
        chunk_size: Tokens per LMCache chunk.
        complete: Given the counter changes and the prompt's cached prefix,
            tells whether the awaited activity is complete.

    Returns:
        The output, the cached prefix and the counter changes, which fall
        short of the awaited activity if it did not complete in time.
    """
    before = deployment.counters()
    completion = deployment.complete(PROMPT)
    full_chunks = completion.prompt_tokens // chunk_size
    prefix = CachedPrefix(full_chunks, full_chunks * chunk_size)
    activity = before.since(before)

    def reached() -> bool:
        nonlocal activity
        activity = deployment.counters().since(before)
        return complete(activity, prefix)

    _wait_for(reached, STATE_TIMEOUT_SECONDS)
    return Generation(completion.text, prefix, activity)


def check_output(report: Report, step: str, cold_text: str, text: str) -> None:
    """Check that a completion reproduces the output of the first, cold one.

    Args:
        report: Receives the check outcome.
        step: Label that prefixes the check name.
        cold_text: Output of the cold prompt before any sleep.
        text: Output to compare.
    """
    same = text == cold_text
    report.check(
        f"{step}: output equals the cold output",
        same,
        "identical" if same else f"cold {cold_text!r}, got {text!r}",
    )


def check_sleep(
    deployment: Deployment,
    report: Report,
    step: str,
    reset_connector: bool,
    awake_mib: int,
) -> None:
    """Put vLLM to sleep and check that the KV cache memory is released.

    Args:
        deployment: The servers under test.
        report: Receives the check outcomes.
        step: Label that prefixes the check names.
        reset_connector: Whether sleep also clears the LMCache server.
        awake_mib: GPU memory in use before the first sleep, in MiB.
    """
    deployment.sleep(reset_connector=reset_connector)
    sleeping = deployment.is_sleeping()
    report.check(f"{step}: vLLM reports sleeping", sleeping, f"is_sleeping={sleeping}")

    unregistered = _wait_for(
        lambda: not deployment.registered_kv_caches(), STATE_TIMEOUT_SECONDS
    )
    report.check(
        f"{step}: LMCache server holds no KV cache registration",
        unregistered,
        f"registered ids {deployment.registered_kv_caches()}",
    )

    limit_mib = int(awake_mib * SLEEP_MAX_FRACTION)
    released = _wait_for(
        lambda: deployment.gpu_used_mib() <= limit_mib, STATE_TIMEOUT_SECONDS
    )
    report.check(
        f"{step}: GPU memory is released",
        released,
        f"{deployment.gpu_used_mib()} MiB in use, limit {limit_mib} MiB, "
        f"{awake_mib} MiB before sleep",
    )


def check_wake_up(
    deployment: Deployment, report: Report, step: str, awake_mib: int
) -> None:
    """Wake vLLM up and check that the KV cache is registered again.

    Args:
        deployment: The servers under test.
        report: Receives the check outcomes.
        step: Label that prefixes the check names.
        awake_mib: GPU memory in use before the first sleep, in MiB.
    """
    deployment.wake_up()
    sleeping = deployment.is_sleeping()
    report.check(f"{step}: vLLM reports awake", not sleeping, f"is_sleeping={sleeping}")

    registered = _wait_for(
        lambda: len(deployment.registered_kv_caches()) == 1, STATE_TIMEOUT_SECONDS
    )
    report.check(
        f"{step}: LMCache server holds one KV cache registration",
        registered,
        f"registered ids {deployment.registered_kv_caches()}",
    )

    used_mib = deployment.gpu_used_mib()
    limit_mib = int(awake_mib * WAKE_MAX_FRACTION)
    report.check(
        f"{step}: GPU memory is not above the level before sleep",
        used_mib <= limit_mib,
        f"{used_mib} MiB in use, limit {limit_mib} MiB, {awake_mib} MiB before sleep",
    )


def run_checks(deployment: Deployment, report: Report) -> None:
    """Run the whole sleep and wake-up scenario.

    Args:
        deployment: The servers under test.
        report: Receives the check outcomes.

    Raises:
        RuntimeError: vLLM or the LMCache server rejected a request.
    """
    extensible = deployment.extensible_kv_cache_enabled()
    report.check(
        "vLLM uses the extensible KV cache",
        extensible,
        f'cache_config_info enable_extensible_kv_cache="True" present: {extensible}',
    )

    chunk_size = deployment.chunk_size()
    cold = run_prompt(deployment, chunk_size, stored)
    cached = cold.prefix
    report.check(
        "cold prompt is stored to LMCache",
        cached.chunks > 0 and cold.activity.stored_chunks >= cached.chunks,
        f"{cold.activity.stored_chunks} chunks stored, the prompt has "
        f"{cached.chunks} full chunks of {chunk_size} tokens",
    )

    deployment.reset_prefix_cache()
    warm = run_prompt(deployment, chunk_size, hit)
    report.check(
        "prompt hits LMCache before sleep",
        warm.activity.external_hit_tokens == cached.tokens,
        f"{warm.activity.external_hit_tokens} tokens from LMCache, "
        f"expected {cached.tokens}",
    )
    check_output(report, "LMCache hit before sleep", cold.text, warm.text)

    awake_mib = deployment.gpu_used_mib()
    check_sleep(
        deployment,
        report,
        "sleep keeping LMCache",
        reset_connector=False,
        awake_mib=awake_mib,
    )
    check_wake_up(deployment, report, "wake-up keeping LMCache", awake_mib)

    # Sleep clears vLLM's own prefix cache, so this hit is retrieved from
    # LMCache into the memory committed at wake-up, through the registration
    # made there.
    kept = run_prompt(deployment, chunk_size, hit_from_l1)
    report.check(
        "prompt hits LMCache after wake-up",
        kept.activity.external_hit_tokens == cached.tokens
        and kept.activity.read_chunks > 0,
        f"{kept.activity.external_hit_tokens} tokens from LMCache, "
        f"expected {cached.tokens}, {kept.activity.read_chunks} chunks read",
    )
    check_output(report, "LMCache hit after wake-up", cold.text, kept.text)

    check_sleep(
        deployment,
        report,
        "sleep clearing LMCache",
        reset_connector=True,
        awake_mib=awake_mib,
    )
    check_wake_up(deployment, report, "wake-up clearing LMCache", awake_mib)

    restored = run_prompt(deployment, chunk_size, stored)
    report.check(
        "prompt misses the cleared LMCache and is stored again",
        restored.activity.external_hit_tokens == 0
        and restored.activity.stored_chunks >= cached.chunks,
        f"{restored.activity.external_hit_tokens} tokens from LMCache, "
        f"{restored.activity.stored_chunks} chunks stored, "
        f"expected at least {cached.chunks}",
    )
    check_output(report, "cold prompt after wake-up", cold.text, restored.text)

    deployment.reset_prefix_cache()
    reread = run_prompt(deployment, chunk_size, hit)
    report.check(
        "prompt stored after wake-up hits LMCache",
        reread.activity.external_hit_tokens == cached.tokens,
        f"{reread.activity.external_hit_tokens} tokens from LMCache, "
        f"expected {cached.tokens}",
    )
    check_output(
        report, "LMCache hit on chunks stored after wake-up", cold.text, reread.text
    )


def main() -> int:
    """Run the scenario and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine-url", required=True, help="vLLM base URL")
    parser.add_argument("--lmcache-url", required=True, help="LMCache HTTP base URL")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument(
        "--gpu-index", required=True, type=int, help="nvidia-smi index of vLLM's GPU"
    )
    parser.add_argument(
        "--transfer-mode",
        required=True,
        choices=sorted(REGISTRATION_STATUS_KEYS),
        help="transfer mode the LMCache server was started with",
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path, help="directory for checks.json"
    )
    args = parser.parse_args()

    report = Report()
    try:
        run_checks(
            Deployment(
                args.engine_url,
                args.lmcache_url,
                args.model,
                args.gpu_index,
                args.transfer_mode,
            ),
            report,
        )
    finally:
        report.write(args.results_dir / "checks.json")

    failures = report.failures()
    if failures:
        print(f"FAIL: {len(failures)} sleep-mode checks failed:")
        for failure in failures:
            print(f"  - {failure.name} ({failure.detail})")
        return 1
    print("PASS: sleep and wake-up kept LMCache working and released GPU memory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
