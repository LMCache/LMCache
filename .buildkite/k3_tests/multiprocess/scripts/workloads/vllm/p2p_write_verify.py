# SPDX-License-Identifier: Apache-2.0
"""Drive and check a main LMCache server that writes into a headless peer.

One engine talks to the main server, which runs with ``--enable-p2p-write``.
The headless peer has no engine, so every object it holds was written by the
main server over P2P.

Fill: distinct long prompts are sent one at a time until the two servers
together hold more KV than the main server's L1 can. Replay: every prompt is
sent again. The engine runs without its own prefix cache, so each cached token
on replay was served by LMCache.

All checks run before the script exits, and the per-request rows are written
to ``<results-dir>/p2p_write.json``.
"""

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import argparse
import json
import random
import sys
import time
import urllib.request

_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
).split()
_POLL_SECONDS = 0.2
_SETTLE_TIMEOUT_SECONDS = 60.0


@dataclass
class ServerState:
    """L1 contents of one LMCache server, taken from its ``/status``."""

    objects: int
    used_bytes: int
    total_bytes: int
    busy: bool


@dataclass
class Row:
    """One request and where the two servers' objects are after it."""

    prompt: int
    prompt_tokens: int
    cached_tokens: int
    text: str
    main_objects: int
    main_used_bytes: int
    main_total_bytes: int
    peer_objects: int
    peer_used_bytes: int
    main_objects_added: int
    peer_objects_added: int
    peer_used_bytes_added: int

    @property
    def main_usage(self) -> float:
        """Used fraction of the main server's L1 after the request."""
        return self.main_used_bytes / self.main_total_bytes

    @property
    def hit_ratio(self) -> float:
        """Share of the prompt's tokens that LMCache served."""
        return self.cached_tokens / self.prompt_tokens

    @property
    def stored_on(self) -> str:
        """Which server received the objects this request stored."""
        if self.main_objects_added > 0 and self.peer_objects_added > 0:
            return "both"
        if self.peer_objects_added > 0:
            return "peer"
        if self.main_objects_added > 0:
            return "main"
        return "nowhere"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", required=True)
    parser.add_argument("--main-url", required=True, help="main server HTTP API")
    parser.add_argument("--peer-url", required=True, help="peer server HTTP API")
    parser.add_argument("--model", required=True)
    parser.add_argument("--usage-threshold", type=float, required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--prompt-words", type=int, default=3000)
    parser.add_argument("--max-prompts", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--min-hit-ratio", type=float, default=0.85)
    return parser.parse_args()


def _prompt_text(prompt: int, num_words: int) -> str:
    rng = random.Random(prompt)
    stream = " ".join(rng.choice(_WORDS) for _ in range(num_words))
    return (
        f"Run tag {prompt}. Read the following stream of words, then answer."
        f"\n\n{stream}"
        "\n\nQuestion: what is the first word of the stream? Answer with that "
        "word only."
    )


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def _server_state(base_url: str) -> ServerState:
    storage = _get_json(f"{base_url}/status")["storage_manager"]
    l1 = storage["l1_manager"]
    in_flight_stores = sum(
        adapter.get("in_flight_stores") or 0 for adapter in storage["l2_adapters"]
    )
    return ServerState(
        objects=l1["total_object_count"],
        used_bytes=l1["memory_used_bytes"],
        total_bytes=l1["memory_total_bytes"],
        busy=bool(
            in_flight_stores
            or l1["staging_object_count"]
            or l1["temporary_count"]
            or l1["read_locked_count"]
        ),
    )


def _wait_until(
    args: argparse.Namespace,
    done: Callable[[ServerState, ServerState], bool],
    what: str,
) -> tuple[ServerState, ServerState]:
    """Poll both servers until ``done(main, peer)`` holds."""
    deadline = time.monotonic() + _SETTLE_TIMEOUT_SECONDS
    while True:
        main, peer = _server_state(args.main_url), _server_state(args.peer_url)
        if done(main, peer):
            return main, peer
        if time.monotonic() > deadline:
            raise TimeoutError(f"{what} (main {main}, peer {peer})")
        time.sleep(_POLL_SECONDS)


def _settle(args: argparse.Namespace) -> tuple[ServerState, ServerState]:
    """Wait until no store, read or transfer is in progress on either server."""
    idle_polls = 0
    last: tuple[int, int] | None = None

    def idle(main: ServerState, peer: ServerState) -> bool:
        nonlocal idle_polls, last
        counts = (main.objects, peer.objects)
        quiet = not main.busy and not peer.busy and counts == last
        idle_polls = idle_polls + 1 if quiet else 0
        last = counts
        return idle_polls >= 3

    return _wait_until(args, idle, "the servers did not become idle")


def _send(args: argparse.Namespace, prompt: int, expect_new_objects: bool) -> Row:
    main_before, peer_before = _settle(args)
    body = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": _prompt_text(prompt, args.prompt_words)}
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "kv_transfer_params": {"cached_token_stats": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{args.engine_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        answer = json.load(response)
    if expect_new_objects:
        stored_before = main_before.objects + peer_before.objects
        _wait_until(
            args,
            lambda main, peer: main.objects + peer.objects > stored_before,
            f"prompt {prompt} stored no object",
        )
    main, peer = _settle(args)
    stats = answer["kv_transfer_params"]["cached_token_stats"]
    return Row(
        prompt=prompt,
        prompt_tokens=answer["usage"]["prompt_tokens"],
        cached_tokens=stats["num_lmcache_cached_tokens"],
        text=answer["choices"][0]["message"]["content"],
        main_objects=main.objects,
        main_used_bytes=main.used_bytes,
        main_total_bytes=main.total_bytes,
        peer_objects=peer.objects,
        peer_used_bytes=peer.used_bytes,
        main_objects_added=main.objects - main_before.objects,
        peer_objects_added=peer.objects - peer_before.objects,
        peer_used_bytes_added=peer.used_bytes - peer_before.used_bytes,
    )


def _print_row(phase: str, row: Row, stored_on: str) -> None:
    print(
        f"[{phase}] prompt {row.prompt}: KV stored on {stored_on:7s} | "
        f"{row.cached_tokens}/{row.prompt_tokens} tokens from LMCache | "
        f"main L1 {row.main_objects} objects, {row.main_usage:.0%} used | "
        f"peer L1 {row.peer_objects} objects, "
        f"{row.peer_used_bytes / 2**30:.2f} GiB | answer {row.text!r}",
        flush=True,
    )


def holds_more_than_main_l1(row: Row) -> bool:
    """Return whether the two servers hold more KV than the main L1 can."""
    return row.main_used_bytes + row.peer_used_bytes > row.main_total_bytes


def verify(
    fill: list[Row],
    replay: list[Row],
    usage_threshold: float,
    min_hit_ratio: float,
) -> list[str]:
    """Check a fill and a replay of the same prompts.

    Args:
        fill: One row per distinct prompt, in the order they were first sent.
        replay: One row per prompt of ``fill``, in the same order.
        usage_threshold: The main server's ``--p2p-write-l1-usage-threshold``.
        min_hit_ratio: Smallest share of a replayed prompt's tokens that must
            come from LMCache.

    Returns:
        One message per failed check; empty when the run is healthy.
    """
    failures = []

    if not any(row.stored_on == "main" for row in fill):
        failures.append(
            "no prompt was stored on the main server alone, so the keep-local "
            "case was not exercised"
        )
    for row in fill:
        # Usage the main L1 would have reached had it kept the whole prompt.
        usage_if_kept = (
            row.main_used_bytes + row.peer_used_bytes_added
        ) / row.main_total_bytes
        if row.peer_objects_added > 0 and usage_if_kept < usage_threshold:
            failures.append(
                f"prompt {row.prompt}: {row.peer_objects_added} objects went to "
                f"the peer although keeping them would have left the main L1 "
                f"at {usage_if_kept:.0%}, below the {usage_threshold:.0%} "
                "threshold"
            )
        if row.stored_on == "main" and row.main_usage >= usage_threshold:
            failures.append(
                f"prompt {row.prompt}: nothing went to the peer although the "
                f"main L1 reached {row.main_usage:.0%}, at or above the "
                f"{usage_threshold:.0%} threshold"
            )

    if not holds_more_than_main_l1(fill[-1]):
        failures.append(
            "the fill ended before the two servers held more KV than the main L1 can"
        )
    peer_only = [row.prompt for row in fill if row.stored_on == "peer"]
    if not peer_only:
        failures.append("no prompt was stored on the peer alone")

    for row in fill:
        if row.main_objects_added < 0 or row.peer_objects_added < 0:
            failures.append(
                f"prompt {row.prompt}: objects disappeared during the fill "
                f"(main {row.main_objects_added:+d}, "
                f"peer {row.peer_objects_added:+d})"
            )

    for first, again in zip(fill, replay, strict=True):
        if again.hit_ratio < min_hit_ratio:
            failures.append(
                f"prompt {first.prompt} (stored on {first.stored_on}): only "
                f"{again.cached_tokens} of {again.prompt_tokens} tokens came "
                "from LMCache on replay"
            )
        if again.text != first.text:
            failures.append(
                f"prompt {first.prompt} (stored on {first.stored_on}): answered "
                f"{again.text!r} on replay, {first.text!r} when first sent"
            )

    placed = (fill[-1].main_objects, fill[-1].peer_objects)
    replayed = (replay[-1].main_objects, replay[-1].peer_objects)
    if replayed != placed:
        failures.append(
            "the replay moved objects: main/peer held "
            f"{placed[0]}/{placed[1]} objects after the fill and "
            f"{replayed[0]}/{replayed[1]} after the replay"
        )
    return failures


def main() -> None:
    """Run the fill and the replay against live servers and exit non-zero on
    a failed check."""
    args = _parse_args()

    fill: list[Row] = []
    for prompt in range(args.max_prompts):
        row = _send(args, prompt, expect_new_objects=True)
        fill.append(row)
        _print_row("fill", row, row.stored_on)
        if holds_more_than_main_l1(row):
            break

    replay: list[Row] = []
    for first in fill:
        row = _send(args, first.prompt, expect_new_objects=False)
        replay.append(row)
        _print_row("replay", row, first.stored_on)

    results = Path(args.results_dir) / "p2p_write.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(
        json.dumps(
            {
                "fill": [asdict(row) for row in fill],
                "replay": [asdict(row) for row in replay],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    failures = verify(fill, replay, args.usage_threshold, args.min_hit_ratio)
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        sys.exit(1)
    last = fill[-1]
    print(
        f"PASS: {len(fill)} prompts, {last.main_objects} objects on the main "
        f"server and {last.peer_objects} on the headless peer; every prompt "
        "was served from LMCache on replay"
    )


if __name__ == "__main__":
    main()
