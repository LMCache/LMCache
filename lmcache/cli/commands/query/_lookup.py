# SPDX-License-Identifier: Apache-2.0
"""Cache-coverage lookup for ``lmcache query kvcache``.

The coordinator's ``POST /directory/lookup`` resolves a request's full token
sequence into one entry per chunk key and reports the placements known for
each key. This module turns that reply into the coverage numbers the command
prints: how many leading chunks are resident, how many tokens that prefix
covers, and which tiers hold it.

Resolution runs through prefix-chained chunk hashes, so the token sequence has
to be the one the engine actually prefilled. A tokenizer that disagrees with
the engine's yields different chunk keys and therefore a lower coverage — it
never overstates what is cached.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import json
import urllib.error
import urllib.request

# First Party
from lmcache.cli.commands.query._coordinator import normalize_url

_DEFAULT_TIMEOUT = 10.0


@dataclass(frozen=True)
class CoverageResult:
    """How much of one prompt the KV cache holds.

    Attributes:
        prompt_tokens: Tokens in the prompt, as tokenized locally.
        total_chunks: Complete chunks the coordinator resolved. A trailing
            partial chunk is not one of them.
        cached_chunks: Length of the leading run of chunks that every rank
            of every object group has a placement for.
        cached_tokens: Tokens covered by that leading run.
        tier_chunks: ``(tier, chunks)`` pairs over the cached prefix, most
            chunks first.
        locations: Distinct ``instance: tier/backend`` placements over the
            cached prefix, in the order the coordinator reported them.
        status: ``"HIT"`` when the whole prompt is resident, ``"MISS"`` when
            no chunk is, and ``"HIT (partial)"`` in between.
    """

    prompt_tokens: int
    total_chunks: int
    cached_chunks: int
    cached_tokens: int
    tier_chunks: tuple[tuple[str, int], ...]
    locations: tuple[str, ...]
    status: str


def post_lookup(
    url: str,
    payload: Mapping[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Send one lookup to the coordinator and parse the reply.

    Args:
        url: Coordinator base URL; a missing scheme is taken as ``http``.
        payload: JSON body — either the ``keys`` form or the ``token_ids``
            form accepted by ``POST /directory/lookup``.
        timeout: Seconds to wait for the reply.

    Returns:
        The parsed reply.

    Raises:
        RuntimeError: On a connection failure, a non-2xx status, a timeout,
            or a reply that is not JSON. The message is meant for the user,
            so the command can print it instead of a traceback.
    """
    request = urllib.request.Request(
        f"{normalize_url(url)}/directory/lookup",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (json.JSONDecodeError, AttributeError, ValueError):
            pass
        raise RuntimeError(f"Coordinator returned {e.code}: {detail}") from e
    except TimeoutError as e:
        raise RuntimeError(f"Timed out contacting {url} after {timeout}s") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach {url} -- is the coordinator running? ({e.reason})"
        ) from e

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Coordinator reply was not JSON: {body[:200]}") from e


def _placement_label(placement: Mapping[str, Any]) -> str:
    """Render one placement as ``instance: tier/backend``.

    Fleet-shared backends report no owning instance; they are labelled
    ``(shared)`` rather than being dropped, since the reader still needs to
    know the tiers involved.
    """
    instance = placement.get("instance_id") or "(shared)"
    return f"{instance}: {placement.get('tier')}/{placement.get('backend')}"


def summarize_coverage(
    body: Mapping[str, Any],
    prompt_tokens: int,
) -> CoverageResult:
    """Summarize a ``/directory/lookup`` reply into a :class:`CoverageResult`.

    Args:
        body: Parsed ``DirectoryLookupResponse`` — ``chunks`` plus ``results``,
            the latter holding one entry per chunk key, chunk-major, with one
            entry per rank fan-out inside each chunk.
        prompt_tokens: Tokens in the prompt, for the report only.

    Returns:
        The coverage of the prompt's leading resident run.

    The prefix — not the total number of resident chunks — is what a
    requesting engine can actually reuse: chunk hashes are chained, so a gap
    makes everything behind it unreachable for that prompt.
    """
    chunks = int(body.get("chunks", 0))
    results = body.get("results") or []
    # A prompt shorter than one chunk resolves to no key at all.
    if chunks <= 0 or len(results) < chunks:
        return CoverageResult(prompt_tokens, 0, 0, 0, (), (), "MISS")

    fan_out = len(results) // chunks
    tier_counts: dict[str, int] = {}
    locations: list[str] = []
    cached_chunks = 0
    cached_tokens = 0

    for chunk in range(chunks):
        entries = results[chunk * fan_out : (chunk + 1) * fan_out]
        if not all(entry.get("placements") for entry in entries):
            break
        cached_chunks += 1
        cached_tokens += len(entries[0].get("token_ids") or [])
        # A chunk pinned in several tiers counts once per tier.
        tiers = {
            str(placement.get("tier"))
            for entry in entries
            for placement in entry["placements"]
        }
        for tier in tiers:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        for entry in entries:
            for placement in entry["placements"]:
                label = _placement_label(placement)
                if label not in locations:
                    locations.append(label)

    if cached_chunks == chunks:
        status = "HIT"
    elif cached_chunks == 0:
        status = "MISS"
    else:
        status = "HIT (partial)"

    tier_chunks = tuple(
        sorted(tier_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return CoverageResult(
        prompt_tokens=prompt_tokens,
        total_chunks=chunks,
        cached_chunks=cached_chunks,
        cached_tokens=cached_tokens,
        tier_chunks=tier_chunks,
        locations=tuple(locations),
        status=status,
    )
