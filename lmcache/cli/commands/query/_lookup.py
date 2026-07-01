# SPDX-License-Identifier: Apache-2.0
"""Cache-coverage lookup helper for ``lmcache query kvcache``.

Turns a prompt into token IDs, asks the controller ``POST /lookup`` how much of
that token sequence is already cached, and summarizes the response into a
:class:`CoverageResult`.
"""

# Standard
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import urllib.error
import urllib.request


def _normalize_url(url: str) -> str:
    """Ensure *url* has an ``http(s)://`` scheme and no trailing slash."""
    url = url.strip()
    if "://" not in url:
        url = f"http://{url}"
    return url.rstrip("/")


@dataclass(frozen=True)
class CoverageResult:
    """Cache-coverage summary for one prompt.

    Attributes:
        prompt_tokens: Total tokens in the prompt.
        cached_tokens: Length of the longest cached prefix (in tokens).
        cache_status: ``"HIT"``, ``"MISS"``, or ``"HIT (partial)"``.
        cached_chunks: Number of cached chunks (derived from ``chunk_size``).
        total_chunks: Total chunks in the prompt (derived from ``chunk_size``).
        locations: ``(instance_id, location)`` pairs holding cached data.
    """

    prompt_tokens: int
    cached_tokens: int
    cache_status: str
    cached_chunks: int
    total_chunks: int
    locations: list[tuple[str, str]]


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return ``ceil(numerator / denominator)``, or ``0`` for a non-positive
    denominator."""
    if denominator <= 0:
        return 0
    return (numerator + denominator - 1) // denominator


def summarize_coverage(
    layout_info: Mapping[str, Sequence[object]],
    total_tokens: int,
    chunk_size: int,
) -> CoverageResult:
    """Summarize a ``/lookup`` ``layout_info`` map into a :class:`CoverageResult`.

    Args:
        layout_info: Map of ``instance_id`` to ``(location, cached_prefix_end)``
            as returned by the controller ``POST /lookup`` endpoint. Values may
            be tuples or (from JSON) lists.
        total_tokens: Total number of tokens in the prompt.
        chunk_size: Tokens per cache chunk, used to derive chunk counts.

    Returns:
        The coverage summary. ``cached_tokens`` is the longest cached prefix
        across all instances; ``cache_status`` is ``MISS`` when nothing is
        cached, ``HIT`` when the whole prompt is cached, else ``HIT (partial)``.
    """
    cached_tokens = 0
    locations: list[tuple[str, str]] = []
    for instance_id, location_end in layout_info.items():
        location = str(location_end[0])
        end = int(str(location_end[1]))
        locations.append((str(instance_id), location))
        cached_tokens = max(cached_tokens, end)

    cached_tokens = min(cached_tokens, total_tokens)
    if cached_tokens <= 0:
        cache_status = "MISS"
    elif cached_tokens >= total_tokens:
        cache_status = "HIT"
    else:
        cache_status = "HIT (partial)"

    return CoverageResult(
        prompt_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cache_status=cache_status,
        cached_chunks=_ceil_div(cached_tokens, chunk_size),
        total_chunks=_ceil_div(total_tokens, chunk_size),
        locations=locations,
    )


class CacheLookup:
    """Read-only cache-coverage client for the controller ``POST /lookup``.

    Tokenizes a prompt with the model's tokenizer, asks the controller how much
    of the token sequence is cached, and summarizes the result.
    """

    def __init__(
        self,
        url: str,
        model: str,
        chunk_size: int = 256,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the lookup client.

        Args:
            url: Base URL of the controller HTTP server.
            model: Tokenizer/model id used to derive token IDs.
            chunk_size: Tokens per cache chunk, used to derive chunk counts.
            timeout: HTTP timeout in seconds.
        """
        self._url = _normalize_url(url)
        self._model = model
        self._chunk_size = chunk_size
        self._timeout = timeout

    def request_lookup(self, tokens: list[int]) -> dict[str, Sequence[object]]:
        """POST *tokens* to ``/lookup`` and return the ``layout_info`` map.

        Args:
            tokens: Token IDs to look up.

        Returns:
            The ``layout_info`` map from the response: ``{instance_id:
            [location, cached_prefix_end]}``.

        Raises:
            RuntimeError: On connection failure, HTTP error, invalid JSON, or a
                response missing ``layout_info``.
        """
        endpoint = f"{self._url}/lookup"
        body = json.dumps({"tokens": tokens}).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:512]
            raise RuntimeError(
                f"POST {endpoint} lookup failed (HTTP {exc.code}): "
                f"{detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"POST {endpoint} lookup failed: {getattr(exc, 'reason', exc)}"
            ) from exc

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from lookup {endpoint}: {exc}") from exc
        layout_info = obj.get("layout_info")
        if layout_info is None:
            raise RuntimeError(f"lookup response missing 'layout_info': {raw[:256]}")
        return layout_info

    def tokenize(self, prompt: str) -> list[int]:
        """Tokenize *prompt* into token IDs using the model's tokenizer.

        Args:
            prompt: The expanded prompt text.

        Returns:
            The list of token IDs.

        Raises:
            RuntimeError: If ``transformers`` is unavailable or the tokenizer
                cannot be loaded (e.g. a gated model without credentials).
        """
        try:
            # Third Party
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required to tokenize the prompt; "
                "install it with `pip install transformers`."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model)
        except Exception as exc:  # noqa: BLE001 - surface any load failure
            raise RuntimeError(
                f"Could not load tokenizer for {self._model!r}: {exc}. "
                "For gated models, run `huggingface-cli login` first."
            ) from exc

        return list(tokenizer.encode(prompt))

    def run(self, prompt: str) -> CoverageResult:
        """Tokenize *prompt*, look up its cache coverage, and summarize it.

        Args:
            prompt: The expanded prompt text.

        Returns:
            The cache-coverage summary.

        Raises:
            RuntimeError: On tokenization or lookup failure.
        """
        tokens = self.tokenize(prompt)
        layout_info = self.request_lookup(tokens)
        return summarize_coverage(layout_info, len(tokens), self._chunk_size)
