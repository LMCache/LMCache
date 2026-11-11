# SPDX-License-Identifier: Apache-2.0
"""
Public API for LMCacheRequestStream, a wrapper of a logical request going
through the SDK.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
import time
import uuid

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.sdk.context import (
    LMCacheSDKCacheKind,
    LMCacheSDKContext,
    ModifyFnType,
    TensorDict,
)

logger = init_logger(__name__)


class LMCacheRequestStreamError(RuntimeError):
    """Raised when a LMCacheRequestStream operation fails."""


@dataclass(frozen=True)
class StreamPerfMetrics:
    """Performance metrics for a single generate() call.

    Args:
        duration: Time taken for the generate() call, in seconds.
        input_tokens: number of input tokens (prompt + suffix).
        output_tokens: Number of tokens generated in this generate() call.
        input_tput: Input tokens per second over this call.
        output_tput: Generated tokens per second over this call.
        tpot: List of times between generated tokens, in seconds.
        ttft: Time to first token, in seconds.
    """

    duration: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    input_tput: float = 0.0
    output_tput: float = 0.0
    tpot: list[float] = field(default_factory=list)
    ttft: float = 0.0


@dataclass(frozen=True)
class TokenEvent:
    """One decoded token reported by the injected post_completion callable.

    Args:
        token_id: The generated token id.
        text: The decoded text for this token.
    """

    token_id: int
    text: str = ""


class PostCompletion(Protocol):
    """Inference function (callable) to be called by request stream."""

    def __call__(
        self,
        prompt_token_ids: list[int],
        sampling_params: dict[str, Any],
        cache_salt: str,
    ) -> Iterable[TokenEvent]:
        """Stream decoded tokens for the given prompt.

        Args:
            prompt_token_ids: Prompt, encoded to list of token ids.
            sampling_params: Sampling parameters for generation.
            cache_salt: Per-user isolation salt, or empty string.

        Returns:
            An iterable yielding one TokenEvent per decoded token.
        """
        ...


def create_request(
    contexts: Iterable[LMCacheSDKContext],
    post_completion: PostCompletion,
    prompt_token_ids: Sequence[int],
    cache_salt: str = "",
) -> LMCacheRequestStream:
    """Create an LMCacheRequestStream for a new request.

    Args:
        contexts: The LMCache SDK contexts used for retrieve/store.
        post_completion: Callable that submits a request to the engine.
        prompt_token_ids: Initial prompt token ids.
        cache_salt: Per-user isolation salt, or empty string.

    Returns:
        A new LMCacheRequestStream.
    """
    return LMCacheRequestStream(
        contexts=contexts,
        post_completion=post_completion,
        prompt_token_ids=prompt_token_ids,
        cache_salt=cache_salt,
    )


class LMCacheRequestStream:
    """Handle for one logical request spanning multiple inference passes.

    Args:
        contexts: The LMCache SDK contexts used for retrieve/store.
        post_completion: Callable for submitting request to inference engine.
        prompt_token_ids: Initial prompt token ids.
        cache_salt: Per-user isolation salt, or empty string.
    """

    def __init__(
        self,
        contexts: Iterable[LMCacheSDKContext],
        post_completion: PostCompletion,
        prompt_token_ids: Sequence[int],
        cache_salt: str = "",
    ) -> None:
        """Initialize per-request state from the initial prompt.

        Beyond the constructor args, sets up: tokens (the live sequence backing
        the KV, starting as the prompt), done (the EOS flag), and internal
        output history / suffix-token / cached-prefix bookkeeping.
        """
        self._contexts: dict[LMCacheSDKCacheKind, LMCacheSDKContext] = {}
        for ctx in contexts:
            self._contexts[ctx.kind] = ctx
        self.post_completion = post_completion
        self.cache_salt = cache_salt
        self.tokens: list[int] = list(prompt_token_ids)
        self.done: bool = False
        self._decoded: int = 0
        self._text_parts: list[str] = []
        # Kept apart from tokens: generate() appends to tokens, while update()
        # and modify_kv() replace or truncate it, so it is not output history.
        self._output_tokens: list[int] = []
        self._request_stream_id: str = str(uuid.uuid4())
        self._suffix_tokens: list[int] = []
        # First token the latest generate() computed (chunk-aligned); the
        # query chain of that pass is keyed from it.
        self._segment_start_token_id: int = 0

    @property
    def request_stream_id(self) -> str:
        """Return the unique stream id."""
        return self._request_stream_id

    @property
    def suffix_tokens(self) -> list[int]:
        """Return a copy of the suffix tokens to be appended to the prompt.

        Mutating the returned list does not change what the next generate()
        sends.
        """
        return list(self._suffix_tokens)

    @property
    def decoded_tokens(self) -> int:
        """Return cumulative tokens decoded across all segments so far."""
        return self._decoded

    @property
    def output_text(self) -> str:
        """Return the concatenated generated text across all segments."""
        return "".join(self._text_parts)

    @property
    def output_tokens(self) -> list[int]:
        """Return a copy of the concatenated generated tokens across all
        segments.

        Only token ids yielded by post_completion are included: never the
        prompt, suffix tokens, or a sequence installed by update() or
        modify_kv(). The length always equals decoded_tokens. Mutating the
        returned list does not change the stream's history.
        """
        return list(self._output_tokens)

    @property
    def is_done(self) -> bool:
        """Return True if the stream has finished generating."""
        return self.done

    def generate(
        self, sampling_params: dict[str, Any], suffix_tokens: Sequence[int] = ()
    ) -> StreamPerfMetrics:
        """Run one inference pass and append the result to the stream history.

        Args:
            sampling_params: Engine sampling params.
            suffix_tokens: Extra tokens not fit into chunk_size.

        Returns:
            StreamPerfMetrics for this call (duration, token counts,
            throughputs, ttft, tpot — all times in seconds).

        Raises:
            LMCacheRequestStreamError: If post_completion fails mid-stream.
        """
        pending = self._suffix_tokens + list(suffix_tokens)
        self._suffix_tokens = []
        if pending:
            self.tokens.extend(pending)

        # Need to take note of the token id where the generate() starts computing,
        # to get the key for the query tensor.
        self._segment_start_token_id = self._cached_prefix_len()

        events = self.post_completion(self.tokens, sampling_params, self.cache_salt)

        input_tokens = len(self.tokens)

        gen_tokens: list[int] = []
        gen_texts: list[str] = []
        start_time = time.perf_counter()
        last_token_time = start_time
        time_between_tokens = []
        try:
            for event in events:
                gen_tokens.append(event.token_id)
                gen_texts.append(event.text)
                time_between_tokens.append(time.perf_counter() - last_token_time)
                last_token_time = time.perf_counter()
        except Exception as e:
            raise LMCacheRequestStreamError(
                f"Stream {self.request_stream_id} failed during generation: {e}"
            ) from e
        finally:
            self._decoded += len(gen_tokens)
            self._text_parts.extend(gen_texts)
            self._output_tokens.extend(gen_tokens)
            self.tokens.extend(gen_tokens)

        # produces less than max_tokens --> EOS
        output_tokens = len(gen_tokens)
        max_tokens = sampling_params.get("max_tokens", 1)
        self.done = output_tokens < max_tokens

        total_time = time.perf_counter() - start_time

        return StreamPerfMetrics(
            duration=total_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tput=input_tokens / total_time if total_time > 0 else 0.0,
            output_tput=output_tokens / total_time if total_time > 0 else 0.0,
            tpot=time_between_tokens[1:] if len(time_between_tokens) > 1 else [],
            ttft=time_between_tokens[0] if time_between_tokens else 0.0,
        )

    def _cached_prefix_len(self) -> int:
        """The number of available cached tokens for the current request stream.

        Returns:
            The cached prefix length in tokens, or 0 when the stream has no KV
            context (nothing is cached, so the pass computes from token 0).
        """
        ctx = self._contexts.get(LMCacheSDKCacheKind.KV)
        if ctx is None:
            return 0
        return ctx.lookup(self.tokens, self.cache_salt)

    def _key_origin(self, kind: LMCacheSDKCacheKind) -> int:
        """The first token id to get the key.

        Args:
            kind: The cache kind whose key chain is being addressed.

        Returns:
            The chunk-aligned token index the chain starts at (see
            LMCacheSDKCacheKind.key_origin).

        Raises:
            LMCacheRequestStreamError: If the stream has no context for kind.
        """
        if kind not in self._contexts:
            raise LMCacheRequestStreamError(
                f"no context available for cache kind {kind}"
            )
        return kind.key_origin(self._segment_start_token_id)

    def retrieve(
        self,
        kind: LMCacheSDKCacheKind,
        timeout: float = 30.0,
        poll_interval: float = 0.2,
        start_token_id: int = 0,
    ) -> torch.Tensor:
        """Retrieve the cached tensor for the current tokens, polling until
        ready.

        Args:
            kind: The type of cache to retrieve.
            timeout: Max seconds to wait for the cached tensor to appear.
            poll_interval: Seconds between retrieve attempts.
            start_token_id: The first token index to retrieve, aligned to
                chunk_size. A kind whose key chain starts later (the query
                tensors, see ``_key_origin``) is read from its chain's start.

        Returns:
            The cached tensor (chunk-aligned) from
            ``max(start_token_id, key origin)``.
            KV shape is [2, L, T, D]. Q shape is [1, L, T, D].

        Raises:
            LMCacheRequestStreamError: If no cached tensor is available within timeout.
        """
        ctx = self._contexts.get(kind)
        if not ctx:
            raise LMCacheRequestStreamError(
                f"no context available for cache kind {kind}"
            )

        # Resolve the origin: KV chains from token 0, query tensors from the
        # first token the latest generate() computed.
        origin = self._key_origin(kind)
        window = self.tokens[origin:]
        relative_start = max(start_token_id - origin, 0)

        deadline = time.perf_counter() + timeout
        result = ctx.retrieve(window, self.cache_salt, start_token_id=relative_start)
        while result is None and time.perf_counter() < deadline:
            time.sleep(poll_interval)
            result = ctx.retrieve(
                window, self.cache_salt, start_token_id=relative_start
            )
        if result is None:
            raise LMCacheRequestStreamError(
                f"no cached {kind} for {self.request_stream_id} "
                f"[{origin + relative_start}, {len(self.tokens)}) after "
                f"{timeout:.0f}s"
            )
        return result[0]

    def _retrieve_until(
        self,
        kind: LMCacheSDKCacheKind,
        origin: int,
        window_tokens: int,
        timeout: float,
        poll_interval: float,
    ) -> torch.Tensor:
        """Retrieve a kind's whole window, retrying until every row is stored.

        The query stores of the latest generate() may still be in flight, so a
        retrieve can return fewer rows than the window holds.

        Args:
            kind: The cache kind to retrieve.
            origin: First token of this kind's key chain (see _key_origin).
            window_tokens: Chunk-aligned tokens of the window from ``origin``.
            timeout: Max seconds to wait for the window to be complete.
            poll_interval: Seconds between attempts.

        Returns:
            The tensor holding ``window_tokens`` rows from ``origin``.

        Raises:
            LMCacheRequestStreamError: If the window is empty, or if it is
                still short at timeout.
        """
        ctx = self._contexts.get(kind)
        if not ctx:
            raise LMCacheRequestStreamError(
                f"no context available for cache kind {kind}"
            )
        if window_tokens <= 0:
            raise LMCacheRequestStreamError(
                f"empty {kind} window at token {origin} for "
                f"{self.request_stream_id}: the cached KV ends where the last "
                f"generate() started computing, so this kind has nothing to "
                f"read (a modify without an intervening generate?)"
            )
        window = self.tokens[origin : origin + window_tokens]
        deadline = time.perf_counter() + timeout
        while True:
            result = ctx.retrieve(window, self.cache_salt)
            tensor = result[0] if result is not None else None
            if tensor is not None and tensor.shape[-2] == window_tokens:
                return tensor
            if time.perf_counter() >= deadline:
                got = 0 if tensor is None else tensor.shape[-2]
                raise LMCacheRequestStreamError(
                    f"{kind} for {self.request_stream_id} returned {got} rows "
                    f"from token {origin} but its window expects "
                    f"{window_tokens} after {timeout:.0f}s "
                    f"(keys chained from token {origin})"
                )
            time.sleep(poll_interval)

    def update(
        self,
        kind: LMCacheSDKCacheKind,
        kv: torch.Tensor,
        tokens: Sequence[int],
    ) -> None:
        """Store an edited KV and reset the stream to back it.

        Replaces tokens with the given tokens and clears done. Logs a warning
        if the store reports the KV was already cached.

        Args:
            kv: The edited KV tensor to store, shape [2, L, T, D].
            tokens: Token ids the KV corresponds to (T must match kv.shape[2]).
        """
        ctx = self._contexts.get(kind)
        if not ctx:
            raise LMCacheRequestStreamError(
                f"no context available for cache kind {kind}"
            )
        if not ctx.store(kv, tokens, self.cache_salt):
            logger.warning(
                "store reported edited KV already cached for stream %s",
                self.request_stream_id,
            )
        self.tokens = list(tokens)
        self.done = False

        # The cache is stored in whole chunks, so the next generate() computes
        # from the end of the stored chunks. Record it so that modify_kv() can
        # address that pass's query tensors (keyed from that token).
        self._segment_start_token_id = (len(tokens) // ctx.chunk_size) * ctx.chunk_size

    def modify_kv(
        self,
        fn: ModifyFnType,
        timeout: float = 30.0,
        poll_interval: float = 0.2,
    ) -> None:
        """Edit the cached KV via a caller-supplied function.

        Retrieves the chunk-aligned KV, records the non-chunk-aligned tail in
        _suffix_tokens (prepended, so it survives until the next generate),
        applies fn to the cached prefix, and stores the result via update_kv.

        Args:
            fn: KV editor given a TensorDict (a dict of the tensor for
                each cache kind used in the modification algorithm, plus
                ``segment_start_token_id``, where the latest decoded segment
                begins) and the cached tokens, returning (new_kv, new_tokens)
                for the edited prefix. Query tensors cover only the tokens
                from ``segment_start_token_id`` to the end of the cached KV.

        Raises:
            LMCacheRequestStreamError: If the generated KV is not fully stored
                within timeout, or a kind's tensors are incomplete.
        """
        # Wait until the KV of the latest generate() is stored, so the
        # retrieved KV covers every chunk of the stream.
        kv_ctx = self._contexts[LMCacheSDKCacheKind.KV]
        expected = (len(self.tokens) // kv_ctx.chunk_size) * kv_ctx.chunk_size
        deadline = time.perf_counter() + timeout
        while kv_ctx.lookup(self.tokens, self.cache_salt) < expected:
            if time.perf_counter() >= deadline:
                raise LMCacheRequestStreamError(
                    f"KV for {self.request_stream_id} still incomplete after "
                    f"{timeout:.0f}s; expected {expected} tokens"
                )
            time.sleep(poll_interval)

        kv = self.retrieve(
            kind=LMCacheSDKCacheKind.KV,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        cached_len = kv.shape[2]
        if cached_len > len(self.tokens):
            raise LMCacheRequestStreamError(
                f"KV covers {cached_len} tokens but the stream has "
                f"{len(self.tokens)}; the stream was rewound without a "
                f"generate() in between"
            )

        tensors = TensorDict(
            {LMCacheSDKCacheKind.KV: kv},
            segment_start_token_id=self._segment_start_token_id,
        )
        for kind in self._contexts:
            if kind is LMCacheSDKCacheKind.KV:
                continue
            origin = self._key_origin(kind)
            tensors[kind] = self._retrieve_until(
                kind,
                origin,
                cached_len - origin,
                timeout,
                poll_interval,
            )

        # Tokens past the cached KV: the remainder of chunks that retrieve()
        # (chunk-aligned) didn't return.
        self._suffix_tokens = list(self.tokens[cached_len:]) + self._suffix_tokens
        new_kv, new_tokens = fn(tensors, self.tokens[:cached_len])
        self.update(kind=LMCacheSDKCacheKind.KV, kv=new_kv, tokens=new_tokens)
