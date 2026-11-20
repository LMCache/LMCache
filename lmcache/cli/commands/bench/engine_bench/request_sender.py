# SPDX-License-Identifier: Apache-2.0
"""Async streaming request sender for ``lmcache bench engine``."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import collections.abc
import os
import time

try:
    # Third Party
    from openai import AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    AsyncOpenAI = None  # type: ignore[assignment]

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Callback signature: (result, response_text) -> None
OnFinishedCallback = Callable[[RequestResult, str], None]


@dataclass
class TokenCompletion:
    """A measured completion and the server's exact generated token IDs.

    ``token_ids`` is empty on failure. It must never be reconstructed from
    decoded text: decoding and encoding can change token boundaries.
    """

    result: RequestResult
    token_ids: list[int]


def _normalize_url(engine_url: str) -> str:
    """Ensure *engine_url* has a scheme and ends with ``/v1``."""
    url = engine_url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def _extract_content(chunk: object, completions_mode: bool) -> str:
    """Return non-empty text from a completion or chat streaming chunk.

    Chat chunks fall back to ``reasoning_content`` and then ``reasoning``
    when ``content`` is empty. Ported from Tensormesh-Benchmark
    ``streaming_utils.py``.

    Args:
        chunk: An OpenAI-compatible streaming response chunk.
        completions_mode: Whether the chunk uses the legacy completions API.

    Returns:
        The chunk's text, or an empty string when it contains no text.
    """
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""

    choice = choices[0]

    if completions_mode:
        text = getattr(choice, "text", None)
        return text if text is not None else ""

    # Chat mode: delta.content, with fallback for reasoning_content
    delta = getattr(choice, "delta", None)
    if delta is None:
        return ""
    content = getattr(delta, "content", None)
    if content:
        return content
    # Fallback for reasoning models
    for attr in ("reasoning_content", "reasoning"):
        fallback = getattr(delta, attr, None)
        if fallback:
            return fallback
    return ""


class RequestSender:
    """Async streaming request sender for inference engines.

    Each ``send_request`` call is a self-contained coroutine.
    Concurrency is controlled externally by the workload module.
    """

    def __init__(
        self,
        engine_url: str,
        model: str,
        completions_mode: bool = False,
        on_finished: list[OnFinishedCallback] = [],  # noqa: B006
        ignore_eos: bool = False,
        extra_body: dict[str, object] = {},  # noqa: B006
    ) -> None:
        """Initialize the sender and its OpenAI client.

        Args:
            engine_url: Base URL of the inference engine.
            model: Model name sent with every request.
            completions_mode: Send legacy ``/completions`` requests instead
                of ``/chat/completions``.
            on_finished: Callbacks invoked when a request completes.  Copied,
                so later mutation of the caller's list has no effect.
            ignore_eos: Add vLLM's ``ignore_eos`` to every request body, so
                generation runs for the full ``max_tokens``.
            extra_body: Extra fields merged into every request body, for
                options the OpenAI client has no parameter for (notably
                ``chat_template_kwargs``).  Copied, and sent verbatim, so the
                caller owns their compatibility with the backend.

        Raises:
            RuntimeError: If the optional OpenAI dependency is unavailable.
        """
        self._model = model
        self._completions_mode = completions_mode
        self._on_finished = list(on_finished)
        self._ignore_eos = ignore_eos
        self._extra_body = dict(extra_body)

        if AsyncOpenAI is None:
            raise RuntimeError(
                "lmcache bench engine requires the optional 'openai' package. "
                "Install it with `pip install openai`."
            )

        base_url = _normalize_url(engine_url)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            api_key = "sk-dummy"
            logger.debug("API key source: default dummy key")
        else:
            logger.debug("API key source: OPENAI_API_KEY env var")

        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=None,
        )

    def add_on_finished_callback(self, callback: OnFinishedCallback) -> None:
        """Register a callback to be invoked when a request finishes."""
        self._on_finished.append(callback)

    async def send_request(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 128,
    ) -> RequestResult:
        """Send a single streaming request and return the result.

        Streams the response via SSE, measures TTFT from the first non-empty
        content or reasoning text, decode speed, and total latency. Extracts
        token counts from server usage reports.
        After collecting the result, invokes all registered
        ``on_finished`` callbacks.

        Args:
            request_id: Identifier included in the result and callbacks.
            messages: Chat messages, or the first message's content as the
                prompt in completions mode.
            max_tokens: Maximum number of output tokens requested.

        Returns:
            Token counts, timings, and success status for this request.
        """
        submit_time = time.time()
        first_token_time = 0.0
        first_chunk_time = 0.0
        tokens: list[str] = []
        num_input_tokens = 0
        num_output_tokens = 0

        try:
            response = await self._create_stream(messages, max_tokens)

            async for chunk in response:
                if not first_chunk_time:
                    first_chunk_time = time.time()

                # Extract usage from final chunk
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    pt = getattr(usage, "prompt_tokens", 0)
                    ct = getattr(usage, "completion_tokens", 0)
                    if pt:
                        num_input_tokens = pt
                    if ct:
                        num_output_tokens = ct

                content = _extract_content(chunk, self._completions_mode)
                if content:
                    if not first_token_time:
                        first_token_time = time.time()
                    tokens.append(content)

            finish_time = time.time()
            if first_token_time == 0.0 and num_output_tokens > 0:
                # Empty-content stream (common with max_tokens=1, e.g. the
                # single token is EOS): use first chunk arrival as TTFT —
                # closer to engine prefill completion than finish_time.
                first_token_time = first_chunk_time or finish_time
            successful = first_token_time > 0.0
            ttft = (first_token_time - submit_time) if successful else -1.0
            request_latency = finish_time - submit_time
            decode_time = (finish_time - first_token_time) if successful else 0.0
            num_output = num_output_tokens if num_output_tokens > 0 else len(tokens)
            decode_speed = (num_output / decode_time) if decode_time > 0 else 0.0

            result = RequestResult(
                request_id=request_id,
                successful=successful,
                ttft=ttft,
                request_latency=request_latency,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output,
                decode_speed=decode_speed,
                submit_time=submit_time,
                first_token_time=first_token_time,
                finish_time=finish_time,
                error="",
            )
            response_text = "".join(tokens)

        except Exception as e:
            finish_time = time.time()
            result = RequestResult(
                request_id=request_id,
                successful=False,
                ttft=-1.0,
                request_latency=finish_time - submit_time,
                num_input_tokens=0,
                num_output_tokens=0,
                decode_speed=0.0,
                submit_time=submit_time,
                first_token_time=0.0,
                finish_time=finish_time,
                error=str(e),
            )
            response_text = ""
            logger.debug(
                "Request %s failed: %s",
                request_id,
                e,
            )

        for cb in self._on_finished:
            cb(result, response_text)

        return result

    async def send_warmup_request(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1,
    ) -> RequestResult:
        """Send a warmup request (``max_tokens=1`` by default)."""
        return await self.send_request(
            request_id,
            messages,
            max_tokens=max_tokens,
        )

    async def send_token_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_tokens: int = 128,
        seed: int = 42,
    ) -> TokenCompletion:
        """Send exact token IDs through vLLM's streaming completions API.

        Uses ``return_token_ids`` and checks the echoed prompt IDs and usage
        counts before allowing a conversation to continue. Existing text
        requests and their callbacks keep their original format. Protocol
        and transport failures produce a failed result and no continuation.

        Args:
            request_id: Identifier passed to the usual completion callbacks.
            prompt_token_ids: Nonempty sequence of nonnegative token IDs.
            max_tokens: Positive maximum number of generated tokens.
            seed: vLLM sampling seed for this request.

        Returns:
            Request measurements and generated IDs, including special tokens.

        Raises:
            ValueError: If the prompt or output limit is invalid.
        """
        if not prompt_token_ids or any(
            type(token) is not int or token < 0 for token in prompt_token_ids
        ):
            raise ValueError("prompt_token_ids must be nonempty nonnegative integers")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        prompt = list(prompt_token_ids)
        body = dict(self._extra_body)
        body.update(
            return_token_ids=True, add_special_tokens=False, echo=False, seed=seed
        )
        if self._ignore_eos:
            body["ignore_eos"] = True
        started = time.time()
        first_token = 0.0
        generated: list[int] = []
        text: list[str] = []
        checked_prompt = False
        finished_choice = False
        output_usage: int | None = None
        error = ""
        try:
            async with await self._client.with_options(
                max_retries=0
            ).completions.create(
                model=self._model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=body,
            ) as stream:
                async for chunk in stream:
                    if chunk.usage is not None:
                        if chunk.usage.prompt_tokens != len(prompt):
                            raise ValueError("Server changed the token prompt length")
                        output_usage = chunk.usage.completion_tokens
                    if not chunk.choices:
                        continue
                    if len(chunk.choices) != 1:
                        raise ValueError("Expected exactly one completion choice")
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        if choice.finish_reason not in ("stop", "length"):
                            raise ValueError(
                                "Server did not finish generation normally"
                            )
                        finished_choice = True
                    echoed = getattr(choice, "prompt_token_ids", None)
                    if echoed is not None:
                        if echoed != prompt:
                            raise ValueError("Server changed the token prompt")
                        checked_prompt = True
                    delta = getattr(choice, "token_ids", None)
                    if delta is None or any(
                        type(token) is not int or token < 0 for token in delta
                    ):
                        raise ValueError("Server must return delta token_ids")
                    if delta and not generated:
                        first_token = time.time()
                    generated.extend(delta)
                    text.append(choice.text)
            if not checked_prompt or not generated or not finished_choice:
                raise ValueError("Server did not return a complete token-ID response")
            if len(generated) > max_tokens or (
                output_usage is not None and output_usage != len(generated)
            ):
                raise ValueError("Generated token IDs disagree with output length")
        except Exception as exc:
            # This is an RPC boundary; failed streams cannot supply a prefix.
            logger.exception("Token request %s failed", request_id)
            error = f"{type(exc).__name__}: {exc}"
            generated = []
        finished = time.time()
        successful = not error
        result = RequestResult(
            request_id=request_id,
            successful=successful,
            ttft=first_token - started if successful else -1.0,
            request_latency=finished - started,
            num_input_tokens=len(prompt),
            num_output_tokens=len(generated),
            decode_speed=(
                len(generated) / (finished - first_token)
                if successful and finished > first_token
                else 0.0
            ),
            submit_time=started,
            first_token_time=first_token,
            finish_time=finished,
            error=error,
        )
        for callback in self._on_finished:
            callback(result, "".join(text) if successful else "")
        return TokenCompletion(result, generated)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _create_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> collections.abc.AsyncIterator:
        """Dispatch the streaming API call (chat or completions).

        When ``ignore_eos`` is set on the sender, ``{"ignore_eos": true}`` is
        added to the request body (a vLLM sampling extension) so generation
        always runs for the full ``max_tokens`` instead of stopping at the
        model's EOS token. This makes decode-throughput numbers reproducible.

        The sender's ``extra_body`` fields are merged in alongside it, for
        request options the OpenAI client has no parameter for (notably
        ``chat_template_kwargs``).
        """
        # Attach extra_body only when something needs it; otherwise send the
        # plain request so no vLLM-specific field reaches non-vLLM backends.
        body: dict[str, object] = dict(self._extra_body)
        if self._ignore_eos:
            body["ignore_eos"] = True
        extra: dict[str, dict[str, object]] = {"extra_body": body} if body else {}
        if self._completions_mode:
            prompt = messages[0]["content"] if messages else ""
            return await self._client.completions.create(
                model=self._model,
                prompt=prompt,
                stream=True,
                max_tokens=max_tokens,
                temperature=0.0,
                stream_options={"include_usage": True},
                **extra,
            )
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=0.0,
            stream_options={"include_usage": True},
            **extra,
        )
