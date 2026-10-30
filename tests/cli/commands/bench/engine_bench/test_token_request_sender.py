# SPDX-License-Identifier: Apache-2.0
"""Exercise token completions through the real OpenAI SDK and mock HTTP."""

# Standard
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch
import asyncio
import json

# Third Party
from openai import AsyncOpenAI
import httpx
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench import request_sender as sender_mod


def token_response(
    prompt: list[int], generated: list[int], *, usage_count: int | None = None
) -> httpx.Response:
    """Build vLLM 0.27.1-style delta-ID SSE, including a usage-only chunk."""
    chunks = [
        {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": None,
                    "prompt_token_ids": prompt,
                    "token_ids": generated[:1],
                }
            ],
        },
        {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "text": "answer",
                    "finish_reason": "length",
                    "token_ids": generated[1:],
                }
            ],
        },
        {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 0,
            "model": "m",
            "choices": [],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(generated)
                if usage_count is None
                else usage_count,
                "total_tokens": len(prompt) + len(generated),
            },
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return httpx.Response(200, text=body + "data: [DONE]\n\n")


class TestTokenRequestSender:
    def test_cancelled_stream_closes_transport_without_continuation(self) -> None:
        class InterruptedStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield b": keep-alive\n\n"
                raise asyncio.CancelledError()

            async def aclose(self) -> None:
                self.closed = True

        async def run() -> None:
            stream = InterruptedStream()
            client = AsyncOpenAI(
                api_key="test",
                base_url="http://test/v1",
                max_retries=2,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, stream=stream)
                    )
                ),
            )
            callback = MagicMock()
            with patch.object(sender_mod, "AsyncOpenAI", return_value=client):
                sender = sender_mod.RequestSender(
                    "http://test", "m", on_finished=[callback]
                )
            try:
                with pytest.raises(asyncio.CancelledError):
                    await sender.send_token_request("r", [1, 2], 2)
                assert stream.closed
                callback.assert_not_called()
            finally:
                await sender.close()

        asyncio.run(run())

    def test_exact_ids_protocol_and_callbacks(self) -> None:
        async def run() -> None:
            requests: list[dict[str, object]] = []

            def handler(request: httpx.Request) -> httpx.Response:
                assert request.url.path == "/v1/completions"
                body = json.loads(request.content)
                requests.append(body)
                return token_response(body["prompt"], [91, 92])

            client = AsyncOpenAI(
                api_key="test",
                base_url="http://test/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
            callback = MagicMock()
            with patch.object(sender_mod, "AsyncOpenAI", return_value=client):
                sender = sender_mod.RequestSender(
                    "http://test",
                    "m",
                    on_finished=[callback],
                    ignore_eos=True,
                    extra_body={"echo": True, "add_special_tokens": True},
                )
            try:
                prompt = [0, 10, 11]
                completion = await sender.send_token_request("r", prompt, 2, seed=7)
                assert prompt == [0, 10, 11]
                assert completion.token_ids == [91, 92]
                assert completion.result.successful
                assert completion.result.num_input_tokens == 3
                assert completion.result.num_output_tokens == 2
                callback.assert_called_once_with(completion.result, "answer")
                assert requests[0]["prompt"] == prompt
                assert requests[0]["return_token_ids"] is True
                assert requests[0]["add_special_tokens"] is False
                assert requests[0]["echo"] is False
                assert requests[0]["ignore_eos"] is True
                assert requests[0]["seed"] == 7
            finally:
                await sender.close()
            assert client.is_closed()

        asyncio.run(run())

    @pytest.mark.parametrize(
        "failure",
        ["missing_ids", "changed_prompt", "usage", "limit", "http", "no_finish"],
    )
    def test_protocol_failures_never_supply_continuation(self, failure: str) -> None:
        async def run() -> None:
            attempts = 0

            def handler(request: httpx.Request) -> httpx.Response:
                nonlocal attempts
                attempts += 1
                prompt = json.loads(request.content)["prompt"]
                if failure == "http":
                    return httpx.Response(500, json={"error": {"message": "failed"}})
                response = token_response(
                    prompt + [9] if failure == "changed_prompt" else prompt,
                    [3, 4, 5] if failure == "limit" else [3, 4],
                    usage_count=1 if failure == "usage" else None,
                )
                if failure == "missing_ids":
                    return httpx.Response(
                        200, text=response.text.replace('"token_ids"', '"ignored"')
                    )
                if failure == "no_finish":
                    return httpx.Response(
                        200,
                        text=response.text.replace(
                            '"finish_reason": "length"', '"finish_reason": null'
                        ),
                    )
                return response

            client = AsyncOpenAI(
                api_key="test",
                base_url="http://test/v1",
                max_retries=2,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
            callback = MagicMock()
            with patch.object(sender_mod, "AsyncOpenAI", return_value=client):
                sender = sender_mod.RequestSender(
                    "http://test", "m", on_finished=[callback]
                )
            try:
                completion = await sender.send_token_request("r", [1, 2], 2)
                assert not completion.result.successful
                assert completion.result.error
                assert completion.token_ids == []
                assert attempts == 1
                callback.assert_called_once_with(completion.result, "")
            finally:
                await sender.close()

        asyncio.run(run())

    @pytest.mark.parametrize(
        "prompt,limit", [([], 2), ([True], 2), ([-1], 2), ([1], 0)]
    )
    def test_invalid_inputs_fail_before_http(
        self, prompt: list[int], limit: int
    ) -> None:
        async def run() -> None:
            client = MagicMock()
            with patch.object(sender_mod, "AsyncOpenAI", return_value=client):
                sender = sender_mod.RequestSender("http://test", "m")
            with pytest.raises(ValueError):
                await sender.send_token_request("r", prompt, limit)
            client.completions.create.assert_not_called()

        asyncio.run(run())
