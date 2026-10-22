# SPDX-License-Identifier: Apache-2.0
"""Chat endpoint that replays the server's own output as assistant history.

Pairs with :mod:`aiperf_prt.loader`, which tags assistant messages that stand
for a turn's full output with ``prt_src``.

* ``format_payload`` replaces a tagged message whose source output is stored
  with ``{"role": "assistant", "reasoning": "", "content": "<think>\\n" +
  decode(output_ids)}``, cut to the trace message's token length.
* ``process_responses`` stores a complete PROFILING response's ``token_ids``
  under ``(root_correlation_id, conversation_id, turn_index)``, and every
  request's prompt ids as ``$PRT_LIVE_HISTORY_LOG_DIR/prompts/<x_request_id>.i32``
  (native-endian int32), from which PRT scores the ideal cache hit share.

Why this rendering: the server generated after
``<|im_start|>assistant\\n<think>\\n``, and the Step template renders an
earlier assistant message as ``<|im_start|>assistant\\n`` + content.
``reasoning=""`` stops it splitting the content at ``</think>``, so the cached
sequence is reproduced up to BPE re-tokenization. Outputs are decoded from
token ids because the reasoning parser strips ``</think>`` and detokenization
skips special tokens.

Outputs are kept in memory and in ``$PRT_LIVE_HISTORY_LOG_DIR/store/``, shared
by all worker processes; the tokenizer is
``$PRT_LIVE_HISTORY_TOKENIZER/tokenizer.json``.
"""

# Future
from __future__ import annotations

# Standard
from array import array
from collections import OrderedDict
from typing import Any
import hashlib
import os

# Third Party
from aiperf.common.enums import CreditPhase
from aiperf.common.models import ParsedResponse, RequestInfo, RequestRecord, Turn
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf_prt.loader import SRC_KEY
from tokenizers import Tokenizer

_MAX_OUTPUTS = 50_000
_THINK_PREFIX = "<think>\n"

OutputKey = tuple[str, str, int]


class LiveHistoryChatEndpoint(ChatEndpoint):
    """``chat`` endpoint whose assistant history is the server's real output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize as the base endpoint, with the shared store and tokenizer.

        Raw ``tokenizers``, as vLLM uses: transformers 5's AutoTokenizer drops
        whitespace for this checkpoint.

        Args:
            *args: Passed to the base endpoint.
            **kwargs: Passed to the base endpoint.

        Raises:
            KeyError: ``PRT_LIVE_HISTORY_LOG_DIR`` or
                ``PRT_LIVE_HISTORY_TOKENIZER`` is unset.
        """
        super().__init__(*args, **kwargs)
        self._prt_store_dir = os.path.join(
            os.environ["PRT_LIVE_HISTORY_LOG_DIR"], "store"
        )
        os.makedirs(self._prt_store_dir, exist_ok=True)
        self._prt_prompt_dir = os.path.join(
            os.environ["PRT_LIVE_HISTORY_LOG_DIR"], "prompts"
        )
        os.makedirs(self._prt_prompt_dir, exist_ok=True)
        self._prt_tok = Tokenizer.from_file(
            os.path.join(os.environ["PRT_LIVE_HISTORY_TOKENIZER"], "tokenizer.json")
        )
        # Value: [ids, render cache (trace_n, content) or None].
        self._prt_outputs: OrderedDict[OutputKey, list[Any]] = OrderedDict()

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format as the base endpoint, with tagged assistant messages substituted.

        Args:
            request_info: The request.

        Returns:
            The JSON payload.
        """
        payload = super().format_payload(request_info)
        root = request_info.root_correlation_id or request_info.x_correlation_id
        messages = []
        for msg in payload["messages"]:
            src = msg.get(SRC_KEY)
            if src is None:
                messages.append(msg)
                continue
            conv, turn, trace_n = src.rsplit("#", 2)
            hit = self._prt_lookup((root, conv, int(turn)))
            text = None if hit is None else self._prt_render(hit, int(trace_n))
            if text is None:
                # Copy: ``msg`` aliases the session's raw_messages.
                messages.append({k: v for k, v in msg.items() if k != SRC_KEY})
            else:
                messages.append({"role": "assistant", "reasoning": "", "content": text})
        payload["messages"] = messages
        payload["return_token_ids"] = True
        return payload

    def process_responses(
        self,
        record: RequestRecord,
        *,
        capture_assistant_turn: bool,
    ) -> tuple[list[ParsedResponse], Turn | None]:
        """Parse as the base endpoint, then log the prompt and store a complete
        turn's output ids.

        The prompt is logged in every phase, errors included, whenever the
        server returned its ids: whatever it prefilled can be cached.

        Args:
            record: The completed request record.
            capture_assistant_turn: Passed through.

        Returns:
            The base endpoint's result.
        """
        result = super().process_responses(
            record, capture_assistant_turn=capture_assistant_turn
        )
        info = record.request_info
        if info is not None and record.responses:
            # responses[0] is the role chunk: prompt_token_ids, no output ids.
            prompt = (record.responses[0].get_json() or {}).get("prompt_token_ids")
            if prompt:
                path = os.path.join(self._prt_prompt_dir, f"{info.x_request_id}.i32")
                with open(path, "wb") as f:
                    f.write(array("i", prompt).tobytes())
        if (
            info is None
            or record.has_error
            or info.credit_phase != CreditPhase.PROFILING
            or info.max_tokens is None
        ):
            return result
        ids: list[int] = []
        for response in record.responses[1:]:
            obj = response.get_json()
            for choice in (obj or {}).get("choices") or []:
                ids.extend(choice.get("token_ids") or [])
        if ids and len(ids) == info.max_tokens:
            root = info.root_correlation_id or info.x_correlation_id
            self._prt_store(
                (root, info.conversation_id, info.turn_index), array("i", ids)
            )
        return result

    def _prt_render(self, out: list[Any], trace_n: int) -> str | None:
        """Return the content for output ``out``, or None when nothing fits.

        Cut to ``trace_n`` tokens (``<think>\\n`` counts) so ISL never exceeds
        the trace's; a prefix of the ids is still a cached prefix.
        """
        ids, cached = out
        if cached is not None and cached[0] == trace_n:
            return cached[1]
        keep = min(len(ids), trace_n - 2)
        if keep <= 0:
            return None
        text = self._prt_tok.decode(list(ids[:keep]), skip_special_tokens=False)
        out[1] = (trace_n, _THINK_PREFIX + text)
        return out[1][1]

    def _prt_store(self, key: OutputKey, ids: array) -> None:
        """Keep an output locally and publish it to the shared store.

        A session's turns can land on different workers. The write finishes
        before the credit returns, so later requests see it.
        """
        self._prt_outputs[key] = [ids, None]
        while len(self._prt_outputs) > _MAX_OUTPUTS:
            self._prt_outputs.popitem(last=False)
        path = self._prt_store_path(key)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(ids.tobytes())
        os.replace(tmp, path)

    def _prt_lookup(self, key: OutputKey) -> list[Any] | None:
        """Return ``[ids, render cache]`` of an output, from memory or the store."""
        out = self._prt_outputs.get(key)
        if out is not None:
            return out
        # Misses are not cached: another worker may store the output later.
        try:
            with open(self._prt_store_path(key), "rb") as f:
                ids = array("i")
                ids.frombytes(f.read())
        except FileNotFoundError:
            return None
        self._prt_outputs[key] = out = [ids, None]
        return out

    def _prt_store_path(self, key: OutputKey) -> str:
        root, conv, turn = key
        h = hashlib.sha1(f"{root}\0{conv}".encode()).hexdigest()[:24]
        return os.path.join(self._prt_store_dir, f"{h}_{turn}.i32")
