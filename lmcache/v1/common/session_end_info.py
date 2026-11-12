# SPDX-License-Identifier: Apache-2.0
"""How a finished request ended, as the engine reports it with END_SESSION.

The client builds a :class:`SessionEndInfo` and the server's commit policy
reads it, so it sits here with the other data both endpoints share, where the
shared gRPC codec can import it without depending on the server.
"""

# Standard
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionEndInfo:
    """What the engine knows about a finished request, sent with END_SESSION.

    The commit policy decides from these facts whether the request's final
    window is worth committing to L2 (see
    ``lmcache/v1/multiprocess/commit_policy.py``). They are raw observations,
    not a decision: the engine reports how the request ended and the policy
    owns the rule.

    Every field has a default so an engine adapter that knows nothing sends
    ``SessionEndInfo()`` and the policy sees "no information", which every
    built-in policy treats as "do not commit".
    """

    finish_reason: str = ""
    """How the request finished, lowercased vLLM ``FinishReason`` ("stop",
    "length", "abort", "error", "repetition"). Empty means unknown."""

    stop_token_id: int = -1
    """Token id the generation stopped on, ``-1`` when unknown or when it
    stopped on a stop string, which is not a token. For a chat model this is
    the turn-boundary marker the model emits to end its turn (Qwen:
    ``<|im_end|>`` = 151645). The engine reports the last generated token when
    it has no more specific stop reason, which is the ordinary case for a model
    that ends its turn on its own EOS."""


NO_SESSION_END_INFO = SessionEndInfo()
"""The "engine reported nothing" value, for adapters that do not observe how a
request finished. Shared because :class:`SessionEndInfo` is frozen."""
