#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Basic P2P correctness test between two vLLM+LMCache nodes.

Leg A->B: a fresh long prompt goes to node A (cold: A computes and stores the KV), then
the same prompt goes to node B. B never saw it, vLLM prefix caching is off, and B's only
remote source is its P2P peer, so ``num_lmcache_cached_tokens > 0`` on B proves the KV
came from A over P2P. Then 16 tokens are generated on both nodes and must match. Leg
B->A repeats with a new prompt. Finally the prompt is sent to B again (P2P-loaded keys
are not retained by default, so this is normally a second P2P hit, not an L1 hit).

Requests are non-streaming because vLLM returns
``kv_transfer_params.cached_token_stats`` only on non-streaming responses; TTFT is
the latency of a ``max_tokens=1`` request.

usage: basic_test.py [--a URL] [--b URL] [--model ID] [--words N] [--gap S]
       (defaults come from env.sh: --a = VLLM_URL, --b = PEER_VLLM_URL)
exit 0 on PASS, 1 on FAIL.
"""

# Standard
import argparse
import json
import random
import sys
import time
import urllib.request

# Third Party
import p2penv

WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


def make_prompt(n_words: int, seed: int) -> str:
    """Return a deterministic prompt of ``n_words`` Greek-letter names.

    Each name is about 1.17 tokens.
    """
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return (
        f"Run tag {seed}. Memorize the following token stream and then answer."
        f"\n\n{body}"
        "\n\nQuestion: what is the first word of the stream?"
    )


def chat(base: str, model: str, prompt: str, max_tokens: int = 1) -> dict:
    """Send one non-streaming chat request.

    Returns latency, generated text, cache stats and prompt token count.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "kv_transfer_params": {"cached_token_stats": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    lat = time.perf_counter() - t0
    return {
        "ttft_s": round(lat, 3),
        "text": d["choices"][0]["message"]["content"],
        "stats": (d.get("kv_transfer_params") or {}).get("cached_token_stats"),
        "prompt_tokens": d["usage"]["prompt_tokens"],
    }


def evaluate(
    cold: dict, peer: dict, text_src: str, text_dst: str, min_hit: float = 0.8
) -> tuple[bool, str]:
    """Judge one leg: identical outputs and a peer hit >= ``min_hit`` of the prompt.

    Returns (passed, one-line verdict). Pure function so it can be unit-tested.
    """
    cached = (peer["stats"] or {}).get("num_lmcache_cached_tokens", 0)
    same = text_src == text_dst
    ok = same and cached > 0 and cached >= min_hit * (peer["prompt_tokens"] or 1)
    speed = cold["ttft_s"] / max(peer["ttft_s"], 1e-6)
    return ok, (
        f"outputs_identical={same} p2p_cached_tokens={cached}/{peer['prompt_tokens']} "
        f"ttft_speedup={speed:.1f}x -> {'PASS' if ok else 'FAIL'}"
    )


def leg(
    name: str, src: str, dst: str, model: str, seed: int, n_words: int, gap: float
) -> bool:
    prompt = make_prompt(n_words, seed)
    r1 = chat(src, model, prompt)
    print(
        f"  [{name}] cold on {src}: ttft={r1['ttft_s']}s "
        f"prompt_tokens={r1['prompt_tokens']} stats={r1['stats']}"
    )
    time.sleep(gap)  # let the source finish storing the KV into its L1
    r2 = chat(dst, model, prompt)
    print(
        f"  [{name}] peer on {dst}: ttft={r2['ttft_s']}s "
        f"prompt_tokens={r2['prompt_tokens']} stats={r2['stats']}"
    )
    t1 = chat(src, model, prompt, 16)["text"]
    t2 = chat(dst, model, prompt, 16)["text"]
    ok, verdict = evaluate(r1, r2, t1, t2)
    print(
        f"  [{name}] {verdict}\n        src text: {t1[:70]!r}"
        f"\n        dst text: {t2[:70]!r}"
    )
    return ok


def main() -> int:
    env = p2penv.load()
    va, vb, _, _ = p2penv.url_defaults(env)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", default=va)
    ap.add_argument("--b", default=vb)
    ap.add_argument("--model", default=env.get("MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--words", type=int, default=6000)
    ap.add_argument("--gap", type=float, default=3.0)
    a = ap.parse_args()
    seed = time.time_ns() % 10**9
    results = [
        leg("A->B", a.a, a.b, a.model, seed, a.words, a.gap),
        leg("B->A", a.b, a.a, a.model, seed + 1, a.words, a.gap),
    ]
    r = chat(a.b, a.model, make_prompt(a.words, seed))
    print(f"  [B repeat] ttft={r['ttft_s']}s stats={r['stats']}")
    print("BASIC_TEST:", "PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
