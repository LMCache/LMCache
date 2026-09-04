# SPDX-License-Identifier: Apache-2.0
"""Send one HMA faithfulness request and save the completion for comparison.

The prompt is a numbered list of rules, each embedding a unique codeword,
followed by a question answerable only by reading codewords back out of the
prefix -- so a faithful answer requires attention over the whole cached
prefix, and corrupted KV shows up as a wrong or missing codeword. Requests go
to /v1/completions at temperature 0, so a given prompt has one correct
continuation.

Each request writes its measured local/external split to
``<out>.split.json`` for run-hma-faithfulness.sh to assert on.

run-hma-faithfulness.sh owns the phase protocol: which prompts to send, when to
reset vLLM's local prefix cache, and which completions to compare.
"""

# Standard
import argparse
import json
import sys
import urllib.request

RULE_TEMPLATE = (
    "Rule %d: The secret codeword of rule %d is kiwi-%d. Always be precise, "
    "cite the rule number you applied, and keep the answer short. "
)
QUESTION = (
    "\n\nQuestion: List the secret codewords of rules 3, 17, and 41, "
    "one per line."
)
# Closes Qwen3.5's <think> block, as its chat template does for
# enable_thinking=false; /v1/completions applies no template.
THINK_OFF = "\n\n<think>\n\n</think>\n\n"
# vLLM's per-connector prefix-cache counters, in tokens.
QUERIES_METRIC = "vllm:external_prefix_cache_queries_total"
HITS_METRIC = "vllm:external_prefix_cache_hits_total"


def build_prompt(num_rules: int) -> str:
    """Build the deterministic test prompt.

    Args:
        num_rules: Number of codeword-bearing rules in the prefix. Must be
            large enough that the prefix spans several LMCache chunks and
            covers every rule number named in the question.

    Returns:
        The full prompt string (prefix plus question).
    """
    prefix = "You are a careful assistant.\n" + "\n".join(
        RULE_TEMPLATE % (i, i, i) for i in range(num_rules)
    )
    return prefix + QUESTION + THINK_OFF


def scrape_external_counters(port: int) -> tuple[float, float]:
    """Read vLLM's external-prefix-cache counters.

    Args:
        port: TCP port of the vLLM server.

    Returns:
        ``(queries, hits)`` in tokens, as reported by ``/metrics``. Both are
        ``0.0`` when the counters are absent, which is the case when no KV
        connector is attached.

    Raises:
        urllib.error.URLError: If the metrics endpoint is unreachable.
    """
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=60
    ) as response:
        body = response.read().decode()
    queries = hits = 0.0
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(QUERIES_METRIC):
            queries += float(line.rsplit(" ", 1)[1])
        elif line.startswith(HITS_METRIC):
            hits += float(line.rsplit(" ", 1)[1])
    return queries, hits


def send_request(
    port: int, model: str, prompt: str, max_tokens: int
) -> tuple[str, int]:
    """Send the completion request and return the generated text.

    Args:
        port: TCP port of the vLLM /v1/completions endpoint.
        model: Served model name to put in the request body.
        prompt: The exact prompt string to complete.
        max_tokens: Completion length cap; large enough for three codewords.

    Returns:
        ``(text, prompt_tokens)``: the completion, and the prompt length vLLM
        billed for it -- the denominator of the phase's local/external split.

    Raises:
        urllib.error.URLError: If the server is unreachable or rejects the
            request.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    usage = payload.get("usage", {})
    return payload["choices"][0]["text"], int(usage.get("prompt_tokens", 0))


def main() -> int:
    """Run one phase, writing the completion and its measured split.

    Writes the completion text to ``--out`` and the request's local/external
    token split to ``<out>.split.json``.

    Returns:
        Process exit code: 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, help="File to write the text to")
    parser.add_argument("--num-rules", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()

    prompt = build_prompt(args.num_rules)

    before_queries, before_hits = scrape_external_counters(args.port)
    text, prompt_tokens = send_request(
        args.port, args.model, prompt, args.max_tokens
    )
    after_queries, after_hits = scrape_external_counters(args.port)

    external = int(after_hits - before_hits)
    # Not supplied externally: an APC hit or a recompute -- these counters do
    # not separate the two.
    local = max(prompt_tokens - external, 0)
    split = {
        "prompt_tokens": prompt_tokens,
        "external_tokens": external,
        "local_tokens": local,
        "external_queried_tokens": int(after_queries - before_queries),
    }
    with open(args.out, "w") as f:
        f.write(text)
    with open(args.out + ".split.json", "w") as f:
        json.dump(split, f)
    share = external / prompt_tokens if prompt_tokens else 0.0
    print(
        f"prompt_tokens={prompt_tokens} local={local} external={external} "
        f"({share:.0%} external)"
    )
    print(f"completion: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
