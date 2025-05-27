# TTFT Benchmark

A tiny, self‑contained script for measuring **Time‑to‑First‑Token (TTFT)** and
follow‑up latency from any **vLLM** server that exposes an
*OpenAI‑compatible* endpoint.

> **Why run it?**  
> • Compare *cold* latency vs. *cache‑hit* latency.  
> • Test whether a VRAM prefix‑cache (or any other caching tier) really speeds
>   things up.  
> • Gather JSONL data you can plot later.

----------------------------------------------------------------------
1 · Prerequisites
----------------------------------------------------------------------

| Requirement        | Notes                                                                  |
|--------------------|------------------------------------------------------------------------|
| Running **vLLM**   | Must expose the OpenAI `/v1` API on the port you test (default 8000). |
| Python 3.9 +       | `pip install openai transformers`                                      |

*Nothing else is required.*  
The benchmark is agnostic to LMCache, in‑VRAM cache, SSD cache, etc. – it
simply times the server’s responses.

----------------------------------------------------------------------
2 · Command‑line flags
----------------------------------------------------------------------

| Flag / shorthand        | Default                         | Purpose                                                    |
|-------------------------|---------------------------------|------------------------------------------------------------|
| `--api_base`            | `http://localhost:8000/v1`      | URL of your vLLM server                                    |
| `--api_key`             | `EMPTY`                         | Any string – vLLM ignores it                               |
| `--model`               | first model from `/models`      | Explicit model ID                                          |
| `-C`, `--context_file`  | see table below                 | Pick the document to embed                                 |
| `--max_ctx_tokens`      | `131072`                        | Upper‑bound after token truncation                         |
| `--prompt`              | `"Summarize this text"`         | Prompt appended after the doc                              |
| `--num_following`       | `0`                             | Extra TTFT‑measured requests after run 1                   |
| `-F`, `--flush_cache`   | off                             | Flush GPU KV‑cache once after run 1                        |
| `--out`                 | `benchmark.jsonl`               | JSONL record of each run (file is cleared first)           |

Behaviour of `--context_file`:

| Invocation                        | Document source                  |
|-----------------------------------|----------------------------------|
| flag **omitted**                  | synthetic 100 000‑char ASCII doc |
| `--context_file` (no path)        | bundled `../ffmpeg.txt`          |
| `--context_file /path/to/file`    | text loaded from that path       |

*Legacy shorthand* – you may still run  
`python bench.py <PORT>` and everything else stays default.

----------------------------------------------------------------------
3 · Basic usage
----------------------------------------------------------------------

Run against a local server on port 8000 using the **ffmpeg** doc, *no* cache
flush, and two follow‑up requests:

    python bench.py --context_file --num_following 2

Example output:

    === Run 1: baseline TTFT ===
    TTFT_1 = 0.433s • …
    (no KV‑cache flush requested)

    === Run 2: TTFT continued ===
    TTFT_2 = 0.089s • …

`benchmark.jsonl`:

    {"run_index":1,"context_chars":120934,"ttft_seconds":0.433}
    {"run_index":2,"context_chars":120934,"ttft_seconds":0.089}

----------------------------------------------------------------------
4 · Advanced examples
----------------------------------------------------------------------

### 4.1 · Measure VRAM cache benefit *with* a flush

    python bench.py \
        -C war_and_peace.txt \
        --num_following 3 \
        --flush_cache \
        --prompt "Give me a concise outline." \
        --out warpeace_flush.jsonl

* Run 1 – cold path  
* Run 2 – KV‑cache flushed → tier‑2 storage latency  
* Runs 3‑4 – warm path (cache hits)

### 4.2 · Synthetic stress without touching disk

    python bench.py --num_following 1 -F


----------------------------------------------------------------------
5 · Tips & notes
----------------------------------------------------------------------

* **Spinner** – Turns red while waiting for the first token; stops once it
  arrives, then you watch the stream.
* **Tokenization fallback** – If a tokenizer cannot be loaded, the script uses
  a “4 chars ≈ 1 token” heuristic.

Happy benchmarking!

