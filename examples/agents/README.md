# Prefix Hit‑Rate Analysis (Agents)

This example estimates KV cache hit rates under different cache pool sizes by replaying a trace of prompts.

## What this demonstrates
- Prefix‑based reuse vs. substring reuse
- How cache size (GB) translates to hit rate

## Prerequisites
- Python 3.9+
- Install dependencies:
  - pip install -r requirements.txt

## Input format
The input file must be JSONL with an `input` field per line:
{"input": "your prompt here"}
{"input": "another prompt"}

## Run
python prefix_analysis.py -i trace.jsonl

## Output
- A PNG plot file (default: prefix_cache_hit_rate.png)
- Console summary of hit rates for each pool size

## Notes on cache sizing
`--tokens-per-gb` converts cache size to a token budget. Adjust it for your tokenizer/model if needed.


