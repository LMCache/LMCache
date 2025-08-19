#!/usr/bin/env python3
import argparse
import json
import ast
import os
from typing import List, Iterable, Dict, Any, Tuple

import pandas as pd
from rapidfuzz.distance import Levenshtein

# If you prefer a different client, swap this out.
# Requires: pip install openai>=1.0.0
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None


def parse_args():
    ap = argparse.ArgumentParser(description="Send first 5 rows twice, score EditSim vs references.")
    ap.add_argument("--input-file", required=True, help="Path to CSV dataset with columns like dataset,index_in_dataset,language,context,input,answers")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="OpenAI model name")
    ap.add_argument("--rounds", type=int, default=2, help="How many times to send each row (default: 2)")
    ap.add_argument("--max-rows", type=int, default=5, help="How many rows to take from the top (default: 5)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-csv", default="", help="Optional path to save per-round results as CSV")
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                help="OpenAI-compatible base URL (e.g., http://localhost:8000/v1)")
    ap.add_argument("--api-key", default="EMPTY",
                help="API key if your server requires one (ignored by many local servers)")

    return ap.parse_args()


def _parse_answers(field: Any) -> List[str]:
    """Robustly parse the 'answers' column which is typically a JSON/py-list string."""
    if field is None:
        return []
    if isinstance(field, list):
        return [str(x) for x in field]
    s = str(field).strip()
    if not s:
        return []
    # Try JSON first, then Python literal (to handle single quotes)
    try:
        val = json.loads(s)
    except Exception:
        try:
            val = ast.literal_eval(s)
        except Exception:
            # Fallback: treat the whole cell as a single answer
            return [s]
    if isinstance(val, list):
        return [str(x) for x in val]
    return [str(val)]


def edit_sim_rf(gen: str, ref: str) -> float:
    return Levenshtein.normalized_similarity(gen or "", ref or "", weights=(1,1,2))


def row_score(gen: str, refs: Iterable[str]) -> float:
    refs = list(refs)
    if not refs:
        return 0.0
    return max(edit_sim_rf(gen, r) for r in refs)


def build_prompt(row: pd.Series) -> str:
    ds = str(row.get("dataset", "")).strip()
    if ds == "repobench-p_e":
        prompt = (
            f"TASK: Next-line code prediction (LANG={row.language}).\n"
            "Continue the CURRENT function/method/block in this file.\n"
            "Do NOT start a new function or class.\n\n"
            f"--- Repository context ---\n{row.context}\n\n"
            f"--- File snippet (cursor at end) ---\n{row.input}\n\n"
            "OUTPUT RULES:\n"
            "- Output EXACTLY one line of raw code (no trailing newline).\n"
            "- Preserve leading whitespace/indentation exactly.\n"
            "- Do NOT start a new class/function/interface.\n"
            "- Do NOT output code fences (like ```java, ```python), quotes, comments, or explanations."
        )
    elif ds == "lcc_e":
        prompt = (
            f"TASK: Next-line code prediction (LANG={row.language}).\n"
            "Continue the CURRENT function/method/block in the snippet below.\n"
            "Do NOT to start a new function or class.\n\n"
            f"--- Code snippet (cursor at end) ---\n{row.context}\n\n"
            "OUTPUT RULES:\n"
            "- Output EXACTLY one line of raw code (no trailing newline).\n"
            "- Preserve leading whitespace/indentation exactly.\n"
            "- Do NOT start a new class/function/interface.\n"
            "- Do NOT output code fences (like ```java, ```python), quotes, or explanations."
        )
    else:
        # Default fallback: treat like lcc_e wording
        prompt = (
            f"This is a next-line prediction task for user {row.get('index_in_dataset')} in the {ds} dataset "
            f"(language: {row.get('language')}).\n\n"
            "You are an expert code assistant. Given the provided contexts, "
            "predict the exact next line of code.\n\n"
            f"{row.get('context', '')}\n\n{row.get('input', '')}\n\n"
            "Do not include any extra text, comments, or explanations."
        )
    return prompt


def call_model(client, model: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    """Return the model's text (best-effort clean)."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stop=["\n"]
    )
    text = resp.choices[0].message.content or ""
    # Light cleanup: strip surrounding whitespace/newlines
    return text.strip()


def main():
    args = parse_args()

    if OpenAI is None:
        raise RuntimeError("openai package not available. Install with: pip install openai")

    client = OpenAI(
        base_url=args.base_url,
        api_key=os.getenv("OPENAI_API_KEY", args.api_key)  # many local servers ignore this
    )

    df = pd.read_csv(args.input_file)
    if df.empty:
        print("No rows found in input.")
        return

    # Take top-N rows
    work = df.head(args.max_rows).copy()

    results: List[Dict[str, Any]] = []

    for ridx, row in work.iterrows():
        prompt = build_prompt(row)
        messages = [{"role": "user", "content": prompt}]

        refs = _parse_answers(row.get("answers"))
        for round_id in range(1, args.rounds + 1):
            try:
                gen = call_model(client, args.model, messages, args.temperature)
            except Exception as e:
                gen = ""
                err = f"{type(e).__name__}: {e}"
            else:
                err = ""

            score = row_score(gen, refs)
            out_row: Dict[str, Any] = dict(
                row_id=ridx,
                dataset=row.get("dataset"),
                index_in_dataset=row.get("index_in_dataset"),
                language=row.get("language"),
                round=round_id,
                score=score,
                gen=gen,
                n_refs=len(refs),
                error=err,
            )
            results.append(out_row)
            # Print per-request score line
            print(
                f"[row {ridx} | {row.get('dataset')} #{row.get('index_in_dataset')} | round {round_id}] "
                f"score={score:.4f}"
            )
            # print(gen)  # <-- print the returned answer

    out_df = pd.DataFrame(results)
    if not out_df.empty:
        avg_score = out_df["score"].mean()
        print(f"\nAverage score across {len(out_df)} generations: {avg_score:.4f}")
    
    if args.out_csv:
        out_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved per-round results to: {args.out_csv}")


if __name__ == "__main__":
    main()
