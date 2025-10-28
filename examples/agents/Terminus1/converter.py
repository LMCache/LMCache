#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Converter to transform token_count_record.json to trace.json format.

Input format: {"messages": [{"role": "user", "content": "..."}]}
Output format: {"timestamp": 0, "input": "...", "output": "...", "session_id": 0}
"""

# Standard
from pathlib import Path
from typing import Any
import json


def extract_input_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract the input prompt from messages array."""
    if not messages:
        return ""

    # Find the first user message
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")

    return ""


def extract_output_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract the output response from messages array."""
    if not messages:
        return ""

    # Find the first assistant message
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content", "")

    return ""


def convert_record_to_trace(
    record: dict[str, Any], timestamp: int, session_id: int
) -> dict[str, Any]:
    """Convert a single record to trace format."""
    messages = record.get("messages", [])

    return {
        "timestamp": timestamp,
        "input": extract_input_from_messages(messages),
        "output": extract_output_from_messages(messages),
        "session_id": session_id,
    }


def main():
    input_file = Path("appworld-dev.json")
    output_file = Path("trace.json")

    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return

    print(f"Reading from: {input_file.absolute()}")

    traces = []
    session_id = 0
    timestamp = 0

    # Read JSONL file line by line
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                trace = convert_record_to_trace(record, timestamp, session_id)
                traces.append(trace)

                # Increment timestamp for each record
                timestamp += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue

    # Write trace.json as JSONL (one JSON object per line)
    with open(output_file, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    print(f"Converted {len(traces)} records")
    print(f"Output written to: {output_file.absolute()}")


if __name__ == "__main__":
    main()
