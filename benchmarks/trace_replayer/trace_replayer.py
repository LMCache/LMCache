# SPDX-License-Identifier: Apache-2.0

# Standard library
# Standard
from dataclasses import dataclass
from typing import List, Optional
import argparse
import asyncio
import csv
import datetime
import json
import logging
import time

# Third Party
# Third-party
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

OUTPUT_FILENAME = f"summary-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"


@dataclass
class TraceRequest:
    """Represents a single request from the trace file."""

    req_id: str
    timestamp: float
    input_length: int
    target_output_length: int


@dataclass
class RequestResult:
    """Stores the result of a single request execution."""

    req_id: str
    ttft: float
    input_token_len: int
    output_token_len: int
    launch_time: float
    finish_time: float


class TraceReplayer:
    """Replays a trace of LLM requests against a specified model."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_duration: float = 60.0,
        max_input_tokens: Optional[int] = None,
    ):
        """
        Args:
            model: Model name or path to serve requests.
            base_url: Base URL of the LLM server.
            api_key: API key if required.
            max_duration: Maximum duration to replay trace requests.
            max_input_tokens: Optional max input length to truncate requests.
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_duration = max_duration
        self.max_input_tokens = max_input_tokens
        self.csv_file = None
        self.csv_writer = None

    def load_trace(self, trace_file: str = "trace.jsonl") -> List[TraceRequest]:
        """
        Load trace data from a JSONL file and expand hash_ids into individual
        TraceRequests.

        Truncates input_length if max_input_tokens is provided.

        Returns:
            List of TraceRequest objects.
        """
        requests = []
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            timestamp = float(data["timestamp"])
                            input_length = int(data.get("input_length", 0))
                            output_length = int(data.get("output_length", 0))
                            hash_ids = data.get("hash_ids", [])

                            # Expand each hash_id into a TraceRequest
                            for hid in hash_ids:
                                requests.append(
                                    TraceRequest(
                                        req_id=str(hid),
                                        timestamp=timestamp,
                                        input_length=(
                                            min(input_length, self.max_input_tokens)
                                            if self.max_input_tokens
                                            else input_length
                                        ),
                                        target_output_length=output_length,
                                    )
                                )
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Skipping malformed line {line_num}: {e}")
                            continue

            # Sort by timestamp and filter by max_duration
            requests.sort(key=lambda x: x.timestamp)
            if requests:
                min_timestamp = requests[0].timestamp
                for req in requests:
                    req.timestamp -= min_timestamp
                requests = [r for r in requests if r.timestamp <= self.max_duration]

            logger.info(
                f"Loaded {len(requests)} requests from {trace_file} "
                f"(duration: {self.max_duration}s)"
            )

            return requests
        except FileNotFoundError:
            logger.error(f"Trace file {trace_file} not found")
            return []

    def init_csv(self):
        """Initialize the CSV file for storing request results."""
        self.csv_file = open(OUTPUT_FILENAME, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "req_id",
                "ttft",
                "input_token_len",
                "output_token_len",
                "launch_time",
                "finish_time",
            ]
        )
        self.csv_file.flush()

    def write_result_to_csv(self, result: RequestResult):
        """Write a single request result to the CSV file."""
        if self.csv_writer:
            self.csv_writer.writerow(
                [
                    result.req_id,
                    f"{result.ttft:.4f}",
                    result.input_token_len,
                    result.output_token_len,
                    f"{result.launch_time:.4f}",
                    f"{result.finish_time:.4f}",
                ]
            )
            self.csv_file.flush()

    def close_csv(self):
        """Close the CSV file."""
        if self.csv_file:
            self.csv_file.close()

    async def send_request(self, request: TraceRequest) -> RequestResult:
        """
        Send a single request to the LLM asynchronously and record the result.

        Returns:
            RequestResult object containing execution metrics.
        """
        launch_time = time.time()
        first_token_time = None
        response_content = ""

        try:
            # Generate a fake prompt for this chunk
            prompt_text = "x" * request.input_length  # placeholder text

            messages = [{"role": "user", "content": prompt_text}]
            response = await self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0,
                stream=True,
                max_tokens=request.target_output_length,
                stream_options={"include_usage": True},
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if first_token_time is None and content.strip():
                        first_token_time = time.time()
                    response_content += content

            finish_time = time.time()
            input_tokens = (
                chunk.usage.prompt_tokens
                if hasattr(chunk, "usage") and chunk.usage
                else 0
            )
            output_tokens = (
                chunk.usage.completion_tokens
                if hasattr(chunk, "usage") and chunk.usage
                else 0
            )

            ttft = (first_token_time - launch_time) if first_token_time else 0.0

            result = RequestResult(
                req_id=request.req_id,
                ttft=ttft,
                input_token_len=input_tokens,
                output_token_len=output_tokens,
                launch_time=launch_time,
                finish_time=finish_time,
            )

            self.write_result_to_csv(result)
            return result

        except Exception as e:
            logger.error(f"Request {request.req_id} failed: {e}")
            finish_time = time.time()
            result = RequestResult(
                req_id=request.req_id,
                ttft=0.0,
                input_token_len=0,
                output_token_len=0,
                launch_time=launch_time,
                finish_time=finish_time,
            )
            self.write_result_to_csv(result)
            return result

    async def replay_trace(self, requests: List[TraceRequest]):
        """
        Replay the list of TraceRequests, preserving their relative timestamps.

        Requests are sent asynchronously in order of timestamp.
        """
        if not requests:
            logger.warning("No requests to replay")
            return

        self.init_csv()
        start_time = time.time()
        logger.info(
            f"Starting trace replay with {len(requests)} requests "
            f"over {self.max_duration}s"
        )

        tasks = []
        for request in requests:
            absolute_send_time = start_time + request.timestamp
            current_time = time.time()
            if absolute_send_time > current_time:
                await asyncio.sleep(absolute_send_time - current_time)
            task = asyncio.create_task(self.send_request(request))
            tasks.append(task)
            logger.info(
                f"Launched request {request.req_id} at {request.timestamp:.2f}s "
                f"(target output: {request.target_output_length})"
            )

        await asyncio.gather(*tasks)
        self.close_csv()

    def print_summary(self):
        """Print a summary of all requests from the CSV file."""
        try:
            # Third Party
            import pandas as pd

            df = pd.read_csv(OUTPUT_FILENAME)
            if len(df) == 0:
                logger.warning("No completed requests to summarize")
                return

            total_requests = len(df)
            avg_ttft = df["ttft"].mean()
            total_input_tokens = df["input_token_len"].sum()
            total_output_tokens = df["output_token_len"].sum()
            total_duration = df["finish_time"].max() - df["launch_time"].min()

            print("\n" + "=" * 60)
            print("TRACE REPLAY SUMMARY")
            print("=" * 60)
            print(f"Total Requests: {total_requests}")
            print(f"Total Duration: {total_duration:.2f}s")
            print(f"Average TTFT: {avg_ttft:.4f}s")
            print(f"Total Input Tokens: {total_input_tokens}")
            print(f"Total Output Tokens: {total_output_tokens}")
            print(f"Throughput: {total_requests / total_duration:.2f} req/s")
            print("=" * 60)

        except ImportError:
            logger.warning("pandas not available, skipping summary")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Trace Replayer for LLM benchmarking")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=None,
        help="Optional max input length (tokens) to truncate requests if needed",
    )
    parser.add_argument(
        "--trace_file",
        type=str,
        default="conversation_trace.jsonl",
        help="Trace JSONL file",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=60.0,
        help="Max duration to replay trace (seconds)",
    )
    args = parser.parse_args()

    replayer = TraceReplayer(
        model=args.model,
        max_duration=args.max_duration,
        max_input_tokens=args.max_input_length,
    )

    requests = replayer.load_trace(args.trace_file)

    if not requests:
        logger.error("No requests loaded")
        return

    # Replay the trace
    await replayer.replay_trace(requests)

    replayer.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
