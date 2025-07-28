# Standard
from dataclasses import dataclass, asdict
import argparse
import atexit
import contextlib
import logging
import random
import signal
import sys
import time
from typing import List

# Third Party
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
import pandas as pd

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME, lmcache_get_config
from lmcache.v1.cache_engine import LMCacheEngineBuilder

from utils import (
    PromptBuildMethodType,
    build_rag_prompt_tokens,
    build_qa_prompt,
    build_fewshot_prompt,
    compute_f1,
    compute_rl,
    init_logger,
    load_dataset,
)

logger = init_logger(__name__, logging.INFO)

system_prompt_set = {
    PromptBuildMethodType.QA: "You will be asked a question after reading several passages. "  # noqa: E501
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. "
    "The answer should be within 5 words.\nPassages:\n",
    PromptBuildMethodType.FEW_SHOT: "Summarize the dialogue into a few short sentences. "  # noqa: E501
    "The following are some examples.\n\n",
}
query_prompt_set = {
    PromptBuildMethodType.QA: "\n\nAnswer the question directly based on the given passages."  # noqa: E501
    " Do NOT repeat the question. "
    "The answer should be within 5 words. \nQuestion:",
    PromptBuildMethodType.FEW_SHOT: "",
}


@contextlib.contextmanager
def build_llm_with_lmcache(model: str, max_model_len: int = 8000):
    """Build LLM with LMCache for offline serving"""
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
        # enforce_eager=True,  # NOTE: for debug
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


@dataclass
class WorkloadConfig:
    # Model name
    model: str
    # Tokenizer name
    tokenizer: str
    # Dataset.
    dataset: str
    # Start index of the workload
    start_index: int
    # End index of the workload
    end_index: int
    # Random shuffle.
    shuffle: bool
    # System prompt.
    system_prompt: str
    # Separator.
    separator: str
    # Query prompt.
    query_prompt: str
    # Prompt build method.
    prompt_build_method: PromptBuildMethodType
    # Max tokens for each generation.
    max_tokens: int
    # KV chunk size
    kv_chunk_size: int


@dataclass
class Response:
    request_id: int
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse RAG benchmark configurations.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--tokenizer", type=str, default="", help="Tokenizer name")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset path")
    parser.add_argument(
        "--start-index", type=int, default=0, help="Start index of the workload"
    )
    parser.add_argument(
        "--end-index", type=int, default=-1, help="End index of the workload"
    )
    parser.add_argument("--shuffle", action="store_true", help="Random shuffle")
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt")
    parser.add_argument("--query-prompt", type=str, default="", help="Query prompt")
    parser.add_argument(
        "--prompt-build-method",
        type=str,
        required=True,
        help="Prompt build method",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name for the summary csv",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to enable verbose logging",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens for each generation",
    )
    args = parser.parse_args()
    return args


def parse_size(size: str) -> int:
    """Parse size string like '30GB' to bytes"""
    if len(size) == 0:
        return -1
    else:
        size = size.upper()
        if size.endswith("KB"):
            return int(size[:-2]) * 1024
        elif size.endswith("MB"):
            return int(size[:-2]) * 1024 * 1024
        elif size.endswith("GB"):
            return int(size[:-2]) * 1024 * 1024 * 1024
        elif size.endswith("TB"):
            return int(size[:-2]) * 1024 * 1024 * 1024 * 1024
        elif size.endswith("B"):
            return int(size[:-1])
        else:
            raise ValueError(f"Invalid size unit {size}")


class KVSizeCalculator:
    def __init__(
        self,
        num_key_value_heads: int,
        head_dim: int,
        num_layers: int,
        precision: int,
    ):
        self.ratio = num_key_value_heads * head_dim * num_layers * precision * 2

    def get_kv_size(self, token_cnt: int) -> int:
        return token_cnt * self.ratio


class OfflineRAGManager:
    def __init__(self, workload_config: WorkloadConfig):
        self.workload_config = workload_config
        eval_dataset = load_dataset(workload_config.dataset)
        start_index = workload_config.start_index
        end_index = workload_config.end_index
        if end_index < 0:
            end_index = len(eval_dataset)
        eval_dataset = eval_dataset[start_index:end_index]
        if workload_config.shuffle:
            random.shuffle(eval_dataset)

        self._tokenizer = AutoTokenizer.from_pretrained(workload_config.tokenizer)
        self._model_config = AutoConfig.from_pretrained(workload_config.model)
        self._document_tokens = []  # Store document tokens for precompute
        self._request_tokens = []  # Store full request tokens
        self._answers = []
        self._build_method = workload_config.prompt_build_method
        self._results = []
        # Preprocess all prompts into token format
        system_prompt_tokens = self._tokenizer.encode(workload_config.system_prompt)
        # Remove BOS
        separator_tokens = self._tokenizer.encode(workload_config.separator)[1:]

        for ex in eval_dataset:
            if workload_config.prompt_build_method == PromptBuildMethodType.QA:
                doc_prompts, q_prompt = build_qa_prompt(
                    ex, workload_config.query_prompt
                )
            elif workload_config.prompt_build_method == PromptBuildMethodType.FEW_SHOT:
                doc_prompts, q_prompt = build_fewshot_prompt(ex)
            else:
                raise ValueError(
                    f"Invalid prompt build method {workload_config.prompt_build_method}"
                )

            # Convert document prompts to tokens
            doc_tokens_list = []
            for doc_prompt in doc_prompts:
                doc_tokens = self._tokenizer.encode(doc_prompt)[1:]  # Remove BOS
                doc_tokens_list.append(doc_tokens)

            # Convert query to tokens
            full_q_tokens = self._tokenizer.encode(q_prompt)[1:]  # Remove BOS

            # Build full prompt with separators
            prompt_tokens = build_rag_prompt_tokens(
                system_prompt_tokens,
                doc_tokens_list,
                full_q_tokens,
                separator_tokens,
            )
            fix_doc_tokens_list = []
            for doc_tokens in doc_tokens_list:
                fix_doc_tokens_list.append(
                    system_prompt_tokens + separator_tokens + doc_tokens
                )

            # Store document tokens for precompute
            self._document_tokens.append(fix_doc_tokens_list)
            self._request_tokens.append(prompt_tokens)
            self._answers.append(ex["answers"])

    def _precompute_documents(self, llm: LLM):
        """Precompute KV cache for document chunks using the same LLM instance"""
        logger.info("Starting document precomputation...")

        # Calculate KV size
        kv_size_calculator = KVSizeCalculator(
            self._model_config.num_key_value_heads,
            self._model_config.head_dim
            if self._model_config.head_dim
            else self._model_config.hidden_size
            // self._model_config.num_attention_heads,
            self._model_config.num_hidden_layers,
            2,  # FP16 precision
        )

        current_size_taken = 0
        precomputed_count = 0
        round_up_token_cnt = self.workload_config.kv_chunk_size

        for i, doc_tokens_list in enumerate(self._document_tokens):
            # TODO: add the size limit if needed
            # if current_size_taken >= kv_storage_size:
            #     break

            # Calculate size for this document set
            total_doc_tokens = sum(len(doc_tokens) for doc_tokens in doc_tokens_list)
            # Round up to chunk size
            total_doc_tokens = (
                (total_doc_tokens + round_up_token_cnt - 1) // round_up_token_cnt
            ) * round_up_token_cnt

            this_case_size = kv_size_calculator.get_kv_size(total_doc_tokens)

            # if current_size_taken + this_case_size > kv_storage_size:
            #     break

            # Precompute each document chunk
            for doc_tokens in doc_tokens_list:
                # Use minimal generation to trigger KV cache storage
                sampling_params = SamplingParams(temperature=0, max_tokens=1)
                try:
                    llm.generate(
                        prompt_token_ids=doc_tokens, sampling_params=sampling_params
                    )
                except Exception as e:
                    logger.warning(f"Precompute failed for document chunk: {e}")
                    continue

            current_size_taken += this_case_size
            precomputed_count += 1

        logger.info(
            f"Precomputed {precomputed_count} document sets, "
            f"used {current_size_taken} bytes of KV cache"
        )
        return precomputed_count

    def run_benchmark(
        self,
        llm: LLM,
        sampling_params: SamplingParams,
    ) -> float:
        """Run the benchmark - optimized using vLLM's batch processing approach like benchmark_throughput.py"""
        self._results = []

        # Step 1: Precompute document KV cache using the same LLM instance
        precomputed_count = self._precompute_documents(llm)

        logger.info(f"Precomputed {precomputed_count} document sets for KV cache")
        # Step 2: Run benchmark requests - using vLLM's batch processing approach
        logger.info("Starting benchmark (throughput-focused with batch processing)...")

        # Build prompts and sampling params lists exactly like vLLM benchmark
        prompts = []
        sampling_params_list = []

        for prompt_tokens in self._request_tokens:
            # Create TokensPrompt for each request (like vLLM benchmark)
            prompts.append(TokensPrompt(prompt_token_ids=prompt_tokens))
            # Create individual SamplingParams for each request
            sampling_params_list.append(sampling_params)

        # Run batch inference - exactly like vLLM benchmark
        start_time = time.perf_counter()
        try:
            outputs = llm.generate(prompts, sampling_params_list)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # Process all outputs
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                prompt_token_count = len(output.prompt_token_ids)
                generation_token_count = len(output.outputs[0].token_ids)

                response = Response(
                    request_id=i,
                    body=generated_text,
                    ttft=0.0,  # Not meaningful for batch processing
                    generation_time=elapsed_time
                    / len(outputs),  # Average time per request
                    prompt_tokens=prompt_token_count,
                    generation_tokens=generation_token_count,
                    launch_time=start_time,
                    finish_time=end_time,
                )
                self._results.append(response)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fall back to sequential processing if batch fails
            logger.info("Falling back to sequential processing...")
            elapsed_time = self._run_sequential_fallback(
                llm, sampling_params, self._request_tokens
            )

        # Calculate and print throughput metrics like vLLM benchmark
        total_prompt_tokens = sum(r.prompt_tokens for r in self._results)
        total_output_tokens = sum(r.generation_tokens for r in self._results)
        total_tokens = total_prompt_tokens + total_output_tokens

        print("\n=== Throughput Results ===")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Total requests: {len(self._results)}")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Requests per second: {len(self._results) / elapsed_time:.2f}")
        print(f"Tokens per second: {total_tokens / elapsed_time:.2f}")
        print(f"Output tokens per second: {total_output_tokens / elapsed_time:.2f}")

        logger.info(f"Completed {len(self._results)} requests")
        return elapsed_time

    def _run_sequential_fallback(
        self,
        llm: LLM,
        sampling_params: SamplingParams,
        prompt_tokens_list: List[List[int]],
    ) -> float:
        """Fallback to sequential processing if batch processing fails"""
        start_time = time.perf_counter()

        for i, prompt_tokens in enumerate(prompt_tokens_list):
            try:
                output = llm.generate(
                    prompt_token_ids=prompt_tokens, sampling_params=sampling_params
                )

                generated_text = output[0].outputs[0].text
                prompt_token_count = len(output[0].prompt_token_ids)
                generation_token_count = len(output[0].outputs[0].token_ids)

                response = Response(
                    request_id=i,
                    body=generated_text,
                    ttft=0.0,
                    generation_time=0.0,  # Will be calculated at the end
                    prompt_tokens=prompt_token_count,
                    generation_tokens=generation_token_count,
                    launch_time=start_time,
                    finish_time=0.0,
                )
                self._results.append(response)

            except Exception as e:
                logger.error(f"Error processing request {i}: {e}")
                # Add dummy response for failed requests
                response = Response(
                    request_id=i,
                    body="",
                    ttft=0.0,
                    generation_time=0.0,
                    prompt_tokens=len(prompt_tokens),
                    generation_tokens=0,
                    launch_time=start_time,
                    finish_time=0.0,
                )
                self._results.append(response)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Update all responses with final timing
        for response in self._results:
            response.generation_time = elapsed_time / len(self._results)
            response.finish_time = end_time

        return elapsed_time

    def summary(self, total_time: float, is_online: bool = False) -> pd.DataFrame:
        """Generate summary statistics"""
        cnt = len(self._results)
        assert cnt > 0, "No results to summarize"
        if is_online:
            ttfts = [r.ttft for r in self._results]
            tpots = [
                r.generation_time / r.generation_tokens
                if r.generation_tokens > 0
                else 0
                for r in self._results
            ]

            avg_ttft = sum(ttfts) / cnt
            avg_tpot = sum(tpots) / cnt

        generation_times = [r.generation_time for r in self._results]
        prefill_token_cnts = [r.prompt_tokens for r in self._results]
        generation_token_cnts = [r.generation_tokens for r in self._results]

        # Calculate quality scores
        quality = []
        for i in range(cnt):
            generated_text = self._results[i].body
            if self._build_method == PromptBuildMethodType.QA:
                quality.append(
                    max(
                        [
                            compute_f1(generated_text, answer, self._tokenizer)
                            for answer in self._answers[i]
                        ]
                    )
                )
            elif self._build_method == PromptBuildMethodType.FEW_SHOT:
                quality.append(
                    max(
                        [
                            compute_rl(generated_text, answer)
                            for answer in self._answers[i]
                        ]
                    )
                )
            else:
                raise ValueError(f"Invalid prompt build method {self._build_method}")

        avg_quality = sum(quality) / cnt
        thput = cnt / total_time

        if is_online:
            df = pd.DataFrame(
                {
                    "quality": quality,
                    "ttft": ttfts,
                    "tpot": tpots,
                    "generation_time": generation_times,
                    "prefill_token_cnt": prefill_token_cnts,
                    "generation_token_cnt": generation_token_cnts,
                }
            )

            logger.info(
                f"Summary: {cnt} requests, average_ttft={avg_ttft:.4f} (second)\n"
                f"average_tpot={avg_tpot:.4f} (second)\n"
                f"throughput={thput:.4f} (req/s)\n"
                f"average_quality={avg_quality:.4f}\n"
            )
        else:
            df = pd.DataFrame(
                {
                    "quality": quality,
                    "throughput": [cnt / total_time] * cnt,
                    "generation_time": generation_times,
                    "prefill_token_cnt": prefill_token_cnts,
                    "generation_token_cnt": generation_token_cnts,
                }
            )

            logger.info(
                f"Summary: {cnt} requests, total_time={total_time:.4f} (second)\n"
                f"throughput={thput:.4f} (req/s)\n"
                f"average_quality={avg_quality:.4f}\n"
            )

        return df


def run_rag_benchmark(args):
    """Main function to run the RAG benchmark"""
    build_prompt_method_str = args.prompt_build_method.upper()
    if build_prompt_method_str == "QA":
        build_prompt_method = PromptBuildMethodType.QA
    elif build_prompt_method_str == "FEW_SHOT":
        build_prompt_method = PromptBuildMethodType.FEW_SHOT
    else:
        raise ValueError(f"Invalid prompt build method {build_prompt_method_str}")

    # TODO: use LMConfig to detect separator
    lmconfig = lmcache_get_config()
    workload_config = WorkloadConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        dataset=args.dataset,
        start_index=args.start_index,
        end_index=args.end_index,
        shuffle=args.shuffle,
        system_prompt=args.system_prompt,
        separator=lmconfig.blend_special_str,
        query_prompt=args.query_prompt,
        prompt_build_method=build_prompt_method,
        max_tokens=args.max_tokens,
        kv_chunk_size=lmconfig.chunk_size,
    )

    manager = OfflineRAGManager(workload_config)

    with build_llm_with_lmcache(args.model) as llm:
        # FIXME: top_p
        sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

        total_time = manager.run_benchmark(llm, sampling_params)

        logger.info(f"Finished benchmarking, dumping summary to {args.output}")
        summary = manager.summary(total_time)
        summary.to_csv(args.output, index=False)


def cleanup_handler(signum, frame):
    """Handle cleanup on termination signals"""
    logger.info(f"Received signal {signum}, cleaning up...")
    # Ensure LMCache engine is properly destroyed
    try:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
    sys.exit(0)


def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    # Register atexit handler as backup
    atexit.register(lambda: LMCacheEngineBuilder.destroy(ENGINE_NAME))

    args = parse_arguments()
    build_prompt_method_str = args.prompt_build_method.upper()
    if build_prompt_method_str == "QA":
        build_prompt_method = PromptBuildMethodType.QA
    elif build_prompt_method_str == "FEW_SHOT":
        build_prompt_method = PromptBuildMethodType.FEW_SHOT
    else:
        raise ValueError(f"Invalid prompt build method {build_prompt_method_str}")

    if len(args.system_prompt) == 0:
        args.system_prompt = system_prompt_set[build_prompt_method]
    if len(args.query_prompt) == 0:
        args.query_prompt = query_prompt_set[build_prompt_method]
    if len(args.tokenizer) == 0:
        args.tokenizer = args.model

    args.system_prompt = args.system_prompt.encode().decode("unicode_escape")
    args.query_prompt = args.query_prompt.encode().decode("unicode_escape")

    if args.verbose:
        global logger
        logger = init_logger(__name__, log_level=logging.DEBUG)

    run_rag_benchmark(args)


if __name__ == "__main__":
    main()
