# Standard
from dataclasses import dataclass, asdict
import argparse
import contextlib
import logging
import os
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# Third Party
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
import pandas as pd

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
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
    PromptBuildMethodType.QA: "You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. "
    "The answer should be within 5 words.\nPassages:\n",
    PromptBuildMethodType.FEW_SHOT: "Summarize the dialogue into a few short sentences. "
    "The following are some examples.\n\n",
}
query_prompt_set = {
    PromptBuildMethodType.QA: "\n\nAnswer the question directly based on the given passages."
    " Do NOT repeat the question. "
    "The answer should be within 5 words. \nQuestion:",
    PromptBuildMethodType.FEW_SHOT: "",
}


def setup_lmcache_environment(
    chunk_size: int = 256,
    blend_special_str: str = " # # ",
    use_disk: bool = False,
    max_cpu_size_gb: int = 5,
    max_disk_size_gb: int = 10,
):
    """Setup LMCache environment variables for blending"""
    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"

    if use_disk:
        os.environ["LMCACHE_LOCAL_CPU"] = "False"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(max_cpu_size_gb)
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(max_disk_size_gb)
    else:
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(max_cpu_size_gb)


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
        enable_prefix_caching=False,
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
    parser.add_argument("--separator", type=str, default=" # # ", help="Separator")
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
        "--warmup", action="store_true", help="Whether to enable warmup"
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
    parser.add_argument("--kv-chunk-size", type=int, default=256, help="KV chunk size")
    parser.add_argument(
        "--use-disk", action="store_true", help="Use disk backend for LMCache"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max worker threads for concurrent processing",
    )
    parser.add_argument(
        "--kv-storage-size",
        type=str,
        default="30GB",
        help="KV storage size for precomputation",
    )
    parser.add_argument(
        "--kv-precision-bit", type=int, default=16, help="KV cache precision in bits"
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
        self._results_lock = threading.Lock()

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

    def _precompute_documents(self, llm: LLM, kv_storage_size: int):
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
            if current_size_taken >= kv_storage_size:
                break

            # Calculate size for this document set
            total_doc_tokens = sum(len(doc_tokens) for doc_tokens in doc_tokens_list)
            # Round up to chunk size
            total_doc_tokens = (
                (total_doc_tokens + round_up_token_cnt - 1) // round_up_token_cnt
            ) * round_up_token_cnt

            this_case_size = kv_size_calculator.get_kv_size(total_doc_tokens)

            if current_size_taken + this_case_size > kv_storage_size:
                break

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

    def _process_single_request(
        self,
        request_data: Tuple[int, List[int], List[str]],
        llm: LLM,
        sampling_params: SamplingParams,
    ) -> Response:
        """Process a single request (for multithreading)"""
        request_id, prompt_tokens, answers = request_data

        start_time = time.time()
        try:
            # Generate response
            outputs = llm.generate(
                prompt_token_ids=prompt_tokens, sampling_params=sampling_params
            )
            end_time = time.time()

            # Extract response
            generated_text = outputs[0].outputs[0].text
            finish_reason = outputs[0].outputs[0].finish_reason
            prompt_token_count = len(outputs[0].prompt_token_ids)
            generation_token_count = len(outputs[0].outputs[0].token_ids)

            # For offline mode, we approximate TTFT as a fraction of total time
            generation_time = end_time - start_time
            ttft = generation_time * 0.1  # Approximate first token time as 10% of total

            response = Response(
                request_id=request_id,
                body=generated_text,
                ttft=ttft,
                generation_time=generation_time - ttft,
                prompt_tokens=prompt_token_count,
                generation_tokens=generation_token_count,
                launch_time=start_time,
                finish_time=end_time,
            )

            return response

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            # Return a dummy response for failed requests
            return Response(
                request_id=request_id,
                body="",
                ttft=0.0,
                generation_time=0.0,
                prompt_tokens=len(prompt_tokens),
                generation_tokens=0,
                launch_time=start_time,
                finish_time=time.time(),
            )

    def run_benchmark(
        self,
        llm: LLM,
        sampling_params: SamplingParams,
        max_workers: int = 4,
        kv_storage_size: int = 30 * 1024 * 1024 * 1024,
    ):
        """Run the benchmark with multithreading support"""
        self._results = []
        total_requests = len(self._request_tokens)

        # Step 1: Precompute document KV cache using the same LLM instance
        precomputed_count = self._precompute_documents(llm, kv_storage_size)

        # Step 2: Run benchmark requests with multithreading
        logger.info(f"Starting benchmark with {max_workers} workers...")

        # Prepare request data
        request_data = [
            (i, self._request_tokens[i], self._answers[i])
            for i in range(min(precomputed_count, total_requests))
        ]

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            future_to_request = {
                executor.submit(
                    self._process_single_request, data, llm, sampling_params
                ): data[0]
                for data in request_data
            }

            # Collect results as they complete
            for future in as_completed(future_to_request):
                request_id = future_to_request[future]
                try:
                    response = future.result()
                    with self._results_lock:
                        self._results.append(response)
                    logger.info(
                        f"Completed request {request_id + 1}/{len(request_data)}"
                    )
                except Exception as e:
                    logger.error(f"Request {request_id} generated an exception: {e}")

        # Sort results by request_id for consistent output
        self._results.sort(key=lambda x: x.request_id)
        logger.info(f"Completed {len(self._results)} requests")

    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        """Generate summary statistics"""
        cnt = len(self._results)
        assert cnt > 0, "No results to summarize"

        ttfts = [r.ttft for r in self._results]
        tpots = [
            r.generation_time / r.generation_tokens if r.generation_tokens > 0 else 0
            for r in self._results
        ]
        generation_times = [r.generation_time for r in self._results]
        prefill_token_cnts = [r.prompt_tokens for r in self._results]
        generation_token_cnts = [r.generation_tokens for r in self._results]

        avg_ttft = sum(ttfts) / cnt
        avg_tpot = sum(tpots) / cnt

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

        total_time = end_time - start_time
        thput = cnt / total_time

        logger.info(
            f"Summary: {cnt} requests, average_ttft={avg_ttft:.4f} (second)\n"
            f" average_tpot={avg_tpot:.4f} (second)\n"
            f"throughput={thput:.4f} (req/s)\n"
            f"average_quality={avg_quality:.4f}\n"
        )

        return df


# def warmup_engine(llm: LLM, tokenizer):
#     """Warmup the engine with some simple requests"""
#     logger.info("Warming up the engine")
#     sampling_params = SamplingParams(temperature=0, max_tokens=10)

#     for i in range(5):
#         prompt = f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}."
#         prompt_tokens = tokenizer.encode(prompt)
#         llm.generate(prompt_token_ids=prompt_tokens, sampling_params=sampling_params)

#     logger.info("Warm up finished.")


def run_rag_benchmark(args):
    """Main function to run the RAG benchmark"""
    build_prompt_method_str = args.prompt_build_method.upper()
    if build_prompt_method_str == "QA":
        build_prompt_method = PromptBuildMethodType.QA
    elif build_prompt_method_str == "FEW_SHOT":
        build_prompt_method = PromptBuildMethodType.FEW_SHOT
    else:
        raise ValueError(f"Invalid prompt build method {build_prompt_method_str}")

    # Setup LMCache environment
    setup_lmcache_environment(
        chunk_size=args.kv_chunk_size,
        blend_special_str=args.separator,
        use_disk=args.use_disk,
    )

    workload_config = WorkloadConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        dataset=args.dataset,
        start_index=args.start_index,
        end_index=args.end_index,
        shuffle=args.shuffle,
        system_prompt=args.system_prompt,
        separator=args.separator,
        query_prompt=args.query_prompt,
        prompt_build_method=build_prompt_method,
        max_tokens=args.max_tokens,
        kv_chunk_size=args.kv_chunk_size,
    )

    manager = OfflineRAGManager(workload_config)

    with build_llm_with_lmcache(args.model) as llm:
        # if args.warmup:
        #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        #     warmup_engine(llm, tokenizer)

        # FIXME: top_p
        sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

        start_time = time.time()
        kv_storage_size = parse_size(args.kv_storage_size)
        manager.run_benchmark(llm, sampling_params, args.max_workers, kv_storage_size)
        end_time = time.time()

        logger.info(f"Finished benchmarking, dumping summary to {args.output}")
        summary = manager.summary(start_time, end_time)
        summary.to_csv(args.output, index=False)


def main():
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
