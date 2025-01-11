import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from transformers import AutoTokenizer
import random
import openai
import pandas as pd
from utils import AsyncLoopWrapper, init_logger
from utils import build_rag_prompt, PromptBuildMethodType, load_dataset, compute_f1, compute_rl

logger = init_logger(__name__, logging.INFO)


@dataclass
class WorkloadConfig:
    # Overall QPS
    qps: float
    # Model name
    model: str
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
    # Model name for openai API.
    model_api_name: str
    

@dataclass
class Response:
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float
    


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse RAG benchmark configurations.")
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="The dataset path")
    parser.add_argument("--start-index",
                        type=int,
                        default=0,
                        help="Start index of the workload")
    parser.add_argument("--end-index",
                        type=int,
                        default=-1,
                        help="End index of the workload")
    parser.add_argument("--shuffle",
                        action="store_true",
                        help="Random shuffle")
    parser.add_argument("--system-prompt",
                        type=str,
                        required=True,
                        help="System prompt")
    parser.add_argument("--separator",
                        type=str,
                        default="",
                        help="Separator")
    parser.add_argument("--query-prompt",
                        type=str,
                        default="",
                        help="Query prompt")
    parser.add_argument("--prompt-build-method",
                        type=str,
                        required=True,
                        help="Prompt build method")
    parser.add_argument("--base-url",
                        type=str,
                        required=True,
                        help="Base URL of the serving engine endpoint")
    parser.add_argument("--api-key",
                        type=str,
                        default="EMPTY",
                        help="API key of the serving engine endpoint")
    parser.add_argument("--output",
                        type=str,
                        default="summary.csv",
                        help="The output file name (ended with csv or txt) "
                        "for the summary csv and txt")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip precompute")
    parser.add_argument("--kv-storage-size",
                        type=str,
                        default="",
                        help="KV storage size")
    parser.add_argument("--kv-storage-token-unit",
                        type=int,
                        default=256,
                        help="KV storage token unit")
    parser.add_argument("--kv-precision-bit",
                        type=int,
                        default=16,
                        help="KV cache precision bit")
    parser.add_argument("--time",
                        type=int,
                        default=None,
                        help="The total running time in seconds")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Whether to enable verbose logging")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=32,
                        help="Max tokens for each generation")
    parser.add_argument("--model-api-name", type=str,
                        default="",
                        help="Model API name.")
    args = parser.parse_args()
    return args

class RequestExecutor:
    def __init__(self, base_url: str, api_key: str, prompt_build_method: PromptBuildMethodType, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        self.prompt_build_method = prompt_build_method
    async def _async_launch_request(self, prompt, max_tokens):
        start_time = time.time()
        first_token_time = None
        words = ""
        response = None
        if self.prompt_build_method == PromptBuildMethodType.QA:
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0,
                stream=True,
                max_tokens=max_tokens,
                stream_options={"include_usage": True})
        elif self.prompt_build_method == PromptBuildMethodType.FEW_SHOT:
            response = await self.client.completions.create(
                prompt=prompt,
                model=self.model,
                temperature=0,
                stream=True,
                max_tokens=max_tokens,
                stream_options={"include_usage": True})
        else:
            raise ValueError(f"Invalid prompt build method {self.prompt_build_method}")
        async for tok in response:
            if not tok.choices:
                continue
            chunk_message = tok.choices[0].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != "":
                    first_token_time = time.time()
                words += chunk_message
        tokens_out = tok.usage.completion_tokens
        tokens_prefill = tok.usage.prompt_tokens
        finish_time = time.time()
        return Response(body=words,
                        ttft=first_token_time - start_time,
                        generation_time=finish_time - first_token_time,
                        prompt_tokens=tokens_prefill,
                        generation_tokens=tokens_out,
                        launch_time=start_time,
                        finish_time=finish_time)
    def launch_request(self, prompt, max_tokens, finish_callback):
        """
        finish_callback: Callable[[Response], None]
        """
        real_callback = lambda x: finish_callback(x.result())
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(prompt, max_tokens), self.loop)
        future.add_done_callback(real_callback)

def warmup_engine(executor: RequestExecutor):
    logger.info("Warming up the engine")
    for i in range(10):
        prompt = f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}."
        executor.launch_request(prompt, 100, lambda x: None)

    AsyncLoopWrapper.WaitLoop()
    
class RAGManager:
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
        self._prompts = []
        self._answers = []
        self._build_method = workload_config.prompt_build_method
        for ex in eval_dataset:
            prompt, _ = build_rag_prompt(workload_config.system_prompt, 
                             ex,
                             workload_config.query_prompt,
                             workload_config.separator,
                             workload_config.prompt_build_method)
            self._prompts.append(prompt)
            self._answers.append(ex["answers"])
        self._tokenizer = AutoTokenizer.from_pretrained(workload_config.model)
        self._last_request_time = -1.0
        self._last_request_index = 0
        assert workload_config.qps > 0
        self._gap = 1.0 / workload_config.qps
        self._max_tokens = workload_config.max_tokens
        self._generated_text = []
        self._generation_time = []
        self._prefill_tok_cnt = []
        self._generation_tok_cnt = []
        self._ttft = []
        self._tpot = []
    def _update_result(self, response: Response):
        self._generated_text.append(response.body)
        self._ttft.append(response.ttft)
        self._tpot.append(response.generation_time / response.generation_tokens)
        self._generation_time.append(response.generation_time)
        self._prefill_tok_cnt.append(response.prompt_tokens)
        self._generation_tok_cnt.append(response.generation_tokens)
    def step(self, timestamp: float, executor: RequestExecutor) -> bool:
        if self._last_request_index >= len(self._prompts):
            return False
        if self._last_request_time < 0 or timestamp >= self._last_request_time + self._gap:
            prompt = self._prompts[self._last_request_index]
            self._last_request_time = timestamp
            self._last_request_index += 1
            executor.launch_request(prompt, self._max_tokens, self._update_result)
        return True
    
    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        cnt = len(self._ttft)
        assert cnt > 0
        avg_ttft = sum(self._ttft) / cnt
        avg_tpot = sum(self._tpot) / cnt
        # Create a dataframe
        quality = []
        for i in range(cnt):
            if self._build_method == PromptBuildMethodType.QA:
                quality.append(max([compute_f1(self._generated_text[i], answer, self._tokenizer) for answer in self._answers[i]]))
            elif self._build_method == PromptBuildMethodType.FEW_SHOT:
                quality.append(max([compute_rl(self._generated_text[i], answer) for answer in self._answers[i]]))
            else:
                raise ValueError(f"Invalid prompt build method {self._build_method}")
        df = pd.DataFrame({
            "quality": quality,
            "ttft": self._ttft,
            "tpot": self._tpot,
            "generation_time": self._generation_time,
            "prefill_token_cnt": self._prefill_tok_cnt,
            "generation_token_cnt": self._generation_tok_cnt
        })
        logger.info(f"Summary: {cnt} requests, average_ttft={avg_ttft}, average_tpot={avg_tpot}")
        return df
        


def run_rag(args):
    build_prompt_method_str = args.prompt_build_method.upper()
    build_prompt_method = None
    if build_prompt_method_str == "QA":
        build_prompt_method = PromptBuildMethodType.QA
    elif build_prompt_method_str == "FEW_SHOT":
        build_prompt_method = PromptBuildMethodType.FEW_SHOT
    else:
        raise ValueError(f"Invalid prompt build method {build_prompt_method_str}")
    workload_config = WorkloadConfig(
        qps=args.qps,
        model=args.model,
        dataset=args.dataset,
        start_index=args.start_index,
        end_index=args.end_index,
        shuffle=args.shuffle,
        system_prompt=args.system_prompt,
        separator=args.separator,
        query_prompt=args.query_prompt,
        prompt_build_method=build_prompt_method,
        max_tokens=args.max_tokens,
        model_api_name=args.model_api_name)
    executor = RequestExecutor(base_url=args.base_url,
                               api_key=args.api_key,
                               prompt_build_method=build_prompt_method, 
                               model=args.model_api_name)
    if args.skip_precompute:
        warmup_engine(executor)
    manager = RAGManager(workload_config)
    # TODO: Step interval accuracy.
    step_interval = 0.1
    num_steps = 0
    start_time = time.time()
    # last_summary_time = start_time
    try:
        while True:
            num_steps += 1
            effective = manager.step(time.time(), executor)
            if not effective:
                break
            time.sleep(step_interval)
            # How to control QPS.
            '''
            if time.time() - last_summary_time > args.log_interval:
                manager.summary(last_summary_time, time.time())
                last_summary_time = time.time()
            '''
            if args.time is not None and time.time() - start_time > args.time:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")

    AsyncLoopWrapper.StopLoop()

    logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    summary = manager.summary(0, time.time())
    summary.to_csv(args.output, index=False)


def main():
    args = parse_arguments()
    args.system_prompt = args.system_prompt.encode().decode('unicode_escape')
    args.query_prompt = args.query_prompt.encode().decode('unicode_escape')
    if args.verbose:
        global logger
        logger = init_logger(__name__, level=logging.DEBUG)
    if not args.skip_precompute:
        from precompute import run_precompute
        st_idx, ed_idx, model = run_precompute(args)
        logger.info(f"Precompute finished, start index={st_idx}, end index={ed_idx}, model {model}")
        assert st_idx == args.start_index
        if args.end_index < 0:
            args.end_index = ed_idx
        if len(args.model_api_name) == 0:
            args.model_api_name = model
    run_rag(args)
    

if __name__ == "__main__":
    main()
