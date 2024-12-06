import argparse
import asyncio
import time
from dataclasses import dataclass

import openai
import pandas as pd
from utils import AsyncLoopWrapper, init_logger

logger = init_logger(__name__)


@dataclass
class WorkloadConfig:
    # Max number of users in the system concurrently
    num_users: int

    # Length of system prompt
    system_prompt_len: int

    # Length of the answer in one round
    answer_len: int

    # Number of rounds in the conversation
    num_rounds: int

    # Overall QPS
    qps: int

    # Model name
    model: str


@dataclass
class UserConfig:
    # User id
    user_id: int

    # System prompt length
    system_prompt_len: int

    # Answer length
    answer_len: int

    # Gap between two requests
    gap_between_requests: int

    # Num rounds
    num_rounds: int

    @staticmethod
    def new_user_config(user_id: int,
                        workload_config: WorkloadConfig) -> 'UserConfig':
        return UserConfig(user_id=user_id,
                          system_prompt_len=workload_config.system_prompt_len,
                          answer_len=workload_config.answer_len,
                          gap_between_requests=workload_config.num_users /
                          workload_config.qps,
                          num_rounds=workload_config.num_rounds)


class ChatHistory:

    def __init__(self, ):
        self.history = []

    def on_user_query(self, query: str):
        if len(self.history) == 0:
            self.history.append({"role": "user", "content": query})
        else:
            assert self.history[-1][
                "role"] == "assistant", "Expect system response"
            self.history.append({"role": "user", "content": query})

    def on_system_response(self, response: str):
        assert len(self.history) > 0, "Expect user query"
        assert self.history[-1]["role"] == "user", "Expect user query"
        self.history.append({"role": "assistant", "content": response})

    def get_messages_for_openai(self):
        return self.history


@dataclass
class Response:
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int


class RequestExecutor:

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()

    async def _async_launch_request(self, messages, max_tokens):
        start_time = time.time()
        first_token_time = None
        words = ""

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True})

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

        return Response(body=words,
                        ttft=first_token_time - start_time,
                        generation_time=time.time() - first_token_time,
                        prompt_tokens=tokens_prefill,
                        generation_tokens=tokens_out)

    def launch_request(self, chat_history: ChatHistory, max_tokens: int,
                       finish_callback):
        """
        finish_callback: Callable[[Response], None]
        """
        messages = chat_history.get_messages_for_openai()
        real_callback = lambda x: finish_callback(x.result())
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens), self.loop)
        future.add_done_callback(real_callback)


class UserSession:

    def __init__(self, user_config: UserConfig):
        self.user_config = user_config
        self.last_request_time = None
        self.chat_history = ChatHistory()
        self.question_id = 0

        self.has_unfinished_request = False

        self.prompt_lengths = []
        self.generation_lengths = []
        self.ttfts = []
        self.generation_times = []

        self.finished = False

    def _update_result(self, response: Response):
        self.prompt_lengths.append(response.prompt_tokens)
        self.generation_lengths.append(response.generation_tokens)
        self.ttfts.append(response.ttft)
        self.generation_times.append(response.generation_time)

    def _build_system_prompt(self):
        dummy_text = ' '.join(["hi"] *
                              (self.user_config.system_prompt_len - 10))
        system_prompt = f"Hi, I'm user {self.user_config.user_id}." + \
                        f"Here are some text: {dummy_text}."
        return system_prompt

    def _build_new_question(self):
        self.question_id += 1
        return f"Here's question #{self.question_id}: can you tell me " + \
                "a new long story with a happy ending?"

    def on_request_finished(self, response: Response):
        self.chat_history.on_system_response(response.body)
        self.has_unfinished_request = False
        logger.debug(
            f"User {self.user_config.user_id} finished one request. "
            f"Prompt tokens: {response.prompt_tokens}, ",
            f"generation tokens: {response.generation_tokens}")
        self._update_result(response)

    def step(self, timestamp: float, request_executor: RequestExecutor):
        if self.question_id >= self.user_config.num_rounds:
            self.finished = True
            return

        if self.last_request_time is None:
            logger.debug(f"Issuing the request {self.question_id}")
            self.last_request_time = timestamp
            system_prompt = self._build_system_prompt()
            question = self._build_new_question()
            self.chat_history.on_user_query(system_prompt + question)
            request_executor.launch_request(self.chat_history,
                                            self.user_config.answer_len,
                                            self.on_request_finished)
            self.has_unfinished_request = True
            return

        if timestamp - self.last_request_time > \
                self.user_config.gap_between_requests:
            if self.has_unfinished_request:
                logger.warning(
                    f"User {self.user_config.user_id} has an unfinished "
                    "request and unable to fit the QPS requirement.")
                return

            logger.debug(f"Issuing the request {self.question_id}")
            self.last_request_time = timestamp
            question = self._build_new_question()
            self.chat_history.on_user_query(question)
            request_executor.launch_request(self.chat_history,
                                            self.user_config.answer_len,
                                            self.on_request_finished)
            self.has_unfinished_request = True
            return

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["prompt_tokens"] = self.prompt_lengths
        df["generation_tokens"] = self.generation_lengths
        df["ttft"] = self.ttfts
        df["generation_time"] = self.generation_times
        df["user_id"] = self.user_config.user_id
        df["question_id"] = range(1, len(self.prompt_lengths) + 1)
        return df


class UserSessionManager:

    def __init__(self, workload_config: WorkloadConfig):
        self.workload_config = workload_config
        self.sessions = []

        gap_between_requests_per_user = \
                workload_config.num_users / workload_config.qps
        session_alive_time = gap_between_requests_per_user * \
                (workload_config.num_rounds - 1)
        self.gap_between_users = session_alive_time / workload_config.num_users

        self.user_id = 0
        self.last_user_join = 0
        self.session_summaries = []
        self.start_time = None

    def _create_user_session(self):
        self.user_id += 1
        user_config = UserConfig.new_user_config(self.user_id,
                                                 self.workload_config)
        user_session = UserSession(user_config)
        self.sessions.append(user_session)

    def _remove_finished_sessions(self):
        sessions_to_remove = [s for s in self.sessions if s.finished]
        if len(sessions_to_remove) > 0:
            logger.info(
                f"Removing {len(sessions_to_remove)} finished sessions, now "
                f"active users: {len(self.sessions) - len(sessions_to_remove)}"
            )
            for session in sessions_to_remove:
                self.session_summaries.append(session.summary())
        self.sessions = [s for s in self.sessions if not s.finished]

    def step(self, timestamp: float, executor: RequestExecutor):
        if self.start_time is None:
            self.start_time = timestamp

        if timestamp - self.last_user_join > self.gap_between_users:
            self._create_user_session()
            self.last_user_join = timestamp
            logger.info(
                f"Joined a new user, now active users: {len(self.sessions)}")

        for session in self.sessions:
            session.step(timestamp, executor)

        self._remove_finished_sessions()

    def summary(self, timestamp: float) -> pd.DataFrame:
        if len(self.session_summaries) == 0 and len(self.sessions) == 0:
            return pd.DataFrame()

        df = pd.concat([s for s in self.session_summaries] +
                       [s.summary() for s in self.sessions])

        # Metrics to calculate:
        # - QPS
        # - Average prompt throughput
        # - Average generation throughput
        # - Average TTFT
        total_time = timestamp - self.start_time
        total_requests = len(df)
        qps = total_requests / total_time
        total_prompt_tokens = df["prompt_tokens"].sum()
        total_generation_tokens = df["generation_tokens"].sum()
        average_prefill_speed = total_prompt_tokens / total_time
        average_generation_speed = total_generation_tokens / total_time
        average_ttft = df["ttft"].mean()
        print(
            "==================== Performance summary ======================")
        print(f"  \033[33mQPS: \033[32m{qps:.4f}\033[0m")
        print("  \033[33mAverage prompt throughput: "
              f"\033[32m{average_prefill_speed:.4f}\033[0m")
        print("  \033[33mAverage generation throughput: "
              f"\033[32m{average_generation_speed:.4f}\033[0m")
        print(f"  \033[33mAverage TTFT: \033[32m{average_ttft:.4f}\033[0m")
        print(
            "===============================================================")
        return df


def warmup_engine(executor):
    logger.info("Warming up the engine")
    for i in range(10):
        chat_history = ChatHistory()
        chat_history.on_user_query(
            f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}.")
        executor.launch_request(chat_history, 100, lambda x: None)

    AsyncLoopWrapper.WaitLoop()


def parse_arguments() -> WorkloadConfig:
    parser = argparse.ArgumentParser(
        description="Parse benchmark configurations.")

    parser.add_argument("--num-users",
                        type=int,
                        required=True,
                        help="Max number of users in the system concurrently")
    parser.add_argument("--system-prompt-len",
                        type=int,
                        required=True,
                        help="Length of system prompt")
    parser.add_argument("--answer-len",
                        type=int,
                        required=True,
                        help="Length of the answer in one round")
    parser.add_argument("--num-rounds",
                        type=int,
                        required=True,
                        help="Number of rounds in the conversation")
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--base-url",
                        type=str,
                        required=True,
                        help="Base URL of the serving engine endpoint")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
        help="The time between two summary loggings in seconds")

    args = parser.parse_args()
    return args

    return WorkloadConfig(num_users=args.num_users,
                          system_prompt_len=args.system_prompt_len,
                          answer_len=args.answer_len,
                          num_rounds=args.num_rounds,
                          qps=args.qps,
                          model=args.model)


def main():
    args = parse_arguments()
    step_interval = 0.1

    executor = RequestExecutor(base_url=args.base_url,
                               api_key="EMPTY",
                               model=args.model)

    warmup_engine(executor)
    workload_config = WorkloadConfig(num_users=args.num_users,
                                     system_prompt_len=args.system_prompt_len,
                                     answer_len=args.answer_len,
                                     num_rounds=args.num_rounds,
                                     qps=args.qps,
                                     model=args.model)

    manager = UserSessionManager(workload_config)

    num_steps = 0
    try:
        while True:
            num_steps += 1
            manager.step(time.time(), executor)
            time.sleep(0.1)

            if num_steps % int(args.log_interval / step_interval) == 0:
                manager.summary(time.time())

    except KeyboardInterrupt:
        logger.info("Interrupted, printing the final result")

    logger.info("Finished the simulation")
    summary = manager.summary(time.time())
    summary.to_csv("summary.csv", index=False)

    AsyncLoopWrapper.StopLoop()


if __name__ == "__main__":
    main()
