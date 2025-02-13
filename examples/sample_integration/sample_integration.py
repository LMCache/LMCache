import asyncio
import time

from lmcache_vllm.vllm import AsyncEngineArgs
from lmcache_vllm.vllm.entrypoints.openai.api_server import \
    build_async_engine_client_from_engine_args
from lmcache_vllm.vllm.entrypoints.openai.protocol import ChatCompletionRequest
from lmcache_vllm.vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from lmcache_vllm.vllm.entrypoints.openai.serving_engine import BaseModelPath

# Model Configuration
MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"  # Updated model
MODEL_NAME = "Mistral-7B-Instruct-v0.2"
RESPONSE_ROLE = "assistant"


def create_chat_request(content):
    """Create a ChatCompletionRequest with the given content."""
    return ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": content
        }],
        max_tokens=50,
        temperature=0.7,
        seed=42,
    )


async def send_request(openai_server, request, index):
    """Send a request and measure response time."""
    start_time = time.time()
    response = await openai_server.create_chat_completion(request)
    end_time = time.time()
    latency = end_time - start_time
    print(f"Response {index + 1} (Latency: {latency:.4f}s): "
          f"{response.choices[0].message.content}")


async def main():
    """
    Initialize vLLM engine, send multiple requests, and print responses
    with latency and cache testing.
    """
    engine_args = AsyncEngineArgs(model=MODEL_PATH)
    print(f"Initializing vLLM engine with args: {engine_args}")

    async with (build_async_engine_client_from_engine_args(engine_args) as
                engine):
        openai_server = OpenAIServingChat(
            engine,
            await engine.get_model_config(),
            [BaseModelPath(name=MODEL_NAME, model_path=MODEL_PATH)],
            RESPONSE_ROLE,
            lora_modules=None,
            prompt_adapters=None,
            request_logger=None,
            chat_template=None)

        prompts = [
            "Hello, how are you?",
            "Hello, how are you?",  # Same prompt to test KV cache
            "Hello, how are you? I am good. Please tell me more about you"
        ]

        # Send requests sequentially
        for idx, content in enumerate(prompts):
            request = create_chat_request(content)
            await send_request(openai_server, request, idx)


if __name__ == "__main__":
    asyncio.run(main())
