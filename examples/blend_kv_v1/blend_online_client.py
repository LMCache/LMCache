# Standard
import argparse
import time

# Third Party
from openai import OpenAI


def print_output(
    client: OpenAI,
    model: str,
    prompt: str,
):
    """Print the output from the API call."""
    start = time.time()
    completion = client.completions.create(
        prompt=prompt,
        model=model,
        temperature=0,
        top_p=0.95,
        max_tokens=10,
        stream=True,
    )

    print("-" * 50)
    full_response = ""
    for c in completion:
        print(c)
    print(f"\nGenerated text: {full_response!r}")
    print(f"Generation took {time.time() - start:.2f} seconds.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Online client for LMCache V1 blend example"
    )
    parser.add_argument(
        "--port", type=int, default=8200, help="Port of the vLLM server (default: 8200)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host of the vLLM server (default: localhost)",
    )
    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Specify the special separators to separate chunks (default: ' # # ')",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup OpenAI client to connect to vLLM server
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{args.host}:{args.port}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Get the model name from the server
    models = client.models.list()
    model = models.data[0].id

    # This example runs two requests with a shared prefix.
    # Define the shared prompt and specific prompts
    sys_prompt = "You are a very helpful assistant."
    chunk1_prompt = "Hello, how are you?" * 250
    chunk2_prompt = "Hello, what's up?" * 250
    chunk3_prompt = "Hey, how can i do" * 250
    blend_special_str = args.blend_special_str  # FIXME: should change
    blend_special_str = f" {blend_special_str}"
    precompute_prompts = [
        (sys_prompt + blend_special_str + chunk1_prompt),
        (sys_prompt + blend_special_str + chunk2_prompt), # FIXME: must add sys_prompt
        (sys_prompt + blend_special_str + chunk3_prompt),
    ]
    test_prompt = (
        sys_prompt
        + blend_special_str
        + chunk2_prompt
        + blend_special_str
        + chunk1_prompt
        + blend_special_str
        + chunk3_prompt
        + blend_special_str
        + "Hello, how are you?"
    )

    print("Starting LMCache V1 blend example with online serving...")
    print(f"Server: {openai_api_base}")
    print(f"Model: {model}")
    print(f"Blend special string: {blend_special_str!r}")

    for prompt in precompute_prompts:
        # Add the first prompt to the cache
        print_output(client, model, prompt)

        # Wait for a while to simulate some delay before the second request
        time.sleep(1)

    print("Precompute prompts done. Now testing the blended request...")
    # Print the second output
    print_output(client, model, test_prompt)


if __name__ == "__main__":
    main()
