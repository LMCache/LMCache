from openai import OpenAI
import sys
import time

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <port>")
    exit(1)

port = sys.argv[1]

# Get the looooong context
context_file = "f.txt"
with open(context_file, "r") as fin:
    context = fin.read()

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = f"http://localhost:{port}/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

start = time.perf_counter()
end = None
chat_completion = client.chat.completions.create(
    messages=[
    {
        "role": "user",
        "content": f"I've got a document, here's the content:```\n{context}\n```."
    }, 
    {
        "role":
        "assistant",
        "content":
        "I've got your document"
    }, 
    {
        "role": "user",
        "content": "What does this document mainly talks about?"
    }],
    model=model,
    temperature=0,
    stream=True
)

print("\033[33mChat completion results:\033[0m")
for chunk in chat_completion:
    chunk_message = chunk.choices[0].delta.content
    print(chunk_message, end="", flush=True) if chunk_message is not None else None
    if end is None:
        end = time.perf_counter()
print("")
print("\033[33mTTFT:", end - start, "\033[0m")
print("Total time:", time.perf_counter() - start)
