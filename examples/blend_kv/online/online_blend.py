import sys
import threading
import time
from io import StringIO

from openai import OpenAI

import time

import lmcache_vllm
from lmcache_vllm.blend_adapter import combine_input_prompt_chunks

class KVPreCompute:
    def __init__(self, openai_api_key, openai_api_base):
        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model = self.client.models.list().data[0].id
    def precompute_kv(self, text_chunk):
        completion = self.client.completions.create(
            prompt=text_chunk,
            model=self.model,
            max_tokens=1,
        )
        return completion.choices[0].text
    


class Printer:

    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()

    def _print(self):
        idx = 0
        while not self._stop_event.is_set():
            arrows = ">" * (idx % 6)
            string = "{:6s}".format(arrows)
            print("\033[31m\r" + string + "\033[0m", end="", flush=True)
            idx += 1
            time.sleep(0.2)

    def start(self):
        if self._thread is None:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._print)
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
            print("\033[31m\r>>>>> \033[0m", end="", flush=True)


class ChatSession:

    def __init__(self, openai_api_key, openai_api_base, context_text):
        self.client = client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        models = client.models.list()
        self.model = models.data[0].id
        self.messages = [
            {
                "role":
                "user",
                "content": context_text
            },
            {
                "role": "assistant",
                "content": "I've got your document"
            },
        ]

        self.printer = Printer()

    def on_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def on_server_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def chat(self):
        user_prompt = input("User: ")
        self.on_user_message(user_prompt)

        self.printer.start()
        start = time.perf_counter()
        end = None

        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            temperature=0,
            stream=True)

        output_buffer = StringIO()
        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                self.printer.stop()
                print(chunk_message, end="", flush=True)
                output_buffer.write(chunk_message)
                if end is None:
                    end = time.perf_counter()
        self.on_server_message(output_buffer.getvalue())
        print("")
        print("\033[33mTTFT:", end - start, "\033[0m")
        print("Total time:", time.perf_counter() - start)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <port>", file=sys.stderr)
        exit(1)

    port = sys.argv[1]

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"

    context_files = ["../chunk1.txt", "../chunk2.txt"]
    chunks = []

    for context_file in context_files:
        with open(context_file, "r") as fin:
            context = fin.read()
        chunks.append(context)

    kv_precompute = KVPreCompute(openai_api_key, openai_api_base)
    print(
        "-------------- Pre-computing KV cache for the chunks -------------------")
    for chunk in chunks:
        kv_precompute.precompute_kv(chunk)

    sys_prompt = "Here's a document from the user: "
    context_text = combine_input_prompt_chunks([sys_prompt, chunks[0], chunks[1]])

    chat_session = ChatSession(openai_api_key, openai_api_base, context_text)

    while True:
        chat_session.chat()
        print("")
