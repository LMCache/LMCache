import os
import sys
import threading
import time
from io import StringIO

from openai import OpenAI

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <port>")
    exit(1)

port = sys.argv[1]

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = f"http://localhost:{port}/v1"

context_file_name = os.path.join(os.pardir, "ffmpeg.txt")


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

    def __init__(self):
        self.client = client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        models = client.models.list()
        self.model = models.data[0].id

        context_text = None
        with open(context_file_name, "r") as f:
            context_text = f.read()
        assert context_text is not None
        print(f"Chat session is based on context of file: {context_file_name}")
        self.messages = [
            {
                "role":
                "user",
                "content":
                "I've got a document, "
                f"here's the content:```\n{context_text}\n```."
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


chat_session = ChatSession()

while True:
    chat_session.chat()
    print("")
