import sys

import numpy as np

from openai import OpenAI

assert len(sys.argv) >= 2, f"Usage: python3 {sys.argv[0]} <port1> <port2> ... <portN>"

port_list = sys.argv[1:]
class ChatSession:

    def __init__(self, port, context_separator="###"):
        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{port}/v1"

        self.client = client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        models = client.models.list()
        self.model = models.data[0].id

        self.messages = []

        self.final_context = ""
        self.context_separator = context_separator

    def set_context(self, context_strings):
        contexts = []
        for context in context_strings:
            contexts.append(context)

        self.final_context = self.context_separator.join(contexts)
        self.on_user_message(self.final_context, display=False)
        self.on_server_message("Got it!", display=False)

    def get_context(self):
        return self.final_context

    def on_user_message(self, message, display=True):
        if display:
            print("User message:", message)
        self.messages.append({"role": "user", "content": message})

    def on_server_message(self, message, display=True):
        if display:
            print("Server message:", message)
        self.messages.append({"role": "assistant", "content": message})

    def chat(self, question):
        self.on_user_message(question, False)

        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            temperature=0,
            stream=True)

        server_message = []
        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                server_message.append(chunk_message)

        self.on_server_message("".join(server_message), False)

def warm_up_online(ports_to_warm_up):
    # Generate random string of 32 words, 8 groups.
    dummy_prompt_ints = np.random.randint(10000, size=(8, 32))
    dummy_prompts = [' '.join(map(str, row)) for row in dummy_prompt_ints]
    for port in ports_to_warm_up:
        print(f"Warming up on port {port}")
        for dummy_prompt in dummy_prompts:
            chat_session = ChatSession(port)
            chat_session.set_context(["Wram up. Print the first number."])
            chat_session.chat(dummy_prompt)
        print(f"Warm up on port {port} done.")

if __name__ == "__main__":
    warm_up_online(port_list)