import os

import chat_session
import streamlit as st
from transformers import AutoTokenizer

# Change the following variables as needed

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
PORT = 8000


@st.cache_resource
def get_tokenizer():
    global MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


tokenizer = get_tokenizer()


@st.cache_data
def read_context() -> str:
    context_text = None
    context_file = os.path.join(os.pardir, 'ffmpeg.txt')
    with open(context_file, 'r') as f:
        context_text = f.read()
    assert context_text is not None
    return context_text


context = read_context()

container = st.container(border=True)

with st.sidebar:
    session = chat_session.ChatSession(PORT)

    system_prompt = st.text_area(
        "System prompt:",
        "You are a helpful assistant. I will now give you a document and "
        "please answer my question afterwards based on the content in document",
    )

    session.set_context([system_prompt] + [context])
    num_tokens = tokenizer.encode(session.get_context())
    container.header(f"The context given to LLM: ({len(num_tokens)} tokens)",
                     divider="grey")
    container.text(session.get_context())

    messages = st.container(height=400)
    if prompt := st.chat_input("Type the question here"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))