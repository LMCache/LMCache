# KV blending example
This is a minimal example demonstrating the KV blending functionality of LMCache.

In `blend_kv.py`, the following code will first calculate the KV cache of two text chunks.
```python
for chunk in chunks:
    precompute_kv(chunk, llm)
```

Then, the text chunks are concatenated together, prepended with a system prompt, and appended with a user's quest.
```python
user_prompt= [sys_prompt, chunks[0], chunks[1], question]
user_prompt = combine_input_prompt_chunks(user_prompt)
```

Finally, the prompt will be sent to the serving engine and the KV blending module will blend the KV for the text chunks.


## How to run
```
python3 blend_kv.py
```

## TODO
- [ ] Add configuration file
- [ ] Add online example
