#!/usr/bin/env python
"""Send multiple requests to fill up the KV cache"""
import requests
import json
import time
import random
import string

URL = 'http://localhost:8000/v1/completions'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

def generate_random_prompt(min_words=300, max_words=500):
    """Generate a random prompt with random words"""
    num_words = random.randint(min_words, max_words)
    words = []
    for _ in range(num_words):
        word_len = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
    return ' '.join(words)

# Generate random prompts each time
NUM_REQUESTS = 20
prompts = [generate_random_prompt() for _ in range(NUM_REQUESTS)]

print("Sending multiple RANDOM requests to fill up KV cache...")
print(f"Each prompt is unique to trigger different cache entries")
print("=" * 60)

for i, prompt in enumerate(prompts):
    print(f"\n[Request {i+1}/{len(prompts)}] Sending prompt ({len(prompt)} chars)...")
    try:
        start = time.time()
        response = requests.post(
            URL,
            json={
                'model': MODEL,
                'prompt': prompt,
                'max_tokens': 20
            },
            timeout=60
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            tokens = result.get('usage', {}).get('prompt_tokens', 'N/A')
            print(f"  ✓ Success! Prompt tokens: {tokens}, Time: {elapsed:.2f}s")
        else:
            print(f"  ✗ Error: {response.status_code}")
            print(f"    {response.text[:200]}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    time.sleep(0.5)  # Small delay between requests

print("\n" + "=" * 60)
print("Done! Check the vLLM server logs for cache behavior.")
print("Also check: ls -la /tmp/lmcache_gds_small")
