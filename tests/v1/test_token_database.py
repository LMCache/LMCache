# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase, SegmentTokenDatabase

# Local
from .utils import dumb_metadata, dumb_metadata_with_model_name, generate_tokens


def hf_credentials_available() -> bool:
    token_env = os.getenv("HF_TOKEN")
    hf_home = os.getenv("HF_HOME")
    default_token_file = os.path.expanduser("~/.cache/huggingface/token")
    token_file = os.path.join(hf_home, "token") if hf_home else ""
    return bool(
        token_env or os.path.exists(default_token_file) or os.path.exists(token_file)
    )


@pytest.mark.parametrize("chunk_length", [16, 64, 256])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_chunked_token_database(chunk_length, save_unfull_chunk):
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_length, backend="cpu", save_unfull_chunk=save_unfull_chunk
    )
    metadata = dumb_metadata()

    test_length = 2500
    tokens = generate_tokens(test_length, "cpu")
    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")

    num_falses = [i * chunk_length for i in range(0, test_length // chunk_length)]

    db = ChunkedTokenDatabase(cfg, metadata)

    # Process without mask
    original_results = list(db.process_tokens(tokens=tokens))
    end = (
        test_length if save_unfull_chunk else (test_length - test_length % chunk_length)
    )
    for i in range(0, end, chunk_length):
        st, ed, key = original_results[i // chunk_length]
        assert st == i
        if save_unfull_chunk:
            assert ed == min(i + chunk_length, test_length)
        else:
            assert ed == i + chunk_length

    for i in range(0, test_length // chunk_length):
        mask[: num_falses[i]] = False
        new_results = list(db.process_tokens(tokens=tokens, mask=mask))
        assert len(new_results) == len(original_results) - i

        for j in range(len(new_results)):
            st, ed, key = new_results[j]
            assert st == original_results[j + i][0]
            assert ed == original_results[j + i][1]


@pytest.mark.parametrize("prefix_length", [0, 16, 64, 256])
@pytest.mark.parametrize("chunk_lengths", [[256, 512, 256], [1024, 512, 256]])
@pytest.mark.skipif(
    not hf_credentials_available(), reason="No Hugging Face credentials found"
)
def test_segment_token_database(prefix_length, chunk_lengths):
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ")
    metadata = dumb_metadata_with_model_name("facebook/opt-125m")

    db = SegmentTokenDatabase(cfg, metadata)
    sep_tokens = db.sep_tokens

    sys_length = 25
    query_length = 50
    sys_tokens = generate_tokens(sys_length, "cpu", fixed=True)
    query_tokens = generate_tokens(query_length, "cpu", fixed=True)

    token_chunks = []
    starts = [0]
    ends = [sys_length]
    sys_tuple = tuple(sys_tokens.cpu().tolist())
    sys_hash = hash((None, sys_tuple, None))
    hashes = [sys_hash]
    start = sys_length + len(sep_tokens)
    for idx, chunk_length in enumerate(chunk_lengths):
        token_chunk = generate_tokens(chunk_length, "cpu", fixed=True)

        token_tuple = tuple(token_chunk.cpu().tolist())
        token_hash = hash((None, token_tuple, None))
        hashes.append(token_hash)

        token_chunk = torch.cat([sep_tokens, token_chunk])
        token_chunks.append(token_chunk)
        starts.append(start)
        ends.append(start + chunk_length)
        start += chunk_length + len(sep_tokens)

    query_tuple = tuple(query_tokens.cpu().tolist())
    query_hash = hash((None, query_tuple, None))
    hashes.append(query_hash)
    starts.append(start)
    ends.append(start + query_length)

    tokens = torch.cat([sys_tokens, *token_chunks, sep_tokens, query_tokens])
    total_length = len(tokens)
    mask = torch.full([total_length], True, dtype=torch.bool, device="cpu")
    mask[:prefix_length] = False

    chunk_lists = [sys_tokens, *token_chunks, sep_tokens, query_tokens]
    skip_chunk_num = 0
    cum_length = 0
    for chunk in chunk_lists:
        if prefix_length > cum_length:
            skip_chunk_num += 1
        cum_length += len(chunk)

    starts = starts[skip_chunk_num:]
    ends = ends[skip_chunk_num:]
    hashes = hashes[skip_chunk_num:]

    original_results = list(db.process_tokens(tokens=tokens, mask=mask))
    for i in range(len(original_results)):
        st, ed, key = original_results[i]
        assert st == starts[i]
        assert ed == ends[i]
        assert key.chunk_hash == hashes[i]
        # print(st, starts[i])
        # print(ed, ends[i])


@pytest.mark.parametrize(
    "model_name", ["facebook/opt-125m", "Qwen/Qwen2.5-0.5B-Instruct"]
)
@pytest.mark.parametrize("blend_special_str", [" # # ", "# #"])
@pytest.mark.skipif(
    not hf_credentials_available(), reason="No Hugging Face credentials found"
)
def test_segment_token_database_sep_tokens_match_real_text(
    model_name, blend_special_str
):
    """Regression test: ``SegmentTokenDatabase.sep_tokens`` must equal the
    BPE token sequence produced when ``blend_special_str`` appears inside
    a real prompt. The prior ``tokenizer.encode(blend_special_str)[1:]``
    produced start-of-string tokens that never appear mid-text on tokenizers
    with default-BOS prepending (Llama-2/3, Mistral, OPT) — silently
    breaking CacheBlend retrieval to 0% on those models. Symptom: zero
    segments detected → entire prompt stored as one chunk.

    The fix encodes ``" " + blend_special_str.strip()`` with
    ``add_special_tokens=False``: stripping surrounding whitespace
    normalizes the common-but-fragile production setting ``" # # "``,
    the leading space anchors the same BPE merge that occurs at every
    mid-text occurrence, and ``add_special_tokens=False`` prevents BOS
    being prepended in the first place.
    """
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=blend_special_str)
    metadata = dumb_metadata_with_model_name(model_name)
    db = SegmentTokenDatabase(cfg, metadata)
    sep_tokens = db.sep_tokens.tolist()

    # Production-shape user_content: passages joined by " # # " (matches the
    # real CacheBlend RAG pattern). The configured blend_special_str may
    # vary in whitespace (" # # " vs "# #"), but the literal text in mid-
    # prompt is always the space-padded form — which the fix must handle.
    sep_in_text = " # # "
    user_content = (
        "Reference passages:"
        + sep_in_text
        + sep_in_text.join(
            [
                "[Passage A]\nFirst passage text.",
                "[Passage B]\nSecond passage text.",
                "[Passage C]\nThird passage text.",
            ]
        )
        + sep_in_text
        + "Question: q?"
    )
    prompt_ids = db.tokenizer.encode(user_content, add_special_tokens=False)
    n_seps_in_text = user_content.count(sep_in_text)

    n = len(sep_tokens)
    matches = sum(
        1
        for i in range(len(prompt_ids) - n + 1)
        if prompt_ids[i : i + n] == sep_tokens
    )
    assert matches == n_seps_in_text, (
        f"{model_name} blend_special_str={blend_special_str!r}: "
        f"sep_tokens {sep_tokens} matched {matches} times in user_content "
        f"(expected {n_seps_in_text}). prompt_ids[:30]={prompt_ids[:30]}"
    )
