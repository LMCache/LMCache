import pytest
import torch
from utils import dumb_metadata, generate_tokens

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.token_database import ChunkedTokenDatabase


@pytest.mark.parametrize('chunk_length', [16, 64, 256])
def test_chunked_token_database(chunk_length):
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_length,
                                          backend="cpu")
    metadata = dumb_metadata()

    test_length = 2500
    tokens = generate_tokens(test_length, "cpu")
    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")

    num_falses = [
        i * chunk_length for i in range(0, test_length // chunk_length)
    ]

    db = ChunkedTokenDatabase(cfg, metadata)

    # Process without mask
    original_results = list(db.process_tokens(tokens))
    for i in range(0, test_length, chunk_length):
        st, ed, key = original_results[i // chunk_length]
        assert st == i
        assert ed == min(i + chunk_length, test_length)

    for i in range(0, test_length // chunk_length):
        mask[:num_falses[i]] = False
        new_results = list(db.process_tokens(tokens, mask))
        assert len(new_results) == len(original_results) - i

        for j in range(len(new_results)):
            st, ed, key = new_results[j]
            assert st == original_results[j + i][0]
            assert ed == original_results[j + i][1]
