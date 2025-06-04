import abc
import hashlib
from typing import Iterable, Optional, Tuple
import csv
import torch
from lmcache.logging import init_logger
from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.utils import CacheEngineKey, CacheManagerMetadata
import os
import pandas as pd

logger = init_logger(__name__)

class TokenDatabase(metaclass=abc.ABCMeta):
    """TokenDatabase is used to convert input tokens into list of
    cache engine keys. There are multiple ways to implement this:

    - ChunkedTokenDatabase: It processes tokens into chunks and convert 
    each chunk into a cache engine key using prefix hash.

    - RadixTokenDatabase: more advanced implementation using radix tree.
    """

    @abc.abstractmethod
    def process_tokens(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        id: Optional[int] = None,
    ) -> Iterable[Tuple[int, int, CacheEngineKey, int]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param torch.Tensor tokens: The tokens to process, in 1-D CPU tensor.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key for the tokens.
        """

        raise NotImplementedError
    
def hash_all_tokens(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest()

def load_linear_coefficients(csv_path: str) -> dict:
    """
    Reads the CSV file and returns a dictionary mapping each filename to its (a, b) coefficients.
    """
    coefficients = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coefficients[row["filename"]] = (float(row["a"]), float(row["b"]))
    return coefficients

def compute_score_table(
    threshold_map: dict[float, str],
    token_len: float,
    alpha: float,
    coefficients: dict[str, tuple[float, float]],
    score_lookup: dict[float, float],
) -> list[tuple[float, float]]:
    """
    Compute a list of (threshold, score) pairs.

    For each threshold:
    1. Load (a, b) from coefficients by filename.
    2. Compute base_value = a * token_len + b.
    3. Subtract weighted base_value and the preset score from 1.
    """
    table: list[tuple[float, float]] = []
    for threshold, filename in threshold_map.items():
        a, b = coefficients[filename]
        base_value = a * token_len + b
        preset_score = score_lookup.get(threshold, 0.0)
        final_score = 1.0 - base_value * alpha - preset_score
        table.append((threshold, final_score))
    return table

def streaming_compute_score_table(
    threshold_map: dict[float, str],
    token_len: float,
    alpha: float,
    coefficients: dict[str, tuple[float, float]],
    score_lookup: dict[float, float],
) -> list[tuple[float, float]]:
    """
    Compute a list of (threshold, score) pairs.

    For each threshold:
    1. Load (a, b) from coefficients by filename.
    2. Compute base_value = a * token_len + b.
    3. Subtract weighted base_value and the preset score from 1.
    """
    table: list[tuple[float, float]] = []
    for threshold, filename in threshold_map.items():
        a, b = coefficients[filename]
        if threshold != 1.0 and threshold != 0.0:
            base_value = a * token_len * threshold + b
        else:
            base_value = a * token_len + b
        preset_score = score_lookup.get(threshold, 0.0)
        final_score = 1.0 - base_value * alpha - preset_score
        table.append((threshold, final_score))
    return table

def compute_min_unit_quality_drop(
    score_table: list[tuple[float, float]],
    token_len: float,
) -> float:
    base_threshold, base_score = score_table[0]
    min_unit_drop = float("inf")

    for threshold, score in score_table[1:]:
        # compute the per-unit quality drop
        numerator = base_score - score
        denominator = token_len * (base_threshold - threshold)
        unit_drop = numerator / denominator

        if unit_drop < min_unit_drop:
            min_unit_drop = unit_drop

    return min_unit_drop

def choose_score_dict(
    dataset: str,
    compression_method: str,
) -> dict[float, float]:
    if dataset == "qmsum" and compression_method == "kivi":
        score02 = 0.5056
        score03 = 0.7473
        score06 = 0.8599
        score1 = 0.8780
    elif dataset == "qmsum" and compression_method == "streaming":
        score02 = 0.3712
        score03 = 0.3989
        score06 = 0.4629
        score1 = 0.8780
    elif dataset == "samsum" and compression_method == "kivi":
        score02 = 0.7313
        score03 = 0.9095
        score06 = 0.9565
        score1 = 0.9567
    elif dataset == "samsum" and compression_method == "streaming":
        score02 = 0.6309
        score03 = 0.6434
        score06 = 0.6993
        score1 = 0.9567
    else:
        raise ValueError(f"Unsupported dataset {dataset} and compression method {compression_method}.")
    return {
        1.0: 1 - score1,
        0.728571429: 1 - score06,
        0.485714286: 1 - score03,
        0.371428571: 1 - score02,
        0.0:  0.0,
    }

class ChunkedTokenDatabase(TokenDatabase):

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata):
        self.chunk_size = config.chunk_size
        self.metadata = metadata
        # Load the coefficients once during initialization.
        self.coefficients = load_linear_coefficients("linear_coefficients3.csv")
        self.alpha = config.alpha
        self.compression = config.compression
        logger.info(f"ChunkedTokenDatabase initialized with alpha {self.alpha}.")
        self._dataset_df = pd.read_csv(config.dataset_csv)
        self.method_output_csv = config.method_output_csv

    def _make_key_by_hash(self, chunk_hash: str, total_hashes: str, token_len: int, id: int) -> Tuple[CacheEngineKey, int]:
        dataset_value = self._dataset_df.iloc[id]["dataset"]
        occurrence = self._dataset_df.iloc[id]["occurrence_number"]
        if self.compression == "kivi":
            threshold_file_mapping = {
                1.0:   "cpu_1.csv",
                0.728571429: "cpu_06.csv",
                0.485714286: "cpu_03.csv",
                0.371428571: "cpu_02.csv",
                0.0:   "prefill.csv",
            }
            disk_threshold_file_mapping = {
                1.0:   "1.csv",
                0.728571429: "06.csv",
                0.485714286: "03.csv",
                0.371428571: "02.csv",
            }
            score_dict = choose_score_dict(
                dataset_value, "kivi"
            )
            score_table = compute_score_table(
                threshold_file_mapping, token_len, self.alpha,
                self.coefficients, score_dict
            )
            disk_score_table = compute_score_table(
                disk_threshold_file_mapping, token_len, self.alpha,
                self.coefficients, score_dict
            )
        elif self.compression == "streaming":
            threshold_file_mapping = {
                1.0:   "cpu_1.csv",
                0.728571429: "cpu_1.csv",
                0.485714286: "cpu_1.csv",
                0.371428571: "cpu_1.csv",
                0.0:   "prefill.csv",
            }
            disk_threshold_file_mapping = {
                1.0:   "1.csv",
                0.728571429: "1.csv",
                0.485714286: "1.csv",
                0.371428571: "1.csv",
            }
            score_dict = choose_score_dict(
                dataset_value, "streaming"
            )
            score_table = streaming_compute_score_table(
                threshold_file_mapping, token_len, self.alpha,
                self.coefficients, score_dict
            )
            disk_score_table = streaming_compute_score_table(
                disk_threshold_file_mapping, token_len, self.alpha,
                self.coefficients, score_dict
            )
        elif self.compression == "mix":
            kivi_threshold_file_mapping = {
                1.0:   "cpu_1.csv",
                0.728571429: "cpu_06.csv",
                0.485714286: "cpu_03.csv",
                0.371428571: "cpu_02.csv",
                0.0:   "prefill.csv",
            }
            streaming_threshold_file_mapping = {
                1.0:   "cpu_1.csv",
                0.728571429: "cpu_1.csv",
                0.485714286: "cpu_1.csv",
                0.371428571: "cpu_1.csv",
                0.0:   "prefill.csv",
            }
            kivi_score_dict = choose_score_dict(
                dataset_value, "kivi"
            )
            streaming_score_dict = choose_score_dict(
                dataset_value, "streaming"
            )
            kivi_score_table = compute_score_table(
                kivi_threshold_file_mapping, token_len, self.alpha,
                self.coefficients, kivi_score_dict
            )
            streaming_score_table = streaming_compute_score_table(
                streaming_threshold_file_mapping, token_len, self.alpha,
                self.coefficients, streaming_score_dict
            )
            kivi_min_unit_quality_drop = compute_min_unit_quality_drop(kivi_score_table, token_len)
            streaming_min_unit_quality_drop = compute_min_unit_quality_drop(streaming_score_table, token_len)
            if kivi_min_unit_quality_drop <= streaming_min_unit_quality_drop:
                mode = "kivi"
                score_table = kivi_score_table
                disk_threshold_file_mapping = {
                    1.0:   "1.csv",
                    0.728571429: "06.csv",
                    0.485714286: "03.csv",
                    0.371428571: "02.csv",
                }
                disk_score_table = compute_score_table(
                    disk_threshold_file_mapping, token_len, self.alpha,
                    self.coefficients, kivi_score_dict
                )
            else:
                mode = "streaming"
                score_table = streaming_score_table
                disk_threshold_file_mapping = {
                    1.0:   "1.csv",
                    0.728571429: "1.csv",
                    0.485714286: "1.csv",
                    0.371428571: "1.csv",
                }
                disk_score_table = streaming_compute_score_table(
                    disk_threshold_file_mapping, token_len, self.alpha,
                    self.coefficients, streaming_score_dict
                )
            
            log_file = self.method_output_csv
            file_exists = os.path.exists(log_file)
            if file_exists:
                with open(log_file, "r") as f:
                    total_lines = sum(1 for _ in f) - 1
                    if total_lines < 0:
                        total_lines = 0
            else:
                total_lines = 0
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["index", "method"])
                writer.writerow([total_lines, mode])

        return CacheEngineKey(self.metadata.fmt, self.metadata.model_name,
                              self.metadata.world_size,
                              self.metadata.worker_id, chunk_hash,
                              CacheManagerMetadata([total_hashes], ["kivi"], 1, 0.0, token_len, [score_table], [], [disk_score_table])), occurrence

    def _get_init_hash(self) -> str:
        return ""

    def _hash(
        self,
        tokens: torch.Tensor,
        prefix_hash: str,
    ) -> str:
        # TODO: change it to a more efficient hash function
        return hashlib.sha256(
            prefix_hash.encode("ascii") +
            tokens.cpu().numpy().tobytes()).hexdigest()

    def _chunk_tokens(
        self,
        tokens: torch.Tensor,
    ) -> Iterable[torch.Tensor]:
        """
        Chunk the tokens into chunks of size self.chunk_size.

        :param tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        :return: a generator of chunks of tokens, each with 
                shape [chunk_size]
        """
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i:i + self.chunk_size]

    def _prefix_hash(
        self,
        token_chunks: Iterable[torch.Tensor],
    ) -> Iterable[str]:
        prefix_hash = self._get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            yield prefix_hash

    def process_tokens(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        id: Optional[int] = None,
    ) -> Iterable[Tuple[int, int, CacheEngineKey, int]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param torch.Tensor tokens: The tokens to process, in 1-D CPU tensor.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a 
            multiple of the chunk size.
        """
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum()
        else:
            num_falses = 0

        if num_falses % self.chunk_size != 0:
            raise ValueError("The number of Falses in the mask is not a "
                             "multiple of the chunk size.")
        total_len = len(tokens)

        token_chunks = self._chunk_tokens(tokens)
        prefix_hashes = self._prefix_hash(token_chunks)

        total_hashes = hash_all_tokens(tokens)

        start_idx = 0
        for chunk_id, hash_val in enumerate(prefix_hashes):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_len)
            if start_idx < num_falses:
                continue
            else:
                key, occurrence = self._make_key_by_hash(
                    hash_val, total_hashes, total_len, id)
                yield start_idx, end_idx, key, occurrence
