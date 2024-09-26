from typing import List, Tuple
import torch
from concurrent.futures import ThreadPoolExecutor, Future

from lmcache.blend.interfaces import BlendRetriever, BlendRetrieverTask, BlenderRetrieverResult
from lmcache.config import LMCacheEngineMetadata
from lmcache.cache_engine import LMCacheEngine
from lmcache.logging import init_logger

logger = init_logger(__name__)

class SPTBlendRetrieverTask(BlendRetrieverTask):
    def __init__(
            self, 
            token_segments: List[torch.Tensor],
            tasks: List[Future], 
            fmt: str
        ):
        """Initialize the SBT retriever task by the futures and corresponding token 
        segments.

        The result of tasks should be the Tuple[torch.Tensor, int] and the shape of
        the tensor L2HT or L2TH
        """
        assert len(token_segments) == len(tasks), \
                "The number of token segments and tasks should match."
        self.token_segments = token_segments
        self.tasks = tasks
        self.fmt = fmt 

        self.rebuilt_key = None
        self.rebuilt_value = None
        self.valid_mask = None

    @staticmethod
    def _PrepareOutputTensor(
            fmt: str, 
            input_tensor: torch.Tensor,
            real_length: int,
            expected_length: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input tensor is L2TH or L2HT depending on fmt
        Output tensor is K and V with shape LTH or LHT depending on fmt
        """
        if real_length == expected_length:
            return input_tensor[:, 0, ...], input_tensor[:, 1, ...]

        ret_shape = input_tensor.shape
        match fmt:
            case "vllm": 
                ret_shape[2] = expected_length
                ret_shape[3] = expected_length
            case _:
                raise ValueError(f"Unknown KV format {fmt}")

        ret_tensor = torch.empty(ret_shape, 
                              dtype = input_tensor.dtype, 
                              device = input_tensor.device)
        
        match fmt:
            case "vllm": 
                ret_tensor[:, :, :real_length, ...] = input_tensor
            case "huggingface":
                ret_tensor[:, :, :, :real_length, ...] = input_tensor
            case _:
                raise ValueError(f"Unknown KV format {fmt}")

        return ret_tensor[:, 0, ...], ret_tensor[:, 1, ...]

    def _wait_for_result(self):
        """Wait for the results of the tasks and rebuild the K and V tensors.
        """
        keys = []
        values = []
        valid_masks = []
        for token_segment, task in zip(self.token_segments, self.tasks):
            kv, length = task.result()
            k, v = self._PrepareOutputTensor(
                    self.fmt,
                    kv,
                    length,
                    len(token_segment))

            valid_mask = torch.zeros(len(token_segment), 
                                     dtype = torch.int, 
                                     device = kv.device)
            valid_mask[:length] = 1
                    
            keys.append(k)
            values.append(v)
            valid_masks.append(valid_mask)

        match self.fmt:
            case "vllm": 
                token_dim = 1
            case "huggingface":
                token_dim = 2
            case _:
                raise ValueError(f"Unknown KV format {self.fmt}")

        self.rebuilt_key = torch.cat(keys, dim = token_dim)
        self.rebuilt_value = torch.cat(values, dim = token_dim)
        self.valid_mask = torch.cat(valid_masks, dim = 0)

    def result(self, layer_id: int) -> BlenderRetrieverResult:
        """Blocking function to get a single layer of K and V tensor.
        The returned the K and V tensor should match the length of the input tokens
        passed to the `BlenderRetriever.new_request` function.

        :param int layer_id: the layer id 
        :return: Tuple of K and V tensor
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if self.rebuilt_key is None:
            self._wait_for_result()

        assert self.rebuilt_key is not None
        assert self.rebuilt_value is not None
        assert self.valid_mask is not None

        ret = BlenderRetrieverResult(
                k = self.rebuilt_key[layer_id],
                v = self.rebuilt_value[layer_id],
                valid_mask = self.valid_mask)
        return ret


class SPTBlendRetriever(BlendRetriever):
    """Implement the retrieval logic using "SPecial Token" (SPT) as delimiter.

    This implementation assumes that there MUST be a special token at the end of 
    the input text chunk.

    Example:
      Input = [x, x, x, spt, y, y, spt, z, z, z, z]
      Requests sent to LMCache engine:
        - [x, x, x, spt]
        - [y, y, spt]
        - [z, z, z, z]

    Therefore, to use this retriever, the text chunks are better to also be ended
    with the special token.
    """

    def __init__(
            self, 
            spt: torch.Tensor, 
            cache_engine: LMCacheEngine,
            metadata: LMCacheEngineMetadata,
        ):
        """Initialize the SPT retriever.

        :param torch.Tensor spt: The special token to use as delimiter
        :param LMCacheEngine cache_engine: The cache engine to retrieve the KV caches
        :param LMCacheEngineMetadata metadata: The metadata of the cache engine
        """
        self.spt = spt
        self.cache_engine = cache_engine
        self.metadata = metadata

    def _split_input_tokens(self, input_tokens_single_query: torch.Tensor):
        """Split the input tokens into multiple requests based on the ROI.

        Returns a list of splitted tokens for cache_engine to retrieve
        """
        spt_len = len(self.spt)
        if spt_len == 1:
            indices = (input_tokens_single_query == self.spt).nonzero().squeeze()
        else:
            windows = input_tokens_single_query.unfold(0, spt_len, 1)
            indices = (windows == self.spt).all(dim=1).nonzero().squeeze()

        start = 0
        splitted_tokens = []

        for i in indices:
            splitted_tokens.append(input_tokens_single_query[start:i+spt_len])
            start = i + spt_len

        if start < len(input_tokens_single_query):
            splitted_tokens.append(input_tokens_single_query[start:])
        return splitted_tokens

    def new_request(
            self, 
            input_tokens: torch.Tensor, 
            query_start_loc: torch.Tensor,
        ) -> BlendRetrieverTask:
        """Create a new BlendRetrieverTask to retrieve the KV caches.
        It may launch async tasks in the background during the retrieval.

        :param torch.Tensor input_tokens: The input tokens, could include
            multiple requests in a batch
        :param torch.Tensor query_start_loc: The start location of the query if
            input_tokens has multiple requests in a batch. The length should be 
            the number of requests in the batch.

        :return: The retriever task to retrieve the KV caches
        :rtype: BlendRetrieverTask
        """
        with ThreadPoolExecutor(max_workers = 1) as executor:
            splitted_tokens = self._split_input_tokens(input_tokens)
            logger.debug("Split input tokens into %d requests", len(splitted_tokens))
            print(splitted_tokens)
            tasks = [executor.submit(
                        self.cache_engine.retrive,
                        tokens, False
                    ) for tokens in splitted_tokens]
        
        return SPTBlendRetrieverTask(
                token_segments = splitted_tokens,
                tasks = tasks,
                fmt = self.metadata.fmt
            )

#if __name__ == "__main__":
    #tokens = torch.tensor([1, 2, 3, 4, 5, 2, 6, 7, 8, 2, 3, 9, 10])
    #r1 = SPTBlendRetriever(torch.tensor([2]), None)
    #r2 = SPTBlendRetriever(torch.tensor([2, 3]), None)

    #print(r1._split_input_tokens(tokens))
    #print(r2._split_input_tokens(tokens))

    #vllm_kv = torch.zeros([32, 2, 2000, 128])
    #hf_kv = torch.zeros([32, 2, 128, 2000])

    #k, v = SPTBlendRetrieverTask._PrepareOutputTensor("vllm", vllm_kv, 2000, 3000)
    #print(k.shape)
    #print(v.shape)

    #k, v = SPTBlendRetrieverTask._PrepareOutputTensor("huggingface", hf_kv, 2000, 3000)
    #print(k.shape)
    #print(v.shape)
    pass
