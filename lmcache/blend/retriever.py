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
        the tensor L2HTD or L2THD
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
        Input tensor is L2THD or L2HTD depending on fmt
        Output tensor is K and V with shape LTH or LHT depending on fmt
        Could also be None, None if nothing is retrieved
        """
        if real_length == expected_length:
            return input_tensor[:, 0, ...], input_tensor[:, 1, ...]

        if real_length == 0:
            return None, None

        ret_shape = list(input_tensor.shape)
        match fmt:
            case "vllm": 
                ret_shape[2] = expected_length
            case "huggingface": 
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

        num_layers = None
        num_heads = None
        head_size = None
        dtype = None
        device = None
        def update_shape(kv, fmt):
            nonlocal num_layers, num_heads, head_size, dtype, device
            num_layers = kv.shape[0]
            head_size = kv.shape[-1]
            num_heads = kv.shape[3] if fmt == "vllm" else kv.shape[2]
            dtype = kv.dtype
            device = kv.device

        for token_segment, task in zip(self.token_segments, self.tasks):
            kv, length = task.result()
            if length > 0:
                update_shape(kv, self.fmt)

            k, v = self._PrepareOutputTensor(
                    self.fmt,
                    kv,
                    length,
                    len(token_segment))

            valid_mask = torch.zeros(len(token_segment), 
                                     dtype = torch.int, 
                                     device = "cpu")
            valid_mask[:length] = 1
                    
            keys.append(k)
            values.append(v)
            valid_masks.append(valid_mask)

        # return if nothing is retrieved
        if num_layers is None:
            return

        match self.fmt:
            case "vllm": 
                token_dim = 1
                shape_placeholder = [num_layers, 0, num_heads, head_size]
            case "huggingface":
                token_dim = 2
                shape_placeholder = [num_layers, num_heads, 0, head_size]
            case _:
                raise ValueError(f"Unknown KV format {self.fmt}")

        # Update the shape of the None tensors
        for i, (k, v) in enumerate(zip(keys, values)):
            shape_placeholder[token_dim] = len(self.token_segments[i])
            if k is None:
                keys[i] = torch.empty(shape_placeholder,
                                      dtype = dtype, device = device) 
            if v is None:
                values[i] = torch.empty(shape_placeholder,
                                        dtype = dtype, device = device)

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
        if self.valid_mask is None:
            self._wait_for_result()

        assert self.valid_mask is not None

        ret = BlenderRetrieverResult(
                k = self.rebuilt_key[layer_id] \
                        if self.rebuilt_key is not None else None,
                v = self.rebuilt_value[layer_id] \
                        if self.rebuilt_value is not None else None,
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

        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        start = 0
        splitted_tokens = []

        print("HERE indices is", indices, indices.shape)
        if len(indices.shape) == 0:
            breakpoint()
        #if indices[0] == 999 and len(indices == 1):
        #    breakpoint()

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
            the number of requests in the batch + 1.

        :return: The retriever task to retrieve the KV caches
        :rtype: BlendRetrieverTask
        """
        with ThreadPoolExecutor(max_workers = 1) as executor:
            splitted_tokens = []
            start_loc = query_start_loc[0]
            for loc in query_start_loc[1:]:
                print("HERE", start_loc, loc)
                print("SHAPE:", input_tokens.shape, input_tokens[start_loc:loc].shape)
                splitted_tokens.extend(
                        self._split_input_tokens(input_tokens[start_loc:loc]))
                start_loc = loc
                
            logger.debug("Split input tokens into %d requests", len(splitted_tokens))
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
