import os
import io
import time
import pickle
import torchac
import torchac_cuda
import numpy as np
import torch
from multiprocessing import Pool
from dataclasses import dataclass

def torch_quant(bins, qA):
    # shape (8, 2048)
    MAX = bins // 2 - 1
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
    xq = torch.round(qA * (C / max1)).to(torch.int8)
    
    x = (xq / C * max1).to(torch.float32)
    
    return xq, max1

def concat_dict(dict1, start, end):
    concat_tensor = None
    for i in range(start, end):
        if concat_tensor is None:
            concat_tensor = dict1[i].unsqueeze(0)
        else:
            concat_tensor = torch.cat((concat_tensor, \
                dict1[i].unsqueeze(0)), dim=0)
    return concat_tensor

def concat_max(max1):
    """
    Given a dict of max tensors, concatenate them into a single tensor
    """
    maxes = []
    for i in range(len(max1)):
        maxes.append(max1[i].unsqueeze(0))
    return torch.cat(maxes, dim=0)

def _renorm_cast_cdf_(cdf, precision):
    """ The cdf normalization function in torchac
    """
    Lp = cdf.shape[-1]
    finals = 1  # NHW1
    # RENORMALIZATION_FACTOR in cuda
    f = torch.tensor(2, dtype=torch.float32, device=cdf.device).pow_(precision)
    cdf = cdf.mul((f - (Lp - 1)) / finals)  # TODO
    cdf = cdf.round()
    cdf = cdf.to(dtype=torch.int16, non_blocking=True)
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
    return cdf

@dataclass
class CacheGenConfig:
    key_first_layers: int
    key_second_layers: int
    key_third_layers: int
    key_first_bins: int
    key_second_bins: int
    key_third_bins: int
    value_first_layers: int
    value_first_bins: int
    value_second_bins: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @staticmethod
    def from_model_name(model_name: str) -> "CacheGenConfig":
        family_7b = ["mistralai/Mistral-7B-Instruct-v0.2", "lmsys/longchat-7b-16k"]

        if model_name in family_7b:
            return CacheGenConfig(
                key_first_layers=10,
                key_second_layers=20,
                key_third_layers=32,
                key_first_bins=32,
                key_second_bins=16,
                key_third_bins=16,
                value_first_layers=2,
                value_first_bins=32,
                value_second_bins=16
            )
        else:
            raise ValueError(f"Model {model_name} is not supported")


class CacheGenEncoderImpl:
    def __init__(self, **kwargs) -> None:
        """ 
        Fields: 
        - fp_kv: should be a tensor of shape (num_layers, num_tokens, num_channels)
        - fp_v: should be a tensor of shape (num_layers, num_tokens, num_channels)
        """
        self.fp_k = kwargs["fp_k"]
        self.fp_v = kwargs["fp_v"]
        
        self.quantized_key = {}
        self.max_tensors_key = {}  
        self.quantized_value = {}
        self.max_tensors_value = {} 
        self.config = kwargs["config"]
        
    def quantize(self):
        """ Quantize the key and value tensors 
        (self.fp_k and self.fp_v) 
        """
        for layer in range(len(self.fp_k)):
            if layer < self.config["key_first_layers"]:
                bins = self.config["key_first_bins"]
            elif layer < self.config["key_second_layers"]:
                bins = self.config["key_second_bins"]
            else:
                bins = self.config["key_third_bins"]

            tmp = torch_quant(bins, self.fp_k[layer].float())
            self.quantized_key[layer] = tmp[0] + bins // 2 - 1
            self.max_tensors_key[layer] = tmp[1]

        for layer in range(len(self.fp_v)):
            if layer < self.config["value_first_layers"]:
                bins = self.config["value_first_bins"]
            else:
                bins = self.config["value_second_bins"]
            tmp = torch_quant(bins, self.fp_v[layer].float())
            self.quantized_value[layer] = tmp[0]+ bins // 2 - 1
            self.max_tensors_value[layer] = tmp[1]
            
    def compute_cdf(self, is_key):
        """
        Compute the CDF based on the quantized tensors
        Field: 
        - start_layer: the start layer to compute the CDF
        - end_layer: the end layer to compute the CDF
        """
        # TODO: Add start_index here
        channels = self.fp_k[0].shape[-1]
        tokens = self.fp_k[0].shape[0]
        
        def process_batch(X, max_val):
            """
            input shape should be [channels, tokens]
            """
            nchannels, ntokens = X.shape
            one_hot = torch.nn.functional.one_hot(X.long(), num_classes=max_val + 1).to(torch.float32)  # Use float32 to avoid integer overflow
            counts = one_hot.sum(dim=1) / ntokens
            ret = torch.cumsum(counts, dim=1).roll(1)
            ret[:, 0] = 0
            return ret

        def process_layers(X, max_val):
            """
            x is a iterator of dict values
            each element's shape is [tokens, channels]
            """
            results = []
            for x in X:
                ''' do permute here '''
                batch_counts = process_batch(x.cuda().permute(1, 0), max_val)
                results.append(batch_counts)

            final_counts = torch.cat(results, dim=0)
            
            return final_counts
        
        if is_key:
            X = self.quantized_key.values()
        else:
            X = self.quantized_value.values()
        value_range = 32
        cdfs = process_layers(X, value_range) # 4096 is batch size, ==> 18GB GPU memory
        final_cdf = cdfs.reshape((len(self.fp_k), channels, value_range+1)).cpu()
                
        return final_cdf

@dataclass
class CacheGenEncoderOutput:
    bytestream: bytes
    start_indices: torch.Tensor
    cdf: torch.Tensor
    max_tensors_key: torch.Tensor
    max_tensors_value: torch.Tensor

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    def to_bytes(self) -> bytes:
        """ Save the output to a file """
        with io.BytesIO() as f:
            #torch.save(self, f)
            pickle.dump(self, f)
            return f.getvalue()

    @staticmethod
    def from_bytes(bs: bytes) -> "CacheGenEncoderOutput":
        with io.BytesIO(bs) as f:
            return pickle.load(f)

    @staticmethod
    def serialize_field(field) -> bytes:
        with io.BytesIO() as f:
            #torch.save(field, f)
            pickle.dump(field, f)
            return f.getvalue()

    def check_len(self):
        print("Bytestream length:", len(self.serialize_field(self.bytestream)))
        print("start indices length:", len(self.serialize_field(self.start_indices)))
        print("cdf length:", len(self.serialize_field(self.cdf)))
        print("max tensors k length:", len(self.serialize_field(self.max_tensors_key)))
        print("max tensors v length:", len(self.serialize_field(self.max_tensors_value)))


def _split_kv(tensor: torch.Tensor) -> torch.Tensor:
    """
    Split a blob KV tensor to K and V tensors with the merged heads

    Input:
        tensor: the KV tensor with shape [num_layers, 2, num_tokens, num_heads, head_size]

    Returns:
        K and V tensors with shape [num_layers, num_tokens, num_channels]
    """
    num_layers, _, num_tokens, num_heads, head_size = tensor.shape
    return torch.unbind(tensor.reshape(num_layers, 2, num_tokens, num_heads * head_size), dim=1)

def single_token_encode(cdf, tensor) -> bytes:
    """
    Encode a single token with the given CDF
    """
    return torchac.encode_float_cdf(cdf, tensor)


def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    """Convert floatingpoint CDF to integers. See README for more info.
  
    The idea is the following:
    When we get the cdf here, it is (assumed to be) between 0 and 1, i.e,
      cdf \in [0, 1)
    (note that 1 should not be included.)
    We now want to convert this to int16 but make sure we do not get
    the same value twice, as this would break the arithmetic coder
    (you need a strictly monotonically increasing function).
    So, if needs_normalization==True, we multiply the input CDF
    with 2**16 - (Lp - 1). This means that now,
      cdf \in [0, 2**16 - (Lp - 1)].
    Then, in a final step, we add an arange(Lp), which is just a line with
    slope one. This ensure that for sure, we will get unique, strictly
    monotonically increasing CDFs, which are \in [0, 2**16)
    """
    PRECISION = 16
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(
      2, dtype=torch.float32, device=cdf_float.device).pow_(PRECISION)
    new_max_value = factor
    if needs_normalization:
      new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
      r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
      cdf.add_(r)
    return cdf

def encode_function(kv, model_name, chunk_size) -> CacheGenEncoderOutput:
    """
    Given the path to the original key value cache, encode the KV cache
    """
    output_dict = {}
    config = CacheGenConfig.from_model_name(model_name)
    fp_k, fp_v = _split_kv(kv)
    l = fp_k.shape[0]
    encoder = CacheGenEncoderImpl(fp_k=fp_k, fp_v=fp_v, config=config)
    encoder.quantize()
    cdf_k = encoder.compute_cdf(is_key=True)
    encode_input_key = torch.stack(list(encoder.quantized_key.values()))
    
    cdf_v = encoder.compute_cdf(is_key=False)
    encode_input_value = torch.stack(list(encoder.quantized_value.values()))
    cdf = torch.cat((cdf_k, cdf_v), dim=0)
    encode_input = torch.cat((encode_input_key, encode_input_value), dim=0).cpu()
    current_index = 0
    start_indices = []
    bytestreams = []
    cdf_int = _convert_to_int_and_normalize(cdf, True)
    for l in range(cdf.shape[0]):
        for i in range(chunk_size):
            bits = torchac.encode_int16_normalized_cdf(
                    cdf_int[l:l+1],
                    encode_input[l:l+1, i].to(torch.int16))

            #bits = torchac.encode_float_cdf(cdf[l:l+1], \
            #    encode_input[l:l+1, i].to(torch.int16) )
            bytestreams.append(bits)
            length = len(bits)
            start_indices += [current_index]
            current_index += length

    output = CacheGenEncoderOutput(
        bytestream = b"".join(bytestreams),
        start_indices = torch.tensor(start_indices).int(),
        cdf = _renorm_cast_cdf_(cdf.float(), 16),
        max_tensors_key = concat_max(encoder.max_tensors_key),
        max_tensors_value = concat_max(encoder.max_tensors_value)
    )
    return output

def quant(bins, xq, max1, dim=-1, quant_type="vector"):

    C = bins // 2 - 1
    x = (xq / C * max1).to(torch.float16)
    return x


def decode_function(cdf, bits, start_indices, max_tensors_k, max_tensors_v, quantization_config, chunk_size, output, chunk_id):
    # TODO: dtype and shape -- still have 128 and 8
    """
    Given the path to the encoded key value cache, decode the KV cache
    Fields:
    - cdf: the cdf tensor (used to encode/decode the KV)

    - path_to_encoded_kv: the path to the encoded key value cache
    - quantization_config: the path to the quantization config
    - model_config: the path to the model config
    - chunk_size: the chunk size to decode, NEEDS to be multiples of 20!!!
    Outputs:
    - key: the decoded key tensor in the shape of (layers, num_heads, tokens, heads_dim)
    """
    config = quantization_config
    start_time = time.monotonic()
    concated_string = bits
    nlayers = cdf.shape[0]
    kernel_start = time.monotonic()
    start_indices = torch.tensor(start_indices).int().cuda()

    num_threads = chunk_size
    num_blocks = nlayers
    scale = 1

    torchac_cuda.decode_fast(output,
                            cdf.unsqueeze(0),
                            concated_string,
                            start_indices,
                            chunk_size,
                            num_blocks,
                            num_threads,
                            scale)

    print("kernel computation time: ", time.monotonic() - kernel_start)
    out = output.reshape((2, max_tensors_k.shape[0], chunk_size, 1024))
    key = out[0].half()
    value = out[1].half()
    max_tensors_k = max_tensors_k.cuda()
    max_tensors_v = max_tensors_v.cuda()
    for l in range(key.shape[0]):
        if l < config["key_first_layers"]:
            bins = config["key_first_bins"]
        elif l < config["key_second_layers"]:
            bins = config["key_second_bins"]
        else:
            bins = config["key_third_bins"]
        key[l] = quant(bins, key[l] - (bins // 2 - 1), max_tensors_k[l, chunk_id * chunk_size: (chunk_id + 1) * chunk_size])

    for l in range(value.shape[0]):
        if l < config["value_first_layers"]:
            bins = config["value_first_bins"]
        else:
            bins = config["value_second_bins"]
        value[l] = quant(bins, value[l] - (bins // 2 - 1), max_tensors_v[l, chunk_id * chunk_size: (chunk_id + 1) * chunk_size])
    key = key.reshape(
        key.shape[0],
        key.shape[1],
        8,
        128)
    value = value.reshape(
        value.shape[0],
        value.shape[1],
        8,
        128)
    #kv_tuple = transformer_kv_to_tuple(key, value)
    #torch.cuda.synchronize()
    print("per iteration total time: ", time.monotonic() - start_time)
    return key, value

def decode_function_wrapper(encoded_output: CacheGenEncoderOutput, num_layers, num_channels, chunk_size, model_name):
    quantization_config = CacheGenConfig.from_model_name(model_name)
    cdf = encoded_output.cdf.to(torch.int16)
    if isinstance(encoded_output.bytestream, bytes):
        np_array = np.frombuffer(encoded_output.bytestream, dtype=np.uint8)
        inference_bits = torch.from_numpy(np_array)
    else:
        inference_bits = encoded_output.bytestream
    start_indices = encoded_output.start_indices
    if decode_function_wrapper.OUTPUT_BUFFER is None:
        decode_function_wrapper.OUTPUT_BUFFER = torch.zeros((chunk_size, 2 * num_layers * num_channels)).to(torch.int).cuda()
    #output = torch.zeros((chunk_size, 2 * num_layers * num_channels)).to(torch.int).cuda()
    output = decode_function_wrapper.OUTPUT_BUFFER
    return decode_function(cdf, inference_bits, 
                           start_indices,
                           encoded_output.max_tensors_key,
                           encoded_output.max_tensors_value,
                           quantization_config,
                           chunk_size,
                           output,
                           0)

decode_function_wrapper.OUTPUT_BUFFER = None
                            


def test_encode_performance():
    kv = torch.load("test.pt").cuda()
    print(kv.shape)
    chunk_size = 256
    nlayers = 32
    nchannels = 1024

    import cProfile
    profile = cProfile.Profile()
    profile.runctx("output_dict = encode_function(kv, 'mistralai/Mistral-7B-Instruct-v0.2', chunk_size)", globals(), locals())
    profile.dump_stats("encode.prof")

    st = time.perf_counter()
    output_dict = encode_function(kv, "mistralai/Mistral-7B-Instruct-v0.2", chunk_size)
    ed = time.perf_counter()
    print(output_dict.start_indices.shape, output_dict.start_indices.dtype)
    print(output_dict.cdf.shape, output_dict.cdf.dtype)
    print(output_dict.max_tensors_key.shape, output_dict.max_tensors_key.dtype)
    print(output_dict.max_tensors_value.shape, output_dict.max_tensors_value.dtype)
    print("Check len:", len(output_dict.bytestream))
    print("Total time:", ed - st)
    print("Total size:", len(output_dict.to_bytes()))
    output_dict.check_len()


def test_decode_performance():
    NITER = 100
    def test_func(output_dict, nlayers, nchannels, chunk_size, model_name):
        sums = []
        for i in range(NITER):
            k, v = decode_function_wrapper(output_dict, nlayers, nchannels, chunk_size, model_name)
            sums.append(k.mean())
        return sums

    kv = torch.load("test.pt").cpu()
    print(kv.shape)
    chunk_size = 256
    nlayers = 32
    nchannels = 1024
    output_dict = encode_function(kv, "mistralai/Mistral-7B-Instruct-v0.2", chunk_size)

    import cProfile
    profile = cProfile.Profile()
    profile.runctx("test_func(output_dict, nlayers, nchannels, chunk_size, 'mistralai/Mistral-7B-Instruct-v0.2')", globals(), locals())
    profile.dump_stats("decode.prof")

    st = time.perf_counter()
    test_func(output_dict, nlayers, nchannels, chunk_size, "mistralai/Mistral-7B-Instruct-v0.2")
    ed = time.perf_counter()
    print("Total time per iter:", (ed - st) / NITER)


if __name__ == "__main__":
    test_encode_performance()
    #test_decode_performance()

    #kv = torch.load("test.pt").cpu()
    #print(kv.shape)
    #chunk_size = 256
    #nlayers = 32
    #nchannels = 1024
    #st = time.perf_counter()
    #output_dict, encoder = encode_function(kv, "mistralai/Mistral-7B-Instruct-v0.2", chunk_size)
    #print(len(output_dict.to_bytes()))
    #ed = time.perf_counter()
    #print("Total time:", ed - st)

    ##pickle.dump(output_dict, open("ref2.pkl", "wb"))
    #ref_output = pickle.load(open("ref2.pkl", "rb"))

    #decoder_input = pickle.load(open("../CacheGen/encoded/0.pkl", "rb"))
    #ref_decoder_output = torch.load("../CacheGen/data/final_kv.pt")
    #breakpoint()
    #tmp_input = CacheGenEncoderOutput(
    #    bytestream = decoder_input["bitstreams"],
    #    start_indices = decoder_input["start_indices"],
    #    cdf = decoder_input["cdf"],
    #    max_tensors_key = decoder_input["max_tensors_key"],
    #    max_tensors_value = decoder_input["max_tensors_value"]
    #)
    #k, v = decode_function_wrapper(tmp_input, nlayers, nchannels, chunk_size, 'mistralai/Mistral-7B-Instruct-v0.2')
    #passed = True
    #for layer in range(nlayers):
    #    for token in range(chunk_size):
    #        kequal = torch.equal(k[layer][token].flatten(), ref_decoder_output[layer][0][0, :, token, :].flatten())
    #        vequal = torch.equal(v[layer][token].flatten(), ref_decoder_output[layer][1][0, :, token, :].flatten())
    #        print(f"Layer {layer}, Token {token}, Key Equal: {kequal}, Value Equal: {vequal}")
    #        if not kequal or not vequal:
    #            passed = False
    #            print("Failed, breaking")
    #            breakpoint()
    #            break
    #    if not passed:
    #        break
    #if passed:
    #    print("Passed!")
    #breakpoint()

    #import cProfile
    #profile = cProfile.Profile()
    #profile.runctx("test_func(output_dict, nlayers, nchannels, chunk_size, 'mistralai/Mistral-7B-Instruct-v0.2')", globals(), locals())
    #profile.dump_stats("decode.prof")

    #st = time.perf_counter()
    #test_func(output_dict, nlayers, nchannels, chunk_size, "mistralai/Mistral-7B-Instruct-v0.2")
    #ed = time.perf_counter()
    #print("Total time:", ed - st)

    #print("Checking bitstreams:", output_dict["bytestream"] == ref_output["bytestream"])
    #print("Checking start_indices:", torch.equal(output_dict["start_indices"], ref_output["start_indices"]))
    #print("Checking cdf:", torch.equal(output_dict["cdf"], ref_output["cdf"]))
    #print("Checking max_tensors_key:", torch.equal(output_dict["max_tensors_key"], ref_output["max_tensors_key"]))
    #print("Checking max_tensors_value:", torch.equal(output_dict["max_tensors_value"], ref_output["max_tensors_value"]))

    #breakpoint()
