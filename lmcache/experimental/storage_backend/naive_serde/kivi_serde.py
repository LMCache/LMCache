from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj, MemoryFormat,
                                                    BytesBufferMemoryObj)
from lmcache.experimental.storage_backend.naive_serde.serde import (
    Deserializer, Serializer)
from lmcache.experimental.storage_backend.naive_serde.new_pack import triton_quantize_and_pack_along_last_dim
import io
import torch
import time
from lmcache.utils import _lmcache_nvtx_annotate

def unpack_tensor(v_code: torch.FloatTensor, 
				  bits: int, 
				  pack_dim: int):
	assert bits in [2,4,8]
	shape = v_code.shape
	feat_per_int = 32 // bits
	new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim+1:]
	unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
	i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
	j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
	num = 0xFF >> (8 - bits)
	packed_indices = [slice(None)] * len(new_shape)
	packed_indices[pack_dim] = i
	if pack_dim == 2:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)) & num
	elif pack_dim == 3:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
	else:
		raise NotImplementedError
	return unpacked_v_code
def unpack_and_dequant_kcache(k_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	pack_dim = 2
	assert bits in [2, 4, 8]
	assert len(k_code.shape) == 4
	data = unpack_tensor(k_code, bits, pack_dim=pack_dim)
	shape = data.shape
	num_groups = shape[pack_dim] // group_size
	data = data.view(shape[:pack_dim] + (num_groups, group_size,) + shape[pack_dim+1:])
	data = data.to(torch.float16)
	data = data * scale + mn 
	return data.view(shape)

def unpack_and_dequant_vcache(v_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	assert bits in [2, 4, 8]
	assert len(v_code.shape) == 4
	data = unpack_tensor(v_code, bits, pack_dim=3)
	shape = data.shape
	num_groups = shape[-1] // group_size
	data = data.view(shape[:-1] + (num_groups, group_size,))
	data = data.to(torch.float16)
	data = data * scale + mn 
	return data.view(shape)



class KIVISerializer(Serializer):

    def __init__(self, memory_allocator: MemoryAllocatorInterface):
        self.memory_allocator = memory_allocator
        self.residual = 128
        self.num_heads = 8
        self.head_size = 128
        self.group_size = 32
    def serialize(self, memory_obj: MemoryObj, bits: int = 4) -> MemoryObj:
        assert memory_obj.tensor is not None
        # NOTE(Shaoting): KIVI compression needs cuda
        t = memory_obj.tensor.cuda()
        compressed_dict = []
        for layer in range(t.shape[1]):
            k = t[0][layer].reshape((t[0][layer].shape[0], self.num_heads, self.head_size)).permute((1, 0, 2)).unsqueeze(0)
            v = t[1][layer].reshape((t[0][layer].shape[0], self.num_heads, self.head_size)).permute((1, 0, 2)).unsqueeze(0)
            rounded_len = (k.shape[-2] // self.residual) * self.residual
            if rounded_len > 0:
                rounded_k = k[..., :rounded_len, :]
                rounded_v = v[..., :rounded_len, :]
                quant_k_triton, k_scale_triton, k_min_triton = triton_quantize_and_pack_along_last_dim(rounded_k.transpose(2, 3).contiguous(), self.group_size, bits)
                quant_v_triton, v_scale_triton, v_min_triton = triton_quantize_and_pack_along_last_dim(rounded_v.contiguous(), self.group_size, bits)
                
                quant_k = quant_k_triton.permute(0, 1, 3, 2)
                k_scale  = k_scale_triton.permute(0, 1, 3, 2).reshape((k_scale_triton.shape[0], k_scale_triton.shape[1], k_scale_triton.shape[-1], 1, k_scale_triton.shape[2])) 
                k_min = k_min_triton.permute(0, 1, 3, 2).reshape((k_min_triton.shape[0], k_min_triton.shape[1], k_min_triton.shape[-1], 1, k_min_triton.shape[2]))
                quant_v = quant_v_triton
                v_scale  = v_scale_triton.reshape((v_scale_triton.shape[0], v_scale_triton.shape[1], v_scale_triton.shape[2], v_scale_triton.shape[3], 1))
                v_min = v_min_triton.reshape((v_scale_triton.shape[0], v_scale_triton.shape[1], v_scale_triton.shape[2], v_scale_triton.shape[3], 1))
                if k.shape[2] > rounded_len:
                    residual_k = k[..., rounded_len:, :]
                    residual_v = v[..., rounded_len:, :]
                else:
                    residual_k = None
                    residual_v = None
                
            else:
                quant_k, k_scale, k_min, quant_v, v_scale, v_min = None, None, None, None, None, None
                residual_k = k
                residual_v = v
            compressed_dict += [{
                    "quant_k": quant_k,
                    "k_scale": k_scale,
                    "k_min": k_min,
                    "quant_v": quant_v,
                    "v_scale": v_scale,
                    "v_min": v_min,
                    "residual_k": residual_k,
                    "residual_v": residual_v
                }]
        with io.BytesIO() as f:
            torch.save(compressed_dict, f)
            bytes_obj = BytesBufferMemoryObj(f.getvalue())
            return bytes_obj

class KIVIDeserializer(Deserializer):

    def __init__(self, memory_allocator: MemoryAllocatorInterface):
        self.memory_allocator = memory_allocator
        self.residual = 128
        self.group_size = 32
    @_lmcache_nvtx_annotate
    def deserialize(self, memory_obj: MemoryObj, bits: int = 4) -> MemoryObj:
        # TODO(Shaoting): Definitely need to speed up
        with io.BytesIO(memory_obj.byte_array) as f:
            compressed_dict = torch.load(f)
        ks = []
        vs = []
        
        for layer in range(len(compressed_dict)):
            quant_k = compressed_dict[layer]["quant_k"]
            k_scale = compressed_dict[layer]["k_scale"]
            k_min = compressed_dict[layer]["k_min"]
            quant_v = compressed_dict[layer]["quant_v"]
            v_scale = compressed_dict[layer]["v_scale"]
            v_min = compressed_dict[layer]["v_min"]
            residual_k = compressed_dict[layer]["residual_k"]
            residual_v = compressed_dict[layer]["residual_v"]
            if quant_k is not None:
                dequant_k = unpack_and_dequant_kcache(quant_k, k_scale, k_min, self.group_size, bits)
                dequant_v = unpack_and_dequant_vcache(quant_v, v_scale, v_min, self.group_size, bits)
                
                if residual_k is not None:
                    dequant_k = torch.cat((dequant_k, residual_k), dim=-2)
                    dequant_v = torch.cat((dequant_v, residual_v), dim=-2)
            else:
                dequant_k = residual_k
                dequant_v = residual_v
            
            # now dequant_k and dequant_v are of shape (1, num_heads, seq_len, heads_dim )
            ks += [dequant_k.permute(0, 2, 1, 3)]
            vs += [dequant_v.permute(0, 2, 1, 3)]
        # 
        ks = torch.cat(ks, dim=0).unsqueeze(0)
        vs = torch.cat(vs, dim=0).unsqueeze(0)
        blob = torch.cat((ks, vs), dim=0).to(torch.bfloat16)
        hidden_dim = blob.shape[-1] * blob.shape[-2]
        kv_chunk = blob.reshape(*blob.shape[:-2], hidden_dim)  # [nlayers, 2, ntokens, num_heads, head_size]
        memory_obj = self.memory_allocator.allocate(kv_chunk.shape,
                                                    kv_chunk.dtype,
                                                    fmt=MemoryFormat.KV_BLOB2)
        if memory_obj is None:
            logger.warning("Memory allocation failed in cachegen deserializer")
            return None

        assert memory_obj.tensor is not None
        memory_obj.tensor.copy_(kv_chunk)

        return memory_obj