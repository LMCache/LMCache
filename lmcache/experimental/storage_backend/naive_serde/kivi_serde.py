from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj, MemoryFormat, TensorMemoryObj)
from lmcache.experimental.storage_backend.naive_serde.serde import (
    Deserializer, Serializer)
from lmcache.experimental.storage_backend.naive_serde.new_pack import (triton_quantize_and_pack_along_last_dim, quant_and_pack_kcache, quant_and_pack_vcache)
import torch
from torch import Tensor
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.logging import init_logger
from typing import Union
import time

logger = init_logger(__name__)

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

    def preprocess_compressed_dict(self, compressed_dict, quant_common_dtype=torch.float32):
        """
        Preprocesses a list of dictionary entries by splitting fields into two groups:
        
        1. Non-quant fields (all keys except "quant_k" and "quant_v") are flattened
        and concatenated into a single 1D tensor (`big_tensor`).
        Returns per-entry metadata and global entry offsets.
        
        2. Quant fields ("quant_k" and "quant_v") are cast to a common dtype (default: float32),
        flattened, and concatenated (first quant_k then quant_v for each entry) into a single 1D tensor 
        (`quant_big_tensor`). Returns per-entry metadata and global entry offsets.
        
        Returns:
        big_tensor: 1D tensor holding concatenated non-quant field data.
        non_quant_metadata: List (one per entry) of metadata dicts for non-quant fields.
                            Each dict contains:
                                - "keys_order": fixed order of non-quant keys.
                                - For each key: dict with 'shape', 'numel', and 'offset' (relative to that entry).
                                - 'entry_length': total number of elements in that entry.
        non_quant_entry_offsets: List of (start, end) tuples indicating each entry’s slice in big_tensor.
        
        quant_big_tensor: 1D tensor holding concatenated quant field data (both quant_k and quant_v).
        quant_metadata: List (one per entry) of metadata dicts for quant fields.
                        Each dict contains:
                            - "keys_order": fixed order of quant keys (["quant_k", "quant_v"]).
                            - For each key: dict with 'shape', 'numel', and 'offset' (relative to that entry).
                            - 'entry_length': total number of elements in that entry.
        quant_entry_offsets: List of (start, end) tuples indicating each entry’s slice in quant_big_tensor.
        """
        quant_keys = ["quant_k", "quant_v"]
        # All keys in an entry; non-quant keys are all others.
        # all_keys = sorted(compressed_dict[0].keys())
        all_keys = ["k_min", "v_min", "k_scale", "v_scale", "residual_k", "residual_v"]
        non_quant_keys = [k for k in all_keys if k not in quant_keys]
        
        # --- Process non-quant fields ---
        non_quant_flattened_entries = []
        non_quant_metadata = []
        for d in compressed_dict:
            meta = {'keys_order': non_quant_keys}
            parts = []   # flattened parts for this entry
            offset = 0   # running offset within this entry's flattened vector
            for key in non_quant_keys:
                value = d.get(key, None)
                if value is None:
                    meta[key] = {'shape': None, 'numel': 0, 'offset': offset}
                else:
                    flat_val = value.reshape(-1)
                    numel = flat_val.numel()
                    meta[key] = {'shape': value.shape, 'numel': numel, 'offset': offset}
                    parts.append(flat_val)
                    offset += numel
            meta['entry_length'] = offset
            entry_flat = torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32, device="cuda")
            non_quant_flattened_entries.append(entry_flat)
            non_quant_metadata.append(meta)
        
        # Compute global entry offsets for non-quant fields.
        non_quant_entry_offsets = []
        cumulative = 0
        for entry in non_quant_flattened_entries:
            start = cumulative
            end = cumulative + entry.numel()
            non_quant_entry_offsets.append((start, end))
            cumulative = end
        big_tensor = torch.cat(non_quant_flattened_entries) if non_quant_flattened_entries else torch.tensor([], dtype=torch.float32, device="cuda")
        
        # --- Process quant fields ---
        quant_flattened_entries = []
        quant_metadata = []
        for d in compressed_dict:
            meta = {'keys_order': quant_keys}
            parts = []  # flattened parts for quant fields in this entry
            offset = 0  # running offset for this entry's quant data
            for key in quant_keys:
                value = d.get(key, None)
                if value is None:
                    meta[key] = {'shape': None, 'numel': 0, 'offset': offset}
                else:
                    # Cast to common dtype
                    flat_val = value.reshape(-1)
                    numel = flat_val.numel()
                    meta[key] = {'shape': value.shape, 'numel': numel, 'offset': offset}
                    parts.append(flat_val)
                    offset += numel
            meta['entry_length'] = offset
            entry_flat = torch.cat(parts) if parts else torch.tensor([], dtype=quant_common_dtype, device="cuda")
            quant_flattened_entries.append(entry_flat)
            quant_metadata.append(meta)
        
        # Compute global entry offsets for quant fields.
        quant_entry_offsets = []
        cumulative = 0
        for entry in quant_flattened_entries:
            start = cumulative
            end = cumulative + entry.numel()
            quant_entry_offsets.append((start, end))
            cumulative = end
        quant_big_tensor = torch.cat(quant_flattened_entries) if quant_flattened_entries else torch.tensor([], dtype=quant_common_dtype, device="cuda")
        
        return (big_tensor, non_quant_metadata, non_quant_entry_offsets,
                quant_big_tensor, quant_metadata, quant_entry_offsets)

    def serialize(self, memory_obj: MemoryObj, bits: int = 4) -> MemoryObj:

        # NOTE(Shaoting): KIVI compression needs cuda
        if type(memory_obj) == Tensor: 
            t = memory_obj
        else:
            t = memory_obj.tensor.cuda()
            # t = memory_obj.tensor

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
                
                # quant_k_triton, k_scale_triton, k_min_triton = quant_and_pack_kcache(rounded_k.transpose(2, 3).contiguous(), self.group_size, bits)
                # quant_v_triton, v_scale_triton, v_min_triton = quant_and_pack_vcache(rounded_v.contiguous(), self.group_size, bits)
                
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

        # Assuming you have already preprocessed:
        big_tensor, metadata, entry_offsets, quant_big_tensor, quant_metadata, quant_entry_offsets = self.preprocess_compressed_dict(compressed_dict)

        big_tensor_bytes = big_tensor.view(torch.uint8)
        quant_tensor_bytes = quant_big_tensor.view(torch.uint8)
        saved_tensor = torch.cat([big_tensor_bytes, quant_tensor_bytes])
        split_metadata = {
            "num_big_bytes": big_tensor.numel() * big_tensor.element_size(),                  
            "big_tensor_dtype": big_tensor.dtype,            
            "quant_tensor_dtype": quant_big_tensor.dtype          
        }

        allocated_obj = self.memory_allocator.allocate(saved_tensor.shape, saved_tensor.dtype, fmt=MemoryFormat.KV_BLOB2)
        allocated_obj.tensor.copy_(saved_tensor)
        return allocated_obj, metadata, entry_offsets, split_metadata, quant_metadata, quant_entry_offsets

class KIVIDeserializer(Deserializer):

    def __init__(self, memory_allocator: MemoryAllocatorInterface):
        self.memory_allocator = memory_allocator
        self.residual = 128
        self.group_size = 32
        self.non_quant_stream = torch.cuda.Stream()
        self.quant_stream = torch.cuda.Stream()

    def resume_non_quant_entry(self, big_tensor, meta, entry_offset):
        """
        Returns a dictionary of non-quant fields for one entry as views into big_tensor.
        
        Parameters:
        big_tensor: The 1D tensor holding all non-quant data.
        meta: Metadata for this entry (includes 'keys_order' and per-key info).
        entry_offset: Tuple (start, end) for this entry in big_tensor.
        
        Returns:
        A dict mapping each non-quant key to a tensor view (or None).
        """
        start, end = entry_offset
        entry_tensor = big_tensor[start:end]
        resumed = {}
        for key in meta['keys_order']:
            key_meta = meta[key]
            if key_meta['numel'] == 0:
                resumed[key] = None
            else:
                off = key_meta['offset']
                resumed[key] = entry_tensor[off: off + key_meta['numel']].view(key_meta['shape'])
        return resumed

    def resume_quant_entry(self, quant_big_tensor, meta, entry_offset):
        """
        Returns a dictionary of quant fields for one entry as views into quant_big_tensor.
        
        Parameters:
        quant_big_tensor: The 1D tensor holding all quant data (both quant_k and quant_v)
                            in the common dtype.
        meta: Metadata for this entry for quant fields (includes 'keys_order' for quant keys).
        entry_offset: Tuple (start, end) indicating this entry’s slice in quant_big_tensor.
        
        Returns:
        A dict mapping each quant key ("quant_k" and "quant_v") to a tensor view (or None).
        """
        start, end = entry_offset
        entry_tensor = quant_big_tensor[start:end]
        resumed = {}
        for key in meta['keys_order']:
            key_meta = meta[key]
            if key_meta['numel'] == 0:
                resumed[key] = None
            else:
                off = key_meta['offset']
                resumed[key] = entry_tensor[off: off + key_meta['numel']].view(key_meta['shape'])
        return resumed

    @_lmcache_nvtx_annotate
    def deserialize(self, memory_obj: Union[MemoryObj, str], bits: int = 4, metadata = [], entry_offsets = [], split_metadata = None, quant_metadata = [], quant_entry_offsets = []) -> MemoryObj:
        if isinstance(memory_obj, str):
            with open(memory_obj, 'rb') as f:
                compressed_dict = torch.load(f)
        else:
            pass

        ks = []
        vs = []

        big_tensor_bytes_rec = memory_obj.tensor[:split_metadata["num_big_bytes"]]
        quant_tensor_bytes_rec = memory_obj.tensor[split_metadata["num_big_bytes"]:]

        if type(memory_obj) == TensorMemoryObj:
             self.memory_allocator.ref_count_down(memory_obj)

        big_tensor_rec = big_tensor_bytes_rec.view(split_metadata["big_tensor_dtype"])
        quant_tensor_rec = quant_tensor_bytes_rec.view(split_metadata["quant_tensor_dtype"])
        with torch.cuda.stream(self.non_quant_stream):
            non_quant_gpu_tensor = big_tensor_rec.cuda(non_blocking=True)
        with torch.cuda.stream(self.quant_stream):
            quant_gpu_tensor = quant_tensor_rec.cuda(non_blocking=True)

        torch.cuda.synchronize()

        for layer in range(len(metadata)):
            # Resume non-quant fields.
            non_quant_entry = self.resume_non_quant_entry(non_quant_gpu_tensor, metadata[layer], entry_offsets[layer])
            # Resume quant fields.
            quant_entry = self.resume_quant_entry(quant_gpu_tensor, quant_metadata[layer], quant_entry_offsets[layer])
            
            # Now extract fields as views:
            quant_k    = quant_entry["quant_k"]
            k_scale    = non_quant_entry.get("k_scale")
            k_min      = non_quant_entry.get("k_min")
            quant_v    = quant_entry["quant_v"]
            v_scale    = non_quant_entry.get("v_scale")
            v_min      = non_quant_entry.get("v_min")
            residual_k = non_quant_entry.get("residual_k")
            residual_v = non_quant_entry.get("residual_v") #hahaha

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

        return kv_chunk
