import torch 
import torchac
import numpy as np
import pickle
import os
import timeit
import torchac_cuda
import time
import multiprocessing
def cuda_job(job_id, other_args):
    """
    A function that runs a specific CUDA job.
    :param job_id: An identifier for the job.
    :param other_args: Other arguments needed for the CUDA job.
    """
    print(f"Starting CUDA job {job_id}")
    # Your CUDA code here
    torchac.test(other_args[0], other_args[1], 2000, 10, 100)
    print(f"Finished CUDA job {job_id}")

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

def collect_bytes(output_buffer, output_lengths):
    output_buffer_size = output_buffer.shape[-1]
    flattened_lengths = output_lengths.flatten()
    flattened_buffer = output_buffer.flatten()
    summed_length = (output_buffer_size - flattened_lengths).cumsum(0)
    summed_length = summed_length.roll(1)
    summed_length[0] = 0
    indexes = summed_length.repeat_interleave(flattened_lengths)
    indexes = indexes + torch.arange(len(indexes), device=indexes.device)
    return flattened_buffer[indexes]

def recombine_bytes(bytes_tensor, output_lengths, output_buffer_size):
    offsets = output_lengths.flatten().cumsum(0).roll(1).reshape(output_lengths.shape)
    offsets[0][0] = 0
    indexes = torch.arange(output_buffer_size, device=offsets.device).tile((output_lengths.shape[0], output_lengths.shape[1], 1))
    final_indexes = (indexes + offsets[:, :, None]).clamp(max = len(bytes_tensor) - 1)
    return bytes_tensor[final_indexes]



if __name__ == "__main__":
    cdf = torch.load("/tmp/cdf.pt")
    encode_input = torch.load("/tmp/encode_input.pt")

    nlayers = 64
    nchannels = 1024
    ntokens = 256
    output_buffer_size = 256

    cdf = cdf[:nlayers, :nchannels, :].contiguous().cuda()
    encode_input = encode_input[:nlayers, :ntokens, :nchannels].contiguous().cuda()
    ntokens = encode_input.shape[1]
    Lp = cdf.shape[-1]
    newcdf = cdf[:, None, :, :].repeat((1, ntokens, 1, 1))

    output_buffer = torch.zeros((nlayers, nchannels, output_buffer_size), dtype = torch.uint8, device="cuda")
    output_lengths = torch.zeros((nlayers, nchannels), dtype=torch.int, device="cuda")
    start = time.perf_counter()
    torchac_cuda.encode_fast_new(
            cdf.cuda(),
            encode_input.cuda(),
            output_buffer,
            output_lengths,
    )
    bytes_tensor = collect_bytes(output_buffer, output_lengths)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print("Total encode time: ", end - start)
    print("Total length: ", output_lengths.sum())
    print(cdf.shape)

    decode_buffer = torch.zeros((nlayers, output_buffer_size, nchannels), dtype = torch.uint8, device="cuda")
    start = time.perf_counter()
    output_lengths_prefsum = output_lengths.flatten().cumsum(0).reshape(output_lengths.shape)
    torchac_cuda.decode_fast_prefsum(
            cdf.cuda(),
            bytes_tensor.cuda(),
            output_lengths_prefsum.cuda(),
            decode_buffer
    )

    torch.cuda.synchronize()
    end = time.perf_counter();

    print("Equal check:", torch.equal(decode_buffer, encode_input))
    print("Total decode time: ", end - start)
    breakpoint()

    decode_buffer = torch.zeros((nlayers, output_buffer_size, nchannels), dtype = torch.uint8, device="cuda")
    start = time.perf_counter()
    new_output_buffer = recombine_bytes(bytes_tensor, output_lengths, output_buffer_size)
    torchac_cuda.decode_fast_new(
            cdf.cuda(),
            new_output_buffer.cuda(),
            output_lengths.cuda(),
            decode_buffer
    )

    torch.cuda.synchronize()
    end = time.perf_counter();

    print("Equal check:", torch.equal(decode_buffer, encode_input))
    print("Total decode time: ", end - start)

    breakpoint()

    #bytes_tensor = collect_bytes(output_buffer, output_lengths)
    #ret = timeit.timeit("bytes_tensor = collect_bytes(output_buffer, output_lengths)", globals=globals(), number=100)   
    #print(ret)

    #breakpoint()
    #new_output_buffer = recombine_bytes(bytes_tensor, output_lengths, output_buffer_size)
    #ret = timeit.timeit("bytes_tensor = collect_bytes(output_buffer, output_lengths)", globals=globals(), number=100)   
    #print(ret)
    #torchac_cuda.decode_fast_new(
    #        cdf.cuda(),
    #        new_output_buffer.cuda(),
    #        output_lengths.cuda(),
    #        decode_buffer
    #)
    #print("Equal check:", torch.equal(decode_buffer, encode_input))
    #breakpoint()


    #for i in range(nlayers):
    #    for j in range(nchannels):
    #        torchac_result = torchac.encode_int16_normalized_cdf(
    #                newcdf[i, :, j, :].contiguous().cpu(),
    #                encode_input[i, :, j].contiguous().to(torch.int16).cpu())

    #        our_length = int(output_lengths[i, j])
    #        ref_length = len(torchac_result)
    #        print("Lengths: ", our_length, ref_length)  
    #        our_result = output_buffer[i, j, :output_lengths[i, j]].cpu().numpy()
    #        ref_result = np.frombuffer(torchac_result, dtype=np.uint8)
    #        if our_length != ref_length or not np.equal(our_result, ref_result).all():
    #            print(f"\033[31mFailed layer {i} channel {j}\033[0m")
    #            print("Our result", our_result)
    #            print("Ref result", ref_result)
    #            breakpoint()
    #        else:
    #            print(f"\033[32mPassed layer {i} channel {j}\033[0m")

    print("Total time: ", end - start)
    print("Total length: ", output_lengths.sum())
