import torch 
import pickle
import os
import torchac
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
if __name__ == "__main__":
    bitstreams = pickle.load(open("/home/cc/L3C-PyTorch/test_sanity/test.pkl", \
        "rb"))
    cdf = bitstreams[1][0].repeat((1, 1000, 64, 1)).cuda()
    output = torch.randint(0, 10, (4000, 64000)).cuda().to(torch.int32)
    # start_indices = []
    # end_indices = []
    # input_string = ""
    # for i in range(0, 4000):
    #     if i %100 == 0:
    #         print(i)
    #     start_indices.append(len(input_string))
    #     input_string += str(bitstreams[0][i])
    #     end_indices.append(len(input_string))
        
    # pickle.dump(input_string, open("test_str.pkl", "wb"))
    # pickle.dump(start_indices, open("test_start_indices.pkl", "wb"))
    # pickle.dump(end_indices, open("test_end_indices.pkl", "wb"))
    
    input_string = pickle.load(open("test_str.pkl", "rb"))
    start_indices = pickle.load(open("test_start_indices.pkl", "rb"))
    end_indices = pickle.load(open("test_end_indices.pkl", "rb"))
    print(len(bitstreams[0]))
    for i in range(0, 6):
        st = time.monotonic()
        # out = torchac.decode_fast(output, cdf, bitstreams[0][:4000],  4000, 400, 10, input_string, start_indices)
        # out = torchac.decode_optimize(output, cdf, \
        #     4000, 400, 10, start_indices, end_indices, input_string)
        out = torchac.decode(output, cdf, bitstreams[0][:4000],  4000, 400, 10)
        # out = torchac.decode_optimize( cdf, input_string, start_indices, end_indices, 4000, 400, 10)
        print(f"Total time: {time.monotonic() - st}")
    # breakpoint()
    #     print(f"Total time: {time.monotonic() - st}")
    # breakpoint()
    breakpoint()