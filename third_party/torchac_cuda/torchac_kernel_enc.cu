#include "torchac_kernel.cuh"

struct cdf_ptr
{
    const cdf_t *data; // expected to be a N_sym x Lp matrix, stored in row major.
    const int N_sym;   // Number of symbols stored by `data`.
    const int Lp;      // == L+1, where L is the number of possible values a symbol can take.
    cdf_ptr(const cdf_t *data,
            const int N_sym,
            const int Lp) : data(data), N_sym(N_sym), Lp(Lp){};
};


/** Class to save output bit by bit to a byte string */
class OutCacheString {
private:
public:
    std::string out="";
    uint8_t cache=0;
    uint8_t count=0;
    void append(const int bit) {
        cache <<= 1;
        cache |= bit;
        count += 1;
        if (count == 8) {
            out.append(reinterpret_cast<const char *>(&cache), 1);
            count = 0;
        }
    }
    void flush() {
        if (count > 0) {
            for (int i = count; i < 8; ++i) {
                append(0);
            }
            assert(count==0);
        }
    }
    void append_bit_and_pending(const int bit, uint64_t &pending_bits) {
        append(bit);
        while (pending_bits > 0) {
            append(!bit);
            pending_bits -= 1;
        }
    }
};


/** Class to read byte string bit by bit */
class InCacheString {
private:
    const std::string& in_;

public:
    explicit InCacheString(const std::string& in) : in_(in) {};

    uint8_t cache=0;
    uint8_t cached_bits=0;
    size_t in_ptr=0;

    void get(uint32_t& value) {
        if (cached_bits == 0) {
            if (in_ptr == in_.size()){
                value <<= 1;
                return;
            }
            /// Read 1 byte
            cache = (uint8_t) in_[in_ptr];
            in_ptr++;
            cached_bits = 8;
        }
        value <<= 1;
        value |= (cache >> (cached_bits - 1)) & 1;
        cached_bits--;
    }

    void initialize(uint32_t& value) {
        for (int i = 0; i < 32; ++i) {
            get(value);
        }
    }
};


const void check_sym(const torch::Tensor& sym) {
    TORCH_CHECK(sym.sizes().size() == 1,
                "Invalid size for sym. Expected just 1 dim.")
}


/** Get an instance of the `cdf_ptr` struct. */
const struct cdf_ptr get_cdf_ptr(const torch::Tensor& cdf)
{
    TORCH_CHECK(!cdf.is_cuda(), "cdf must be on CPU!")
    const auto s = cdf.sizes();
    TORCH_CHECK(s.size() == 2, "Invalid size for cdf! Expected (N, Lp)")

    const int N_sym = s[0];
    const int Lp = s[1];
    const auto cdf_acc = cdf.accessor<int16_t, 2>();
    const cdf_t* cdf_ptr = (uint16_t*)cdf_acc.data();

    const struct cdf_ptr res(cdf_ptr, N_sym, Lp);
    return res;
}


const struct cdf_ptr get_cdf_ptr_ours(const at::Tensor &cdf)
{
    // AT_CHECK(!cdf.is_cuda(), "cdf must be on CPU!")
    const auto s = cdf.sizes();
    // AT_CHECK(s.size() == 4 && s[0] == 1, "Invalid size for cdf! Expected 1HWLp")

    const int N_sym = s[1] * s[2];
    const int Lp = s[3];
    const auto cdf_reshaped = at::reshape(cdf, {N_sym, -1});
    const auto cdf_acc = cdf_reshaped.accessor<int16_t, 2>();
    const cdf_t *cdf_ptr = (uint16_t *)cdf_acc.data();

    const struct cdf_ptr res(cdf_ptr, N_sym, Lp);
    return res;
}




__device__ void append_cache_to_string(char* out, uint8_t& cache, int max_out_size) {
    // find the end of the string
    int length = 0;
    while (length < max_out_size - 1 && out[length] != '\0') {
        length++;
    }

    // append the new character if there's space
    if (length < max_out_size - 1) {
        out[length] = static_cast<char>(cache);
        out[length + 1] = '\0';
    }
}


__device__ void append_to_end(char* out, uint8_t cache, int* current_out_length) {
    // find the end of the string
    out[*current_out_length] = static_cast<char>(cache);

    // append the null character
    out[*current_out_length + 1] = '\0';
}


__device__ void append_to_end_uint(char* out, uint8_t cache, 
                                   uint32_t device_out_offset, uint32_t current_out_length) {
    // find the end of the string
    out[device_out_offset + current_out_length] = static_cast<char>(cache);

    // append the null character
    out[device_out_offset + current_out_length + 1] = '\0';
}



// cuda version (multi-threaded)
__global__ void encode_with_cuda(int16_t* device_sym, 
                                 char* device_out, 
                                 const uint32_t max_out_size,
                                 const cdf_t *cdf, 
                                 const int N_sym, 
                                 const int Lp, 
                                 uint32_t* device_out_lengths,
                                 const uint32_t total_num_of_threads) {
    
    // printf("enter encode_with_cuda()\n");

    // printf("N_sym = %d\n", N_sym);
    
    uint8_t cache = 0;
    uint8_t count = 0;
    int bit = 0;

    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint64_t pending_bits = 0;

    const int precision = 16;

    const int max_symbol = Lp - 2;

    // multi-threading related variables
    uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t per_thread_sym_coverage = ceil(N_sym / total_num_of_threads);
    const uint32_t sym_offset = thread_index * per_thread_sym_coverage;
    const uint32_t device_out_offset = thread_index * max_out_size;

    // for (int i = 0; i < N_sym; ++i) {
    for (int i = sym_offset; i < sym_offset + per_thread_sym_coverage; ++i) {

        const uint32_t sym_i = device_sym[i];

        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;

        const int offset = i * Lp;
        // Left boundary is at offset + sym_i
        const uint32_t c_low = cdf[offset + sym_i];
        // Right boundary is at offset + sym_i + 1, except for the `max_symbol`
        // For which we hardcode the maxvalue. So if e.g.
        // L == 4, it means that Lp == 5, and the allowed symbols are
        // {0, 1, 2, 3}. The max symbol is thus Lp - 2 == 3. It's probability
        // is then given by c_max - cdf[-2].
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low =  (low)     + ((span * static_cast<uint64_t>(c_low))  >> precision);

        while (true) {
            if (high < 0x80000000U) {
                // out_cache.append_bit_and_pending(0, pending_bits);
                
                bit = 0;
                cache <<= 1;
                cache |= bit;
                count += 1;
                if (count == 8) {
                    // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                    // append_cache_to_string(device_out, cache, max_out_size);
                    append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                    atomicAdd(&device_out_lengths[thread_index], 1);
                    // printf("Value after increment: %d\n", *current_out_length);
                    count = 0;
                }
                while (pending_bits > 0) {
                    bit = 1;
                    cache <<= 1;
                    cache |= bit;
                    count += 1;
                    if (count == 8) {
                        // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                        // append_cache_to_string(device_out, cache, max_out_size);
                        append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                        atomicAdd(&device_out_lengths[thread_index], 1);
                        // printf("Value after increment: %d\n", *current_out_length);
                        count = 0;
                    }
                    pending_bits -= 1;
                }

                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x80000000U) {
                // out_cache.append_bit_and_pending(1, pending_bits);
                bit = 1;
                cache <<= 1;
                cache |= bit;
                count += 1;
                if (count == 8) {
                    // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                    // append_cache_to_string(device_out, cache, max_out_size);
                    append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                    atomicAdd(&device_out_lengths[thread_index], 1);
                    // printf("Value after increment: %d\n", *current_out_length);
                    count = 0;
                }
                while (pending_bits > 0) {
                    bit = 0;
                    cache <<= 1;
                    cache |= bit;
                    count += 1;
                    if (count == 8) {
                        // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                        // append_cache_to_string(device_out, cache, max_out_size);
                        append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                        atomicAdd(&device_out_lengths[thread_index], 1);
                        // printf("Value after increment: %d\n", *current_out_length);
                        count = 0;
                    }
                    pending_bits -= 1;
                }

                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                pending_bits++;
                low <<= 1;
                low &= 0x7FFFFFFF;
                high <<= 1;
                high |= 0x80000001;
            } else {
                break;
            }
        }
    }

    pending_bits += 1;

    if (pending_bits) {
        if (low < 0x40000000U) {
            // out_cache.append_bit_and_pending(0, pending_bits);
            bit = 0;
            cache <<= 1;
            cache |= bit;
            count += 1;
            if (count == 8) {
                // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                // append_cache_to_string(device_out, cache, max_out_size);
                append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                atomicAdd(&device_out_lengths[thread_index], 1);
                // printf("Value after increment: %d\n", *current_out_length);
                count = 0;
            }
            while (pending_bits > 0) {
                bit = 1;
                cache <<= 1;
                cache |= bit;
                count += 1;
                if (count == 8) {
                    // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                    // append_cache_to_string(device_out, cache, max_out_size);
                    append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                    atomicAdd(&device_out_lengths[thread_index], 1);
                    // printf("Value after increment: %d\n", *current_out_length);
                    count = 0;
                }
                pending_bits -= 1;
            }
        } else {
            // out_cache.append_bit_and_pending(1, pending_bits);
            bit = 1;
            cache <<= 1;
            cache |= bit;
            count += 1;
            if (count == 8) {
                // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                // append_cache_to_string(device_out, cache, max_out_size);
                append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                atomicAdd(&device_out_lengths[thread_index], 1);
                // printf("Value after increment: %d\n", *current_out_length);
                count = 0;
            }
            while (pending_bits > 0) {
                bit = 0;
                cache <<= 1;
                cache |= bit;
                count += 1;
                if (count == 8) {
                    // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                    // append_cache_to_string(device_out, cache, max_out_size);
                    append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                    atomicAdd(&device_out_lengths[thread_index], 1);
                    // printf("Value after increment: %d\n", *current_out_length);
                    count = 0;
                }
                pending_bits -= 1;
            }
        }
    }

    // flush
    if (count > 0) {
        for (int i = count; i < 8; ++i) {
            // append(0);
            bit = 0;
            cache <<= 1;
            cache |= bit;
            count += 1;
            if (count == 8) {
                // out_cache_string.append(reinterpret_cast<const char *>(&cache), 1);
                // append_cache_to_string(device_out, cache, max_out_size);
                append_to_end_uint(device_out, cache, device_out_offset, device_out_lengths[thread_index]);
                atomicAdd(&device_out_lengths[thread_index], 1);
                // printf("Value after increment: %d\n", *current_out_length);
                count = 0;
            }
        }
        assert(count==0);
    }

}



std::vector<py::bytes> encode_cuda(const at::Tensor &cdf, 
                                   const at::Tensor &input_sym, 
                                   const uint32_t max_out_size,
                                   const int blockNum, 
                                   const int threadNum)
{
    const uint32_t total_num_of_threads = blockNum * threadNum;

    // allocate device memory for cdf
    const auto cdf_ptr = get_cdf_ptr(cdf);
    cdf_t *cdf_data;
    const size_t size_cdf = cdf_ptr.N_sym * cdf_ptr.Lp * sizeof(cdf_t); // Calculate the size of the array.
    cudaMalloc(&cdf_data, size_cdf);
    cudaMemcpy(cdf_data, cdf_ptr.data, size_cdf, cudaMemcpyHostToDevice);
    
    // allocate device memory for device_sym (AC input)
    // std::cout << "allocate device memory for device_sym (AC input)" << std::endl;
    int16_t* device_sym;
    const size_t device_sym_size = input_sym.numel() * sizeof(int16_t);
    cudaMalloc(&device_sym, device_sym_size);
    cudaMemcpy(device_sym, input_sym.data_ptr<int16_t>(), device_sym_size, cudaMemcpyHostToDevice);

    // allocate device memory for device_out (AC output)
    // NOTE: need separate memory for each thread
    // std::cout << "allocate device memory for device_out (AC output)" << std::endl;
    char* device_out;
    const size_t device_out_size = max_out_size * total_num_of_threads * sizeof(char);
    cudaMalloc(&device_out, device_out_size);
    cudaMemset(device_out, 0, device_out_size);

    // allocate device memory for device_out_lengths
    // NOTE: need separate memory for each thread
    // std::cout << "allocate device memory for device_out_lengths" << std::endl;
    uint32_t* device_out_lengths;
    const size_t device_out_lengths_size = total_num_of_threads * sizeof(uint32_t);
    cudaMalloc(&device_out_lengths, device_out_lengths_size);
    cudaMemset(device_out_lengths, 0, device_out_lengths_size);

    // std::cout << "before entering encode_with_cuda()" << std::endl;
    encode_with_cuda<<<blockNum, threadNum>>>(device_sym, 
                                              device_out, 
                                              max_out_size, 
                                              cdf_data, 
                                              cdf_ptr.N_sym, 
                                              cdf_ptr.Lp, 
                                              device_out_lengths,
                                              total_num_of_threads);
    // std::cout << "after returning from encode_with_cuda()" << std::endl;

    // for debugging
    uint32_t* local_out_lengths = new uint32_t[device_out_lengths_size];
    cudaMemcpy(local_out_lengths, device_out_lengths, device_out_lengths_size, cudaMemcpyDeviceToHost);
    // std::cout << "printing local_out_lengths" << std::endl;
    // for (uint32_t thread_index = 0; thread_index < total_num_of_threads; thread_index++) {
    //     std::cout << local_out_lengths[thread_index] << " ";
    // }
    // std::cout << std::endl;

    // move encoded data back to main memory
    std::vector<char> host_out(device_out_size);
    cudaMemcpy(host_out.data(), device_out, device_out_size, cudaMemcpyDeviceToHost);
    
    // extract valid results from different threads
    std::vector<char> valid_out;
    std::vector<py::bytes> all_results;
    uint32_t valid_total_length = 0;
    
    for (uint32_t thread_index = 0; thread_index < total_num_of_threads; thread_index++) {
        uint32_t valid_length = local_out_lengths[thread_index];
        valid_total_length += valid_length;

        std::vector<char> current_result(host_out.begin() + thread_index * max_out_size, 
                                         host_out.begin() + thread_index * max_out_size + valid_length);

        // append result to all_results
        all_results.push_back(py::bytes(current_result.data(), valid_length));
    }
    
    cudaFree(cdf_data);
    cudaFree(device_sym);
    cudaFree(device_out);
    
    // std::cout << "before returning py::bytes()" << std::endl;

    // return py::bytes(host_out.data(), local_out_lengths[0]);
    // return py::bytes(valid_out.data(), valid_total_length);
    return all_results;
}



// namespace py = pybind11;

// PYBIND11_MODULE(torchac_cuda, m) {
//     // m.def("decode", &decode, "decode function");
//     m.def("encode_fast", &encode_cuda, "Fast encode function");
// }
