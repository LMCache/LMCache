/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "kernel_operator.h"
#include <stdio.h>
#include "types.h"
#include "utils.h"

template <typename scalar_t, typename slot_t, bool IsMLA> class SingleLayerPagedKVCopy {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline SingleLayerPagedKVCopy()
    {
    }

    __aicore__ inline void init(GM_ADDR cacheTensor, GM_ADDR keyCachePtr, GM_ADDR valueCachePtr, GM_ADDR slotmappings,
                                const int64_t hiddenDims, const int32_t numTokens, const bool page2L, 
                                const bool tokenMajor, AscendC::TPipe *pipe)
    {
        this->pipe_ = pipe;
        this->hiddenDims_ = hiddenDims;
        this->numTokens_ = numTokens;
        this->tokenMajor_ = tokenMajor;
        this->valid_ = true;
        this->page2L_ = page2L;
        if constexpr (IsMLA) {
            this->numKvs_ = 1;
        } else {
            this->numKvs_ = 2;
        }
        // TODO: Not sure how many to allocate, but let's do 4 blocks of hiddenDims_
        // if it was fp16, 2048, we would get 16kb ？
        this->pipe_->InitBuffer(this->pagedTokenQue_, 4, this->hiddenDims_*sizeof(scalar_t));
    }

    __aicore__ inline void reset(){
        this->valid_ = true;
    }

    __aicore__ inline void updateTensorMemOffsetAndProcess(__gm__ uint8_t *pagedTensor, __gm__ uint8_t* nonPagedTensor, 
                                             __gm__ uint8_t *slotmappings, const int tokenIdx, const int kvIdx) 
    {
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);
        int64_t slot = slotmappingPtr[tokenIdx];
    
        if (slot == -1) {
            this->valid_ = false;
            return;
        }

        // for the page tensor
        int64_t pagedIdxOffset = slot * this->hiddenDims_;
        
        // for the lmc tensor
        int64_t nonPagedIdxOffset = -1;
        if (this->tokenMajor_) {
            nonPagedIdxOffset = tokenIdx * this->numKvs_ * this->hiddenDims_ + 
                                kvIdx * this->hiddenDims_;
        } else {
            nonPagedIdxOffset = kvIdx * this->numTokens_ * this -> hiddenDims_ +
                                tokenIdx * this->hiddenDims_;
        }

        if (kvIdx == 0) {
            // keys
            this->keyTokensGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(pagedTensor) + pagedIdxOffset,
                                                    this->hiddenDims_);
            this->lmcBufferKeyGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(nonPagedTensor) + nonPagedIdxOffset,
                                                    this->hiddenDims_);
        } else {
            // values
            this->valueTokensGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(pagedTensor) + pagedIdxOffset,
                                                    this->hiddenDims_);
            this->lmcBufferValueGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(nonPagedTensor) + nonPagedIdxOffset,
                                                    this->hiddenDims_);
        }
    }

    __aicore__ inline void processFunc() {
        if (!this->valid_) {
            return;
        }
        // 1. Alloc Tensor for local page
        local_scalar_t hiddenKeysDimTensor = this->pagedTokenQue_.template AllocTensor<scalar_t>();
        local_scalar_t hiddenValuesDimTensor;
        if constexpr(!IsMLA) {
            hiddenValuesDimTensor = this->pagedTokenQue_.template AllocTensor<scalar_t>();
        }

        // 2. copy from global tensor into local
        if (this->page2L_) {
            AscendC::DataCopy(hiddenKeysDimTensor, this->keyTokensGlobal_, this->hiddenDims_);
            if constexpr (!IsMLA) {
                AscendC::DataCopy(hiddenValuesDimTensor, this->valueTokensGlobal_, this->hiddenDims_);
            }
        } else {
            AscendC::DataCopy(hiddenKeysDimTensor, this->lmcBufferKeyGlobal_, this->hiddenDims_);
            if constexpr(!IsMLA) {
                AscendC::DataCopy(hiddenValuesDimTensor, this->lmcBufferValueGlobal_, this->hiddenDims_);
            }
        }
       
        // 3. enque vecin
        pagedTokenQue_.EnQue(hiddenKeysDimTensor);
        if constexpr(!IsMLA) {
            pagedTokenQue_.EnQue(hiddenValuesDimTensor);
        }

        // 4. deque vecin, possible to reuse due to QueBind
        hiddenKeysDimTensor = pagedTokenQue_.DeQue<scalar_t>();
        if constexpr(!IsMLA) {
            hiddenValuesDimTensor = pagedTokenQue_.DeQue<scalar_t>();
        }

        // 5. datacopy into GM 
        if (this->page2L_) {
            AscendC::DataCopy(this->lmcBufferKeyGlobal_, hiddenKeysDimTensor, this->hiddenDims_);
            if constexpr(!IsMLA) {
                AscendC::DataCopy(this->lmcBufferValueGlobal_, hiddenValuesDimTensor, this->hiddenDims_);
            }
        } else {
            AscendC::DataCopy(this->keyTokensGlobal_, hiddenKeysDimTensor, this->hiddenDims_);
            if constexpr(!IsMLA) {
                AscendC::DataCopy(this->valueTokensGlobal_, hiddenValuesDimTensor, this->hiddenDims_);
            }
        }
        
        // 6. free alloced Tensor
        pagedTokenQue_.FreeTensor(hiddenKeysDimTensor);
        if constexpr(!IsMLA) {
            pagedTokenQue_.FreeTensor(hiddenValuesDimTensor);
        }
    }

private:
    AscendC::TPipe *pipe_;
    // a depth of 2
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4> pagedTokenQue_;

    // [kvs, numPages * pagedSize, heads*headsize]
    AscendC::GlobalTensor<scalar_t> keyTokensGlobal_;
    // iff !isMLA
    AscendC::GlobalTensor<scalar_t> valueTokensGlobal_;

    // Depends on LMC setting whether we store in tokensMajor or not.
    // the layout would be the followings:
    // [tokens, kvs, heads*headsize] or [kvs, tokens, heads*headsize]
    // TODO: check whether should combine the two and use a loop
    AscendC::GlobalTensor<scalar_t> lmcBufferKeyGlobal_;
    AscendC::GlobalTensor<scalar_t> lmcBufferValueGlobal_;

    int64_t hiddenDims_; // heads * headsize
    int32_t numTokens_; // num tokens in the cache tensor chunk
    int16_t numKvs_; // 1 if MLA else 2
    bool page2L_; // whether the direction of copy is from page to lmc
    bool tokenMajor_; // whether the lmc buffer is in token major.
    bool valid_;
};

#define SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(TYPE, SLOTTYPE, ISMLA)                                                 \
        extern "C" __global__ __aicore__ void single_layer_paged_kv_copy_##TYPE##_##SLOTTYPE##_##ISMLA(                \
        __gm__ uint8_t* dstCacheTensor, __gm__ uint8_t* keyCachePtr, __gm__ uint8_t* valueCachePtr,                    \
        __gm__ uint8_t* slotmappings, const int64_t hiddenDims, const int32_t numTokens, const int coreNums,           \
        const bool page2L, const bool tokenMajor)                                                                      \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        SingleLayerPagedKVCopy<TYPE, SLOTTYPE, ISMLA> op{};                                                            \
        op.init(dstCacheTensor, keyCachePtr, valueCachePtr, slotmappings, hiddenDims, numTokens,                       \
                page2L, tokenMajor, &pipe);                                                                            \
        int64_t bIdx = AscendC::GetBlockIdx();                                                                         \
        for (int64_t i = bIdx; i < numTokens; i+=coreNums)                                                             \
        {                                                                                                              \
            op.reset();                                                                                                \
            op.updateTensorMemOffsetAndProcess(keyCachePtr, dstCacheTensor, slotmappings, i, 0);                       \
            if constexpr(!ISMLA) {                                                                                     \
                op.updateTensorMemOffsetAndProcess(valueCachePtr, dstCacheTensor, slotmappings, i, 1);                 \
            }                                                                                                          \
            op.processFunc();                                                                                          \
        }                                                                                                              \
    }

// Declare support kernel entry
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(half, int32_t, false);
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(half, int32_t, true);
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(bfloat16_t, int32_t, false);
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(bfloat16_t, int32_t, true);
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(int8_t, int32_t, false);
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(int8_t, int32_t, true);

SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(half, int64_t, false);
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(half, int64_t, true);
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(bfloat16_t, int64_t, false);
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(bfloat16_t, int64_t, true);
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(int8_t, int64_t, false);
SINGLE_LAYER_PAGED_KV_COPY_TYPE_DECLARE(int8_t, int64_t, true);

namespace lmc {

#define SINGLE_LAYER_PAGED_KV_COPY_KERNEL_CALL(TYPE, SLOTTYPE, ISMLA)                                                  \
    single_layer_paged_kv_copy_##TYPE##_##SLOTTYPE##_##ISMLA<<<blockDim, nullptr, stream>>>(dstCacheTensor,            \
                                                        keyCachePtr, valueCachePtr, slotmappings, hiddenDims,          \
                                                        numTokens, blockDim, page2L, tokenMajor);

template<typename T, typename SlotT, bool ISMLA>
void single_layer_paged_kernel(uint32_t blockDim, void *stream, uint8_t *dstCacheTensor, uint8_t *keyCachePtr, 
                               uint8_t *valueCachePtr, uint8_t *slotmappings, const int64_t hiddenDims, 
                               const int32_t numTokens, const bool page2L, const bool tokenMajor);

#define SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(TYPE, SLOTTYPE, ISMLA)                                             \
template<>                                                                                                             \
void single_layer_paged_kernel<TYPE, SLOTTYPE, ISMLA>(uint32_t blockDim, void *stream, uint8_t *dstCacheTensor,        \
                                               uint8_t *keyCachePtr, uint8_t *valueCachePtr, uint8_t *slotmappings,    \
                                               const int64_t hiddenDims, const int32_t numTokens, const bool page2L,   \
                                               const bool tokenMajor){                                                 \
    SINGLE_LAYER_PAGED_KV_COPY_KERNEL_CALL(TYPE, SLOTTYPE, ISMLA);                                                     \
}


SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(half, int32_t, false);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(half, int64_t, false);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(bfloat16_t, int32_t, false);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(bfloat16_t, int64_t, false);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(int8_t, int32_t, false);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(int8_t, int64_t, false);

SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(half, int32_t, true);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(half, int64_t, true);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(bfloat16_t, int32_t, true);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(bfloat16_t, int64_t, true);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(int8_t, int32_t, true);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(int8_t, int64_t, true);



template<typename T>
void dispatch_single_layer_kernel_on_slot_type(vllm_ascend::AscendType slotType, uint32_t blockDim, void *stream, 
                                               uint8_t *dstCacheTensor, uint8_t *keyCachePtr, uint8_t *valueCachePtr,
                                               uint8_t *slotmappings, const int64_t hiddenDims, const int32_t numTokens, 
                                               const bool page2L, const bool tokenMajor, const bool isMLA) {
    if (isMLA) {
        switch(slotType) {
            case vllm_ascend::AscendType::INT32:
                single_layer_paged_kernel<T, int32_t, true>(blockDim, stream, dstCacheTensor, keyCachePtr, valueCachePtr,
                                                    slotmappings, hiddenDims, numTokens, page2L, tokenMajor);
                break;
            case vllm_ascend::AscendType::INT64:
                single_layer_paged_kernel<T, int64_t, true>(blockDim, stream, dstCacheTensor, keyCachePtr, valueCachePtr,
                                                    slotmappings, hiddenDims, numTokens, page2L, tokenMajor);
                break;
            default:
                return;
        }
    } else {
        switch(slotType) {
            case vllm_ascend::AscendType::INT32:
                single_layer_paged_kernel<T, int32_t, false>(blockDim, stream, dstCacheTensor, keyCachePtr, valueCachePtr,
                                                    slotmappings, hiddenDims, numTokens, page2L, tokenMajor);
                break;
            case vllm_ascend::AscendType::INT64:
                single_layer_paged_kernel<T, int64_t, false>(blockDim, stream, dstCacheTensor, keyCachePtr, valueCachePtr,
                                                    slotmappings, hiddenDims, numTokens, page2L, tokenMajor);
                break;
            default:
                return;
        }
    }
    
}


extern void single_layer_kv_transfer_kernel(vllm_ascend::AscendType type, vllm_ascend::AscendType slotType, 
                                            uint32_t blockDim, void *stream, uint8_t *dstCacheTensor, 
                                            uint8_t *keyCachePtr, uint8_t *valueCachePtr,
                                            uint8_t *slotmappings, const int64_t hiddenDims, const int32_t numTokens, 
                                            const bool page2L, const bool tokenMajor, const bool isMLA)
{
    switch(type) {
        case vllm_ascend::AscendType::FP16:
            dispatch_single_layer_kernel_on_slot_type<half>(slotType, blockDim, stream, dstCacheTensor, keyCachePtr, 
                                                            valueCachePtr, slotmappings, hiddenDims, numTokens, page2L, 
                                                            tokenMajor, isMLA);
            break;
        case vllm_ascend::AscendType::BF16:
            dispatch_single_layer_kernel_on_slot_type<bfloat16_t>(slotType, blockDim, stream, dstCacheTensor, keyCachePtr, 
                                                                  valueCachePtr, slotmappings, hiddenDims, numTokens, 
                                                                  page2L, tokenMajor, isMLA);
            break;
        case vllm_ascend::AscendType::INT8:
            dispatch_single_layer_kernel_on_slot_type<int8_t>(slotType, blockDim, stream, dstCacheTensor, keyCachePtr, 
                                                              valueCachePtr, slotmappings, hiddenDims, numTokens, page2L, 
                                                              tokenMajor, isMLA);
        default:
            return;
    }
}

} // namespace lmc
