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

template <typename scalar_t, typename slot_t> class MultiLayerPagedKVCopy {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline MultiLayerPagedKVCopy()
    {
    }

    __aicore__ inline void init(GM_ADDR pagedKVCaches, GM_ADDR cacheTensor, GM_ADDR slotmappings,
                                const int64_t hiddenDims, const int32_t numLayers, const int64_t pageBuffSize,
                                const int32_t numTokensChunk, const bool page2L, 
                                AscendC::TPipe *pipe)
    {
        this->pipe_ = pipe;
        this->numLayers_ = numLayers;
        this->hiddenDims_ = hiddenDims;
        this->pageBuffSize_ = pageBuffSize;
        this->numTokensChunk_ = numTokensChunk;
        this->page2L_ = page2L;
        this->valid_ = true;
        
        this->pipe_->InitBuffer(pagedTokenQue_, 4, this->hiddenDims_*sizeof(scalar_t));
    }

    __aicore__ inline void reset(){
        this->valid_ = true;
    }

    __aicore__ inline void updateMemOffset(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor, 
                                             __gm__ uint8_t *slotmappings, const int tokenIdx, 
                                             const int kvIdx, const int layerIdx) 
    {
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);
        int64_t slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);

        if (slot == -1) {
            this->valid_ = false;
            return;
        }
        
        // its a pointer within the GM addr space, that point to another GM addr space
        __gm__ uint8_t * __gm__ *pagedKVCachesPtr = reinterpret_cast<__gm__ uint8_t* __gm__ *>(pagedKVCaches);

        // getting the right ptr to the paged kvcache layer
        __gm__ uint8_t *pagedLayerKVCaches = pagedKVCachesPtr[layerIdx];
        
        int64_t pagedIdxOffset = kvIdx * this->pageBuffSize_ * this->hiddenDims_ +
                                 slot * this->hiddenDims_;

        int64_t dstTensorIdxOffset = kvIdx * this->numLayers_ * this->numTokensChunk_  * this->hiddenDims_ +
                                     layerIdx * this->numTokensChunk_  * this->hiddenDims_ + 
                                     tokenIdx * this->hiddenDims_;

        this->pagedTokenGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(pagedLayerKVCaches) + pagedIdxOffset, 
                                                   this->hiddenDims_);
        this->lmcBufferGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(cacheTensor) + dstTensorIdxOffset, 
                                               this->hiddenDims_);
    }

    __aicore__ inline void processFunc() {
        if (!this->valid_) {
            return;
        }
        // 1. Alloc Tensor for local page
        local_scalar_t hiddenDimTensor = pagedTokenQue_.AllocTensor<scalar_t>();

        // 2. copy from global tensor into local
        if (this->page2L_) {
            AscendC::DataCopy(hiddenDimTensor, this->pagedTokenGlobal_, this->hiddenDims_);
        } else {
            AscendC::DataCopy(hiddenDimTensor, this->lmcBufferGlobal_, this->hiddenDims_);
        }
       
        // 3. enque vecin
        pagedTokenQue_.EnQue(hiddenDimTensor);
        // 4. deque vecin, possible to reuse due to QueBind
        hiddenDimTensor = pagedTokenQue_.DeQue<scalar_t>();

        // 5. datacopy into GM 
        if (this->page2L_) {
            AscendC::DataCopy(this->lmcBufferGlobal_, hiddenDimTensor, this->hiddenDims_);
        } else {
            AscendC::DataCopy(this->pagedTokenGlobal_, hiddenDimTensor, this->hiddenDims_);
        }
        
        // 6. free alloced Tensor
        pagedTokenQue_.FreeTensor(hiddenDimTensor);
    }


private:
    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4> pagedTokenQue_;

    // [layers * [kvs, numPages * pagedSize, heads*headsize]]
    AscendC::GlobalTensor<scalar_t> pagedTokenGlobal_;
    // [kvs, layers, numTokensChunk, heads*headsize]
    AscendC::GlobalTensor<scalar_t> lmcBufferGlobal_;
    int32_t numLayers_; // num layers
    int64_t pageBuffSize_; // pages * pageSize
    int64_t hiddenDims_; // heads * headSize
    int32_t numTokensChunk_; // num tokens in the cache tensor chunk
    bool valid_;
    bool page2L_; // true, from pagedTensor to LMC, false otherwise
};

// NOTE: there are potential micro optimizaiton here.
#define MULTI_LAYER_PAGED_KV_COPY_TYPE_DECLARE(TYPE, SLOTTYPE)                                                         \
        extern "C" __global__ __aicore__ void multi_layer_paged_kv_copy_##TYPE##_##SLOTTYPE(                           \
        __gm__ uint8_t* pagedKVCaches, __gm__ uint8_t* dstCacheTensor, __gm__ uint8_t* slotmappings,                   \
        const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,                                          \
        const int64_t pageBuffSize, const int32_t numTokensChunk, const int coreNum, const bool page2L)                \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        MultiLayerPagedKVCopy<TYPE, SLOTTYPE> op{};                                                                    \
        op.init(pagedKVCaches, dstCacheTensor, slotmappings, hiddenDims,                                               \
                numLayers, pageBuffSize, numTokensChunk, page2L, &pipe);                                               \
        int64_t bIdx = AscendC::GetBlockIdx();                                                                         \
        for (int64_t i = bIdx; i < numTokensChunk; i+=coreNum) {                                                       \
            for (int32_t kvIdx = 0; kvIdx < kvs; kvIdx ++) {                                                           \
                for (int32_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {                                         \
                    op.reset();                                                                                        \
                    op.updateMemOffset(pagedKVCaches, dstCacheTensor, slotmappings, i, kvIdx, layerIdx);               \
                    op.processFunc();                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

// Declare support kernel entry
MULTI_LAYER_PAGED_KV_COPY_TYPE_DECLARE(half, int32_t);
MULTI_LAYER_PAGED_KV_COPY_TYPE_DECLARE(half, int64_t);
MULTI_LAYER_PAGED_KV_COPY_TYPE_DECLARE(bfloat16_t, int32_t);
MULTI_LAYER_PAGED_KV_COPY_TYPE_DECLARE(bfloat16_t, int64_t);
MULTI_LAYER_PAGED_KV_COPY_TYPE_DECLARE(int8_t, int32_t);
MULTI_LAYER_PAGED_KV_COPY_TYPE_DECLARE(int8_t, int64_t);

namespace lmc {

#define MULTI_LAYER_PAGED_KV_COPY_KERNEL_CALL(TYPE, SLOTTYPE)                                                          \
    multi_layer_paged_kv_copy_##TYPE##_##SLOTTYPE<<<blockDim, nullptr, stream>>>(pagedKVCaches, dstCacheTensor,        \
                                                                                 slotmappings, hiddenDims, kvs,        \
                                                                                 numLayers, pageBuffSize,              \
                                                                                 numTokensChunk, blockDim, page2L);

template<typename T, typename SlotT>
void multi_layer_paged_kernel(uint32_t blockDim, void *stream, uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, 
                  uint8_t *slotmappings, const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers, 
                  const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L);

#define MULTI_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(TYPE, SLOTTYPE)                                                     \
template<>                                                                                                             \
void multi_layer_paged_kernel<TYPE, SLOTTYPE>(uint32_t blockDim, void *stream, uint8_t *pagedKVCaches,                 \
                                              uint8_t *dstCacheTensor, uint8_t *slotmappings,                          \
                                              const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,    \
                                              const int64_t pageBuffSize, const int32_t numTokensChunk,                \
                                              const bool page2L){                                                      \
    MULTI_LAYER_PAGED_KV_COPY_KERNEL_CALL(TYPE, SLOTTYPE);                                                             \
}

MULTI_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(half, int32_t);
MULTI_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(half, int64_t);
MULTI_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(bfloat16_t, int32_t);
MULTI_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(bfloat16_t, int64_t);
MULTI_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(int8_t, int32_t);
MULTI_LAYER_PAGED_KERNEL_CALL_TYPE_DECLARE(int8_t, int64_t);

template<typename T>
void dispatch_paged_kernel_on_slot_type(vllm_ascend::AscendType slotType, uint32_t blockDim, void *stream,
                                        uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, uint8_t *slotmappings, 
                                        const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,         
                                        const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L) {
    switch(slotType) {
        case vllm_ascend::AscendType::INT32:
            multi_layer_paged_kernel<T, int32_t>(blockDim, stream, pagedKVCaches, dstCacheTensor, slotmappings,
                                     hiddenDims, kvs, numLayers, pageBuffSize, numTokensChunk, page2L);
            break;
        case vllm_ascend::AscendType::INT64:
            multi_layer_paged_kernel<T, int64_t>(blockDim, stream, pagedKVCaches, dstCacheTensor, slotmappings,
                                     hiddenDims, kvs, numLayers, pageBuffSize, numTokensChunk, page2L);
            break;
        default:
            return;
    }
}

extern void multi_layer_kv_transfer_kernel(vllm_ascend::AscendType type, vllm_ascend::AscendType slotType, 
                                           uint32_t blockDim, void *stream, uint8_t *pagedKVCaches, 
                                           uint8_t *dstCacheTensor, uint8_t *slotmappings, 
                                           const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers, 
                                           const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L)
{
    switch(type) {
        case vllm_ascend::AscendType::FP16:
            dispatch_paged_kernel_on_slot_type<half>(slotType, blockDim, stream, pagedKVCaches, dstCacheTensor, 
                                                     slotmappings, hiddenDims, kvs, numLayers, pageBuffSize, 
                                                     numTokensChunk, page2L);
            break;
        case vllm_ascend::AscendType::BF16:
            dispatch_paged_kernel_on_slot_type<bfloat16_t>(slotType, blockDim, stream, pagedKVCaches, dstCacheTensor, 
                                                           slotmappings, hiddenDims, kvs, numLayers, pageBuffSize, 
                                                           numTokensChunk, page2L);
            break;
        case vllm_ascend::AscendType::INT8:
            dispatch_paged_kernel_on_slot_type<int8_t>(slotType, blockDim, stream, pagedKVCaches, dstCacheTensor, 
                                                        slotmappings, hiddenDims, kvs, numLayers, pageBuffSize, 
                                                        numTokensChunk, page2L);
            break;
        default:
            return;
    }
}

} // namespace lmc
