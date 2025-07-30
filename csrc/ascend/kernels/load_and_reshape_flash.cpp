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

template <typename scalar_t, typename slot_t> class LoadAndReshapeFlashCopy {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline LoadAndReshapeFlashCopy()
    {
    }

    __aicore__ inline void init(GM_ADDR cacheTensor, GM_ADDR keyCachePtr, GM_ADDR valueCachePtr, GM_ADDR slotmappings,
                                const int64_t numPages, const int64_t hiddenDims, const int32_t pagedSize,
                                const int32_t numTokens, const int32_t numLayers, const int32_t layerIdx, 
                                const bool page2L, AscendC::TPipe *pipe)
    {
        this->pipe_ = pipe;
        this->numPages_ = numPages;
        this->hiddenDims_ = hiddenDims;
        this->numTokens_ = numTokens;
        this->pagedSize_ = pagedSize;
        this->numLayers_ = numLayers;
        this->layerIdx_ = layerIdx;
        this->valid_ = true;
        this->page2L_ = page2L;

        // TODO: Not sure how many to allocate, but let's do 4 blocks of hiddenDims_
        // if it was fp16, 2048, we would get 16kb.？
        // should check whether hiddenDims_ is > 192KB.
        this->pipe_->InitBuffer(this->pagedTokenQue_, 4, this->hiddenDims_*sizeof(scalar_t));
    }

    __aicore__ inline void reset(){
        this->valid_ = true;
    }

    __aicore__ inline void updateTensorMemOffsetAndProcess(__gm__ uint8_t *pagedKeyTensor, 
                                                           __gm__ uint8_t *pagedValueTensor,
                                                           __gm__ uint8_t* nonPagedTensor, 
                                                           __gm__ uint8_t *slotmappings, const int tokenIdx) 
    {
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);
        int64_t slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);

        if (slot == -1) {
            this->valid_ = false;
            return;
        }

        // for the page tensor
        int64_t pagedIdxOffset = slot * this->hiddenDims_;
        
        // for the lmc tensor
        int64_t nonPagedKeyOffset = this->layerIdx_ * this->numTokens_ * this->hiddenDims_ +
                                    tokenIdx * this->hiddenDims_;

        // values are stored after keys in the non-paged tensor
        int64_t nonPagedValueOffset = this->numLayers_ * this->numTokens_  * this->hiddenDims_ +
                                      this->layerIdx_ * this->numTokens_  * this->hiddenDims_ + 
                                      tokenIdx * this->hiddenDims_;

        // keys
        this->keyTokensGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(pagedKeyTensor) + pagedIdxOffset,
                                                this->hiddenDims_);
        this->lmcBufferKeyGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(nonPagedTensor) + nonPagedKeyOffset,
                                                this->hiddenDims_);
        // values
        this->valueTokensGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(pagedValueTensor) + pagedIdxOffset,
                                                this->hiddenDims_);
        this->lmcBufferValueGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(nonPagedTensor) + nonPagedValueOffset,
                                                this->hiddenDims_);
    }

    __aicore__ inline void processFunc() {
        if (!this->valid_) {
            return;
        }
        // 1. Alloc Tensor for local page
        local_scalar_t hiddenKeysDimTensor = this->pagedTokenQue_.template AllocTensor<scalar_t>();
        local_scalar_t hiddenValuesDimTensor = this->pagedTokenQue_.template AllocTensor<scalar_t>();;

        // 2. copy from global tensor into local (GM -> UB)
        if (this->page2L_) {
            AscendC::DataCopy(hiddenKeysDimTensor, this->keyTokensGlobal_, this->hiddenDims_);
            AscendC::DataCopy(hiddenValuesDimTensor, this->valueTokensGlobal_, this->hiddenDims_);
        } else {
            AscendC::DataCopy(hiddenKeysDimTensor, this->lmcBufferKeyGlobal_, this->hiddenDims_);
            AscendC::DataCopy(hiddenValuesDimTensor, this->lmcBufferValueGlobal_, this->hiddenDims_);
        }
       
        // 3. enque vecin
        pagedTokenQue_.EnQue(hiddenKeysDimTensor);
        pagedTokenQue_.EnQue(hiddenValuesDimTensor);

        // 4. deque vecin, possible to reuse due to QueBind
        hiddenKeysDimTensor = pagedTokenQue_.DeQue<scalar_t>();
        hiddenValuesDimTensor = pagedTokenQue_.DeQue<scalar_t>();

        // 5. datacopy into GM 
        if (this->page2L_) {
            AscendC::DataCopy(this->lmcBufferKeyGlobal_, hiddenKeysDimTensor, this->hiddenDims_);
            AscendC::DataCopy(this->lmcBufferValueGlobal_, hiddenValuesDimTensor, this->hiddenDims_);
        } else {
            AscendC::DataCopy(this->keyTokensGlobal_, hiddenKeysDimTensor, this->hiddenDims_);
            AscendC::DataCopy(this->valueTokensGlobal_, hiddenValuesDimTensor, this->hiddenDims_);
        }
        // 6. free alloced Tensor
        pagedTokenQue_.FreeTensor(hiddenKeysDimTensor);
        pagedTokenQue_.FreeTensor(hiddenValuesDimTensor);
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4> pagedTokenQue_;

    // [numPages, pagedSize, heads*headsize]
    AscendC::GlobalTensor<scalar_t> keyTokensGlobal_;
    AscendC::GlobalTensor<scalar_t> valueTokensGlobal_;

    // Depends on LMC setting whether we store in tokensMajor or not.
    // the layout would be the followings:
    // [tokens, kvs, heads*headsize] or [kvs, tokens, heads*headsize]
    // TODO: check whether should combine the two and use a loop
    AscendC::GlobalTensor<scalar_t> lmcBufferKeyGlobal_;
    AscendC::GlobalTensor<scalar_t> lmcBufferValueGlobal_;

    int64_t numPages_; // num vllm npu blocks
    int32_t pagedSize_; // per npu block tokens
    int64_t hiddenDims_; // heads * headsize
    int32_t numTokens_; // num tokens in the cache tensor chunk
    int32_t numLayers_; // num layers in the cache tensor
    int32_t layerIdx_; // layer idx in the cache tensor
    bool valid_;
    bool page2L_; // true, from pagedTensor to LMC, false otherwise
};

#define LOAD_AND_RESHAPE_FLASH_COPY_TYPE_DECLARE(TYPE, SLOTTYPE)                                                       \
        extern "C" __global__ __aicore__ void load_and_reshape_flash_copy_##TYPE##_##SLOTTYPE(                         \
        __gm__ uint8_t* dstCacheTensor, __gm__ uint8_t* keyCachePtr, __gm__ uint8_t* valueCachePtr,                    \
        __gm__ uint8_t* slotmappings, const int64_t hiddenDims, const int64_t numPages, const int32_t pagedSize,       \
        const int32_t numTokens, const int32_t numLayers, const int32_t layerIdx, const bool page2L,                   \
        const int blockNum)                                                                                            \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        LoadAndReshapeFlashCopy<TYPE, SLOTTYPE> op{};                                                                  \
        op.init(dstCacheTensor, keyCachePtr, valueCachePtr, slotmappings, numPages, hiddenDims, pagedSize,             \
                numTokens, numLayers, layerIdx, page2L, &pipe);                                                        \
        int64_t bIdx = AscendC::GetBlockIdx();                                                                         \
        for (int64_t i = bIdx; i < numTokens; i+=blockNum)                                                             \
        {                                                                                                              \
            op.reset();                                                                                                \
            op.updateTensorMemOffsetAndProcess(keyCachePtr, valueCachePtr, dstCacheTensor, slotmappings, i);           \
            op.processFunc();                                                                                          \
        }                                                                                                              \
    }

// Declare support kernel entry
LOAD_AND_RESHAPE_FLASH_COPY_TYPE_DECLARE(half, int32_t);
LOAD_AND_RESHAPE_FLASH_COPY_TYPE_DECLARE(half, int64_t);
LOAD_AND_RESHAPE_FLASH_COPY_TYPE_DECLARE(bfloat16_t, int32_t);
LOAD_AND_RESHAPE_FLASH_COPY_TYPE_DECLARE(bfloat16_t, int64_t);
LOAD_AND_RESHAPE_FLASH_COPY_TYPE_DECLARE(int8_t, int32_t);
LOAD_AND_RESHAPE_FLASH_COPY_TYPE_DECLARE(int8_t, int64_t);

namespace lmc {

#define LOAD_AND_RESHAPE_FLASH_COPY_KERNEL_CALL(TYPE, SLOTTYPE)                                                        \
        load_and_reshape_flash_copy_##TYPE##_##SLOTTYPE<<<blockDim, nullptr, stream>>>(dstCacheTensor, keyCachePtr,    \
                                                    valueCachePtr, slotmappings, hiddenDims, numPages, pagedSize,      \
                                                    numTokens, numLayers, layerIdx, page2L, blockDim);

template<typename T, typename SlotT>
void load_and_reshape_kernel_call(uint32_t blockDim, void *stream, uint8_t *dstCacheTensor, uint8_t *keyCachePtr, 
                       uint8_t *valueCachePtr, uint8_t *slotmappings, const int64_t hiddenDims, const int64_t numPages, 
                       const int32_t pagedSize, const int32_t numTokens, const int32_t numLayers, 
                       const int32_t layerIdx, const bool page2L);


#define LOAD_AND_RESHAPE_KERNEL_CALL_TYPE_DECLARE(TYPE, SLOTTYPE)                                                      \
template<>                                                                                                             \
void load_and_reshape_kernel_call<TYPE, SLOTTYPE>(uint32_t blockDim, void *stream, uint8_t *dstCacheTensor,            \
                                                  uint8_t *keyCachePtr, uint8_t *valueCachePtr, uint8_t *slotmappings, \
                                                  const int64_t hiddenDims, const int64_t numPages,                    \
                                                  const int32_t pagedSize, const int32_t numTokens,                    \
                                                  const int32_t numLayers, const int32_t layerIdx,                     \
                                                  const bool page2L) {                                                 \
    LOAD_AND_RESHAPE_FLASH_COPY_KERNEL_CALL(TYPE, SLOTTYPE);                                                           \
}

LOAD_AND_RESHAPE_KERNEL_CALL_TYPE_DECLARE(half, int32_t);
LOAD_AND_RESHAPE_KERNEL_CALL_TYPE_DECLARE(half, int64_t);
LOAD_AND_RESHAPE_KERNEL_CALL_TYPE_DECLARE(bfloat16_t, int32_t);
LOAD_AND_RESHAPE_KERNEL_CALL_TYPE_DECLARE(bfloat16_t, int64_t);
LOAD_AND_RESHAPE_KERNEL_CALL_TYPE_DECLARE(int8_t, int32_t);
LOAD_AND_RESHAPE_KERNEL_CALL_TYPE_DECLARE(int8_t, int64_t);

template<typename T>
void dispatch_on_slot_type(vllm_ascend::AscendType slotType, uint32_t blockDim, void *stream, 
                           uint8_t *dstCacheTensor, uint8_t *keyCachePtr, uint8_t *valueCachePtr,
                           uint8_t *slotmappings, const int64_t hiddenDims, const int64_t numPages, 
                           const int32_t pagedSize, const int32_t numTokens, const int32_t numLayers,
                           const int32_t layerIdx, const bool page2L) {
    switch(slotType) {
        case vllm_ascend::AscendType::INT32:
            load_and_reshape_kernel_call<T, int32_t>(blockDim, stream, dstCacheTensor, keyCachePtr, valueCachePtr,
                                          slotmappings, hiddenDims, numPages, pagedSize, numTokens, numLayers, layerIdx,
                                          page2L);
            break;
        case vllm_ascend::AscendType::INT64:
            load_and_reshape_kernel_call<T, int64_t>(blockDim, stream, dstCacheTensor, keyCachePtr, valueCachePtr,
                                          slotmappings, hiddenDims, numPages, pagedSize, numTokens, numLayers, layerIdx,
                                          page2L);
            break;
        default:
            return;
    }
}

extern void load_and_reshape_flash_kernel(vllm_ascend::AscendType type, vllm_ascend::AscendType slotType,
                            uint32_t blockDim, void *stream, 
                            uint8_t *dstCacheTensor, uint8_t *keyCachePtr, uint8_t *valueCachePtr,
                            uint8_t *slotmappings, const int64_t hiddenDims, const int64_t numPages, 
                            const int32_t pagedSize, const int32_t numTokens, const int32_t numLayers,
                            const int32_t layerIdx, bool page2L)
{
    switch(type) {
        case vllm_ascend::AscendType::FP16:
            dispatch_on_slot_type<half>(slotType, blockDim, stream, dstCacheTensor, keyCachePtr, valueCachePtr, 
                                        slotmappings, hiddenDims, numPages, pagedSize, numTokens, numLayers, layerIdx,
                                        page2L);
            break;
        case vllm_ascend::AscendType::BF16:
            dispatch_on_slot_type<bfloat16_t>(slotType, blockDim, stream, dstCacheTensor, keyCachePtr, valueCachePtr, 
                                        slotmappings, hiddenDims, numPages, pagedSize, numTokens, numLayers, layerIdx,
                                        page2L);
            break;
        case vllm_ascend::AscendType::INT8:
            dispatch_on_slot_type<int8_t>(slotType, blockDim, stream, dstCacheTensor, keyCachePtr, valueCachePtr, 
                                        slotmappings, hiddenDims, numPages, pagedSize, numTokens, numLayers, layerIdx,
                                        page2L);
            break;
        default:
            return;
    }
}

} // namespace lmc
