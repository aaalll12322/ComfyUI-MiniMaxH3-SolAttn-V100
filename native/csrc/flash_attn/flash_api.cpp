/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState
#include "philox_unpack.cuh"  // For at::cuda::philox::unpack

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_sparse.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


namespace FLASH_NAMESPACE {

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

std::vector<at::Tensor>
mha_varlen_fwd_sparse(at::Tensor &q,  // total_q x num_heads x head_size
                      const at::Tensor &k,  // total_k x num_heads x head_size
                      const at::Tensor &v,  // total_k x num_heads x head_size
                      std::optional<at::Tensor> &out_,  // total_q x num_heads x head_size
                      const at::Tensor &cu_seqlens_q,  // b+1
                      const at::Tensor &cu_seqlens_k,  // b+1
                      const at::Tensor &block_count,   // [b*h, NUM_ROWS] int32
                      const at::Tensor &block_offset,  // [b*h, NUM_ROWS, NNZ_S] int32
                      const at::Tensor &column_count,  // [b*h, NUM_ROWS] int32
                      const at::Tensor &column_index,  // [b*h, NUM_ROWS, NNZ_V] int32
                      int max_seqlen_q,
                      const int max_seqlen_k,
                      const float softmax_scale) {

    at::cuda::CUDAGuard device_guard{q.device()};

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16, "FlashAttention sparse only support fp16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);
    CHECK_DEVICE(block_count); CHECK_DEVICE(block_offset);
    CHECK_DEVICE(column_count); CHECK_DEVICE(column_index);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_k);
    CHECK_CONTIGUOUS(block_count); CHECK_CONTIGUOUS(block_offset);
    CHECK_CONTIGUOUS(column_count); CHECK_CONTIGUOUS(column_index);

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = q.size(1);
    const int head_size = q.size(2);
    TORCH_CHECK(head_size == 128, "Sparse forward only supports head_dim=128");
    const int total_q = q.size(0);

    const int total_k = k.size(0);
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads, head_size);
    CHECK_SHAPE(v, total_k, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    // kBlockM=64 for head_dim=128 sparse kernel -> NUM_ROWS = ceil(max_seqlen_q / 64)
    const int kBlockM = 64;
    const int NUM_ROWS = (max_seqlen_q + kBlockM - 1) / kBlockM;
    const int NNZ_S = block_offset.size(-1);
    const int NNZ_V = column_index.size(-1);
    TORCH_CHECK(block_count.numel() == batch_size * num_heads * NUM_ROWS,
                "block_count must have numel batch*heads*NUM_ROWS");
    TORCH_CHECK(block_offset.numel() == batch_size * num_heads * NUM_ROWS * NNZ_S,
                "block_offset must have numel batch*heads*NUM_ROWS*NNZ_S");
    TORCH_CHECK(column_count.numel() == batch_size * num_heads * NUM_ROWS,
                "column_count must have numel batch*heads*NUM_ROWS");
    TORCH_CHECK(column_index.numel() == batch_size * num_heads * NUM_ROWS * NNZ_V,
                "column_index must have numel batch*heads*NUM_ROWS*NNZ_V");

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        CHECK_SHAPE(out, total_q, num_heads, head_size);
    } else {
        out = torch::empty_like(q);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    out.zero_();
    softmax_lse.fill_(-std::numeric_limits<float>::infinity());

    Flash_fwd_params_sparse params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     0.0,
                     softmax_scale,
                     -1,
                     -1,
                     0.0,
                     /*seqlenq_ngroups_swapped*/false,
                     /*unpadded_lse*/true);
    params.total_q = total_q;
    params.block_count = block_count.data_ptr<int>();
    params.block_offset = block_offset.data_ptr<int>();
    params.column_count = column_count.data_ptr<int>();
    params.column_index = column_index.data_ptr<int>();
    params.NUM_ROWS = NUM_ROWS;
    params.NNZ_S = NNZ_S;
    params.NNZ_V = NNZ_V;

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd_sparse_<128, false>(params, stream);
    }

    return {out, softmax_lse};
}

} // namespace FLASH_NAMESPACE
