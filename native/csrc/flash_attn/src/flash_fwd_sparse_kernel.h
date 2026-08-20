/******************************************************************************
 * Copyright (c) 2024, PAI, Alibaba Cloud.
 ******************************************************************************/

#pragma once

#include "flash_fwd_kernel.h"

namespace FLASH_NAMESPACE {

using namespace cute;

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void sparse_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kWarpRows = kBlockM / kNWarps;
    constexpr int kBlockRowStride = kBlockM / kWarpRows;
    const int lane_id = tidx % 32;
    const int warp_id = tidx / 32;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    FLASH_NAMESPACE::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    const int rows_valid = binfo.actual_seqlen_q - m_block * kBlockM;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    const index_t row_offset_k_token =
        binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v_token =
        binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gKToken = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k_token),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_stride(_0{}, _1{}));
    Tensor gVToken = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v_token),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_stride(_0{}, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
        
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    constexpr int kSmemQElems = decltype(cute::cosize(typename Kernel_traits::SmemLayoutQ{}))::value;
    constexpr int kSmemKVElems = decltype(cute::cosize(typename Kernel_traits::SmemLayoutKV{}))::value;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : kSmemQElems),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + kSmemKVElems, typename Kernel_traits::SmemLayoutKV{});
    Tensor sP = make_tensor(sV.data() + kSmemKVElems, typename Kernel_traits::SmemLayoutP{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sP_warp = local_tile(sP, Shape<Int<kWarpRows>, Int<kBlockN>>{},
                                make_coord(warp_id, 0));

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKgKBlock = tKgK(_, _, _, 0);
    auto tKgKBlockData = tKgKBlock.data();
    Tensor tKgKToken = gmem_thr_copy_QKV.partition_S(gKToken);  // (KCPY, KCPY_N, KCPY_K)
    auto tKgKTokenData = tKgKToken.data();
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVgVBlock = tVgV(_, _, _, 0);
    auto tVgVBlockData = tVgVBlock.data();
    Tensor tVgVToken = gmem_thr_copy_QKV.partition_S(gVToken);  // (VCPY, VCPY_N, VCPY_K)
    auto tVgVTokenData = tVgVToken.data();
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    Tensor sQ_warp = local_tile(sQ, Shape<Int<kWarpRows>, Int<kHeadDim>>{},
                                make_coord(warp_id, 0));

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(lane_id);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ_warp);                      // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOsQ  = thr_mma.partition_A(sQ_warp);                               // (MMA,MMA_M,MMA_K)
    Tensor tOsK  = thr_mma.partition_B(sK);                                    // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)
    Tensor tOsVtWarp = thr_mma.partition_B(sVt);                               // (MMA, MMA_K,MMA_N)

    Tensor cS = make_identity_tensor(Shape<Int<kWarpRows>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kWarpRows>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    Tensor caccO = make_identity_tensor(Shape<Int<kWarpRows>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(lane_id);
    Tensor tSsQ = smem_thr_copy_Q.retile_S(tOsQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(lane_id);
    Tensor tSsK = smem_thr_copy_K.retile_S(tOsK);
    Tensor tSrQ_from_smem_dbg2 = smem_thr_copy_Q.retile_D(tSrQ);
    Tensor tSrK_from_smem_dbg2 = smem_thr_copy_K.retile_D(tSrK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(lane_id);
    Tensor tOsVt = smem_thr_copy_V.retile_S(tOsVtWarp);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);

    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if constexpr (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    // block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    // num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    // blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    int num_blks = params.block_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    auto* blks_ptr = params.block_offset + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_S;
  
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    FLASH_NAMESPACE::Softmax<kWarpRows> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    FLASH_NAMESPACE::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    // column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    // column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    // num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    // cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V
    int num_cols = params.column_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    int num_cols_block = (num_cols + kBlockN - 1)/ kBlockN;
    if (num_blks <= 0 && num_cols_block <= 0) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    if (num_blks > 0) {
        int block_index = num_blks - 1;
        // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
        tKgKBlock.data() = tKgKBlockData + blks_ptr[block_index] * int64_t(params.k_row_stride);
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgKBlock, tKsK, tKVcKV, tKVpKV,
                                        binfo.actual_seqlen_k - blks_ptr[block_index]);
        for (int n = 0; n < n_masking_steps && block_index >= 0; ++n, --block_index) {
            int start_n = blks_ptr[block_index];  // replace n_block * kBlockN

            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kWarpRows>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            __syncthreads();

            // Advance gV
            tVgVBlock.data() = tVgVBlockData + start_n * int64_t(params.v_row_stride);
            if (block_index < num_blks - 1) {
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgVBlock, tVsV, tKVcKV, tKVpKV);
            } else {
                // Clear the smem tiles to account for predicated off loads
                FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_QKV, tVgVBlock, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - start_n
                );
            }

            FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            // if (cute::thread0()) { print(acc_s); }
            if constexpr (Is_softcap){
                FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
            }

            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s, start_n, m_block * kBlockM + warp_id * kWarpRows, 0
            );

            __syncthreads();
            if (block_index > 0) {
                tKgKBlock.data() = tKgKBlockData + blks_ptr[block_index - 1] * int64_t(params.k_row_stride);
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgKBlock, tKsK, tKVcKV, tKVpKV);
            }
            // TODO: when we have key_padding_mask we'll need to Check_inf
            n == 0
                ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
                : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

            // Convert acc_s from fp32 to fp16
            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            int block_row_idx = m_block * kBlockRowStride + warp_id;
            int block_col_idx = n_block * (kBlockN / 32);
            if (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kBlockRowStride
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if (Is_dropout) {
                dropout.apply_dropout(rP, block_row_idx, block_col_idx, kBlockRowStride);
            }

            auto tOrP = FLASH_NAMESPACE::convert_layout_C_to_A_v2<Kernel_traits>(
                thr_mma, sP_warp, rP, smem_thr_copy_Q, lane_id
            );
            // if (cute::thread0()) { print(tOrP); }
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            // if (cute::thread0()) { print(scores); }
        }
        for (; block_index >= 0; --block_index) {
            int start_n = blks_ptr[block_index];  // replace n_block * kBlockN

            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kWarpRows>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            __syncthreads();
            // Advance gV
            tVgVBlock.data() = tVgVBlockData + start_n * int64_t(params.v_row_stride);
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgVBlock, tVsV, tKVcKV, tKVpKV);

            FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            if constexpr (Is_softcap){
                FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
            }

            __syncthreads();
            if (block_index > 0) {
                tKgKBlock.data() = tKgKBlockData + blks_ptr[block_index - 1] * int64_t(params.k_row_stride);
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgKBlock, tKsK, tKVcKV, tKVpKV);
            }

            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

            // Convert acc_s from fp32 to fp16
            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            int block_row_idx = m_block * kBlockRowStride + warp_id;
            int block_col_idx = n_block * (kBlockN / 32);
            if (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kBlockRowStride
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if (Is_dropout) {
                dropout.apply_dropout(rP, block_row_idx, block_col_idx, kBlockRowStride);
            }

            auto tOrP = FLASH_NAMESPACE::convert_layout_C_to_A_v2<Kernel_traits>(
                thr_mma, sP_warp, rP, smem_thr_copy_Q, lane_id
            );
            // if (cute::thread0()) { print(tOrP); }
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            // if (cute::thread0()) { print(scores); }
        }
    }
     
    if (num_cols > 0) {
        auto* cols_ptr = params.column_index + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_V;
        // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
        #pragma unroll
        for (int m = 0; m < size<1>(tKgKToken); ++m) {
            if (Is_even_MN || get<0>(tKVcKV(0, m, 0)) < num_cols) {  // Is_even_MN
                tKgKToken.data() = tKgKTokenData + cols_ptr[get<0>(tKVcKV(0, m, 0))] * int64_t(params.k_row_stride);
                #pragma unroll
                for (int k = 0; k < size<2>(tKgKToken); ++k) {
                    if (Is_even_K || tKVpKV(k)) {
                        cute::copy(gmem_tiled_copy_QKV, tKgKToken(_, m, k), tKsK(_, m, k));
                    } else {  // Clear_OOB_K
                        cute::clear(tKsK(_, m, k));
                    }
                }
            }
        }
        for (int n = 0; n < num_cols_block; ++n) {
            // cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
            // int start_n = cols_ptr[n * kBlockN];  // replace n_block * kBlockN

            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kWarpRows>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            __syncthreads();

            // Advance gV
            if (n < num_cols_block - 1) {
                #pragma unroll
                for (int m = 0; m < size<1>(tVgVToken); ++m) {
                    tVgVToken.data() = tVgVTokenData + cols_ptr[n * kBlockN + get<0>(tKVcKV(0, m, 0))] * int64_t(params.v_row_stride);
                    #pragma unroll
                    for (int k = 0; k < size<2>(tVgVToken); ++k) {
                        if (Is_even_K || tKVpKV(k)) {
                            cute::copy(gmem_tiled_copy_QKV, tVgVToken(_, m, k), tVsV(_, m, k));
                        } else {  // Clear_OOB_K
                            cute::clear(tVsV(_, m, k));
                        }
                    }
                }
            } else {
                // Clear the smem tiles to account for predicated off loads
                #pragma unroll
                for (int m = 0; m < size<1>(tVgVToken); ++m) {
                    if (Is_even_MN || n * kBlockN + get<0>(tKVcKV(0, m, 0)) < num_cols) {  // Is_even_MN
                        tVgVToken.data() = tVgVTokenData + cols_ptr[n * kBlockN + get<0>(tKVcKV(0, m, 0))] * int64_t(params.v_row_stride);
                        #pragma unroll
                        for (int k = 0; k < size<2>(tVgVToken); ++k) {
                            if (Is_even_K || tKVpKV(k)) {
                                cute::copy(gmem_tiled_copy_QKV, tVgVToken(_, m, k), tVsV(_, m, k));
                            } else {  // Clear_OOB_K
                                cute::clear(tVsV(_, m, k));
                            }
                        }
                    } else {  // Clear_OOB_MN
                        cute::clear(tVsV(_, m, _));
                    }
                }
            }

            FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            // if (cute::thread0()) { print(acc_s); }
            if constexpr (Is_softcap){
                FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
            }

            // Masking phase - optimized for V100 (SM70) with 8-fragment accumulator layout
            // Pre-compute constants outside the loop to reduce register pressure
            if (n >= num_cols_block - n_masking_steps || Is_causal || num_blks > 0) {
                const int max_seqlen_k = binfo.actual_seqlen_k;
                const int max_seqlen_q = binfo.actual_seqlen_q;
                const int row_base = m_block * kBlockM;
                const int col_base = n * kBlockN;
                const int causal_offset = max_seqlen_k - max_seqlen_q;

                #pragma unroll
                for (int i = 0; i < size(acc_s); ++i) {
                    const int row = get<0>(tScS(i));
                    const int col = get<1>(tScS(i));
                    const int row_idx = row_base + row;
                    const int col_list_idx = col_base + col;

                    // Phase 1: Boundary masking (row/column out of bounds)
                    bool mask_out = (row_idx >= max_seqlen_q) | (col_list_idx >= num_cols);

                    // Load token index only if not already masked
                    int token_idx = 0;
                    if (!mask_out) [[likely]] {
                        token_idx = cols_ptr[col_list_idx];
                        mask_out = (token_idx < 0) | (token_idx >= max_seqlen_k);
                    }

                    // Phase 2: Causal masking
                    if constexpr (Is_causal) {
                        if (!mask_out) [[likely]] {
                            // token_idx must be < min(max_seqlen_k, row_idx + 1 + causal_offset)
                            const int col_idx_limit = std::min(max_seqlen_k, row_idx + 1 + causal_offset);
                            mask_out = token_idx >= col_idx_limit;
                        }
                    }

                    // Phase 3: Sparse block masking (mask tokens already computed in dense blocks)
                    if (!mask_out && num_blks > 0) [[likely]] {
                        bool covered_by_block = false;
                        #pragma unroll 4
                        for (int bi = 0; bi < num_blks; ++bi) {
                            const int blk_start = blks_ptr[bi];
                            // Early skip invalid blocks using bitwise or for efficiency
                            if ((blk_start < 0) | (blk_start >= max_seqlen_k)) continue;
                            // Check if token falls within [blk_start, blk_start + kBlockN)
                            const int blk_end = blk_start + kBlockN;
                            covered_by_block = (token_idx >= blk_start) & (token_idx < blk_end);
                            if (covered_by_block) break;
                        }
                        mask_out |= covered_by_block;
                    }

                    // Apply mask
                    if (mask_out) [[unlikely]] {
                        acc_s(i) = -INFINITY;
                    }
                }
            }

            __syncthreads();
            if (n < num_cols_block - 2) {
                #pragma unroll
                for (int m = 0; m < size<1>(tKgKToken); ++m) {
                    int token_idx = cols_ptr[(n + 1) * kBlockN + get<0>(tKVcKV(0, m, 0))];
                    tKgKToken.data() = tKgKTokenData + token_idx * int64_t(params.k_row_stride);
                    #pragma unroll
                    for (int k = 0; k < size<2>(tKgKToken); ++k) {
                        if (Is_even_K || tKVpKV(k)) {
                            cute::copy(gmem_tiled_copy_QKV, tKgKToken(_, m, k), tKsK(_, m, k));
                        } else {  // Clear_OOB_K
                            cute::clear(tKsK(_, m, k));
                        }
                    }
                }
            } else if (n == num_cols_block - 2) {
                #pragma unroll
                for (int m = 0; m < size<1>(tKgKToken); ++m) {
                    if (Is_even_MN || (n + 1) * kBlockN + get<0>(tKVcKV(0, m, 0)) < num_cols) {  // Is_even_MN
                        int token_idx = cols_ptr[(n + 1) * kBlockN + get<0>(tKVcKV(0, m, 0))];
                        tKgKToken.data() = tKgKTokenData + token_idx * int64_t(params.k_row_stride);
                        #pragma unroll
                        for (int k = 0; k < size<2>(tKgKToken); ++k) {
                            if (Is_even_K || tKVpKV(k)) {
                                cute::copy(gmem_tiled_copy_QKV, tKgKToken(_, m, k), tKsK(_, m, k));
                            } else {  // Clear_OOB_K
                                cute::clear(tKsK(_, m, k));
                            }
                        }
                    }
                }
            }

            // TODO: when we have key_padding_mask we'll need to Check_inf
            (num_blks <= 0 && n ==0)
                ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
                : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

            // Convert acc_s from fp32 to fp16
            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            int block_row_idx = m_block * kBlockRowStride + warp_id;
            int block_col_idx = n_block * (kBlockN / 32);
            if (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kBlockRowStride
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if (Is_dropout) {
                dropout.apply_dropout(rP, block_row_idx, block_col_idx, kBlockRowStride);
            }

            auto tOrP = FLASH_NAMESPACE::convert_layout_C_to_A_v2<Kernel_traits>(
                thr_mma, sP_warp, rP, smem_thr_copy_Q, lane_id
            );
            // if (cute::thread0()) { print(tOrP); }
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            // if (cute::thread0()) { print(scores); }
        }
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (kBlockM, kHeadDim)

    // 同步，确保 Q 不再被使用
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    // 将 rO 写入 sO，使用 taccOcO 的坐标
    const int rows_this_block = binfo.actual_seqlen_q - m_block * kBlockM;
    const int warp_row_base = warp_id * kWarpRows;
    #pragma unroll
    for (int i = 0; i < size(rO); ++i) {
        const int row_local = get<0>(taccOcO(i));
        const int col = get<1>(taccOcO(i));
        const int row_global = warp_row_base + row_local;
        if (row_global < binfo.actual_seqlen_q - m_block * kBlockM && (Is_even_K || col < params.d)) {
            sO(warp_row_base + row_local, col) = rO(i);
        }
    }

    // Global memory output tensor
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::SmemTiledCopyOToReg smem_tiled_copy_O_to_reg;
    auto smem_thr_copy_O_to_reg = smem_tiled_copy_O_to_reg.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = smem_thr_copy_O_to_reg.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(smem_tiled_copy_O_to_reg, tOsO, tOrO);

    // LSE 写入 - 使用 SM70 特殊的布局转换
    Tensor taccOcO_logical = make_tensor(taccOcO.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(taccOcO.layout()));
    Tensor taccOcO_row = taccOcO_logical(_, 0);

    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row_global = warp_row_base + get<0>(taccOcO_row(mi));
            if (row_global < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row_global) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_sparse_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    FLASH_NAMESPACE::sparse_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block);
}

} // namespace FLASH_NAMESPACE
