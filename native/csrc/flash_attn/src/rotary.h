/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "namespace_config.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true, bool Clear_OOB_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_interleaved(Tensor<Engine0, Layout0> const &S,
                                               Tensor<Engine1, Layout1> &D,
                                               Tensor<Engine2, Layout2> const &Cos,
                                               Tensor<Engine2, Layout2> const &Sin,
                                               Tensor<Engine3, Layout3> const &identity_MN,
                                               const int max_MN, const int min_MN,
                                               const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    //CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    //CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));                     // MMA_K
    //static_assert(decltype(size<0>(S))::value == decltype(size<0>(Cos))::value * 2);
    //static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    using T = typename Engine0::value_type;
    using RotT = typename Engine2::value_type;
    cutlass::NumericConverter<float, T> to_float;
    cutlass::NumericConverter<float, RotT> rot_to_float;
    cutlass::NumericConverter<T, float> from_float;
    Tensor rS = make_fragment_like(S(_, 0, 0));
    Tensor rCos = make_fragment_like(Cos(_, 0, 0));
    Tensor rSin = make_fragment_like(Sin(_, 0, 0));

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        const int row_idx = get<0>(identity_MN(0, m, 0));
        if (row_idx >= min_MN && row_idx < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                const int col_idx = get<1>(identity_MN(0, 0, k));
                if constexpr (Is_even_K) {
                    auto s_tensor = S(_, m, k);
                    if (col_idx < rotary_dim) {
                        cute::copy(s_tensor, rS);
                        cute::copy(Cos(_, m, k), rCos);
                        cute::copy(Sin(_, m, k), rSin);
                        #pragma unroll
                        for (int i = 0; i < size(rCos); ++i) {
                            const float x0 = to_float(rS(2 * i));
                            const float x1 = to_float(rS(2 * i + 1));
                            const float c = rot_to_float(rCos(i));
                            const float s = rot_to_float(rSin(i));
                            rS(2 * i) = from_float(x0 * c - x1 * s);
                            rS(2 * i + 1) = from_float(x0 * s + x1 * c);
                        }
                        cute::copy(rS, D(_, m, k));
                    } else {
                        cute::copy(s_tensor, D(_, m, k));
                    }
                } else if (col_idx < dim) {
                    auto s_tensor = S(_, m, k);
                    if (col_idx < rotary_dim) {
                        cute::copy(s_tensor, rS);
                        cute::copy(Cos(_, m, k), rCos);
                        cute::copy(Sin(_, m, k), rSin);
                        #pragma unroll
                        for (int i = 0; i < size(rCos); ++i) {
                            const float x0 = to_float(rS(2 * i));
                            const float x1 = to_float(rS(2 * i + 1));
                            const float c = rot_to_float(rCos(i));
                            const float s = rot_to_float(rSin(i));
                            rS(2 * i) = from_float(x0 * c - x1 * s);
                            rS(2 * i + 1) = from_float(x0 * s + x1 * c);
                        }
                        cute::copy(rS, D(_, m, k));
                    } else {
                        cute::copy(s_tensor, D(_, m, k));
                    }
                } else if constexpr (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true, bool Clear_OOB_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_contiguous(Tensor<Engine0, Layout0> const &S,
                                              Tensor<Engine1, Layout1> &D,
                                              Tensor<Engine2, Layout2> const &Cos,
                                              Tensor<Engine2, Layout2> const &Sin,
                                              Tensor<Engine3, Layout3> const &identity_MN,
                                              const int max_MN, const int min_MN,
                                              const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    //CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    //CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    //CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(Cos));                     // MMA
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    using T = typename Engine0::value_type;
    using RotT = typename Engine2::value_type;
    cutlass::NumericConverter<float, T> to_float;
    cutlass::NumericConverter<float, RotT> rot_to_float;
    cutlass::NumericConverter<T, float> from_float;
    Tensor rS = make_fragment_like(S(_, 0, 0));
    Tensor rS_other = make_fragment_like(S(_, 0, 0));
    Tensor rCos = make_fragment_like(Cos(_, 0, 0));
    Tensor rSin = make_fragment_like(Sin(_, 0, 0));
    const int rotary_half = rotary_dim / 2;

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        const int row_idx = get<0>(identity_MN(0, m, 0));
        if (row_idx >= min_MN && row_idx < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                const int col_idx = get<1>(identity_MN(0, 0, k));
                if constexpr (Is_even_K) {
                    auto s_tensor = S(_, m, k);
                    if (col_idx < rotary_dim) {
                        const bool is_left = col_idx < rotary_half;
                        auto s_other_tensor = make_tensor(s_tensor.data() + (is_left ? rotary_half : -rotary_half), s_tensor.layout());
                        auto cos_tensor = make_tensor(Cos(_, m, k).data() + (is_left ? 0 : -rotary_half), Cos(_, m, k).layout());
                        auto sin_tensor = make_tensor(Sin(_, m, k).data() + (is_left ? 0 : -rotary_half), Sin(_, m, k).layout());
                        cute::copy(s_tensor, rS);
                        cute::copy(s_other_tensor, rS_other);
                        cute::copy(cos_tensor, rCos);
                        cute::copy(sin_tensor, rSin);
                        #pragma unroll
                        for (int i = 0; i < size(rS); ++i) {
                            const float x = to_float(rS(i));
                            const float other = to_float(rS_other(i));
                            const float c = rot_to_float(rCos(i));
                            const float s = rot_to_float(rSin(i));
                            rS(i) = from_float(x * c + other * (is_left ? -s : s));
                        }
                        cute::copy(rS, D(_, m, k));
                    } else {
                        cute::copy(s_tensor, D(_, m, k));
                    }
                } else if (col_idx < dim) {
                    auto s_tensor = S(_, m, k);
                    if (col_idx < rotary_dim) {
                        const bool is_left = col_idx < rotary_half;
                        auto s_other_tensor = make_tensor(s_tensor.data() + (is_left ? rotary_half : -rotary_half), s_tensor.layout());
                        auto cos_tensor = make_tensor(Cos(_, m, k).data() + (is_left ? 0 : -rotary_half), Cos(_, m, k).layout());
                        auto sin_tensor = make_tensor(Sin(_, m, k).data() + (is_left ? 0 : -rotary_half), Sin(_, m, k).layout());
                        cute::copy(s_tensor, rS);
                        cute::copy(s_other_tensor, rS_other);
                        cute::copy(cos_tensor, rCos);
                        cute::copy(sin_tensor, rSin);
                        #pragma unroll
                        for (int i = 0; i < size(rS); ++i) {
                            const float x = to_float(rS(i));
                            const float other = to_float(rS_other(i));
                            const float c = rot_to_float(rCos(i));
                            const float s = rot_to_float(rSin(i));
                            rS(i) = from_float(x * c + other * (is_left ? -s : s));
                        }
                        cute::copy(rS, D(_, m, k));
                    } else {
                        cute::copy(s_tensor, D(_, m, k));
                    }
                } else if constexpr (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
