#include "registration.h"
#include "pytorch_shim.h"
#include "namespace_config.h"

#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>

/**
 * SolAttn (V100) keep-or-drop sparse op (flash-attn official, Tri Dao).
 */

namespace FLASH_NAMESPACE {

std::vector<at::Tensor>
mha_varlen_fwd_sparse(at::Tensor &q,  // total_q x num_heads x head_size
                      const at::Tensor &k,  // total_k x num_heads x head_size
                      const at::Tensor &v,  // total_k x num_heads x head_size
                      std::optional<at::Tensor> &out_, // total_q x num_heads x head_size
                      const at::Tensor &cu_seqlens_q,  // b+1
                      const at::Tensor &cu_seqlens_k,  // b+1
                      const at::Tensor &block_count,   // [b*h, NUM_ROWS] int32
                      const at::Tensor &block_offset,  // [b*h, NUM_ROWS, NNZ_S] int32
                      const at::Tensor &column_count,  // [b*h, NUM_ROWS] int32
                      const at::Tensor &column_index,  // [b*h, NUM_ROWS, NNZ_V] int32
                      int max_seqlen_q,
                      const int max_seqlen_k,
                      const float softmax_scale);

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("varlen_fwd_sparse(Tensor! q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
            "int max_seqlen_q, int max_seqlen_k, float softmax_scale) -> Tensor[]");
    ops.impl("varlen_fwd_sparse", torch::kCUDA, make_pytorch_shim(&mha_varlen_fwd_sparse));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);

} // namespace FLASH_NAMESPACE
