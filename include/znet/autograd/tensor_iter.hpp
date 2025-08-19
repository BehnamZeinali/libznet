#pragma once
#include <vector>
#include <cstdint>
#include "znet/autograd/tensor.hpp"

namespace znet {

// ===== Binary iterator: a ⊙ b → out =====
struct TensorIter2 {
    int                       ndim = 0;
    std::vector<int64_t>      sizes;        // output sizes [ndim]  // UPDATED to int64_t
    std::vector<int64_t>      a_strides;    // aligned to ndim
    std::vector<int64_t>      b_strides;    // aligned to ndim
    std::vector<int64_t>      out_strides;  // row-major for output
    int64_t                   a_offset = 0;
    int64_t                   b_offset = 0;
    int64_t                   out_offset = 0; // new outputs start at 0
};

// Primary signature (use this going forward)                           // UPDATED
TensorIter2 make_iter2_for_binary(const Tensor& a,
                                  const Tensor& b,
                                  const std::vector<int64_t>& out_sizes);

// Compatibility wrapper for legacy int-sized callers (optional)        // NEW
inline TensorIter2 make_iter2_for_binary(const Tensor& a,
                                         const Tensor& b,
                                         const std::vector<int>& out_sizes32) {
    std::vector<int64_t> out_sizes(out_sizes32.begin(), out_sizes32.end());
    return make_iter2_for_binary(a, b, out_sizes);
}

// ===== Unary iterator: x ⊙ → out (e.g., ReLU) =====
struct TensorIter1 {
    int                       ndim = 0;
    std::vector<int64_t>      sizes;        // output sizes [ndim]  // UPDATED to int64_t
    std::vector<int64_t>      in_strides;   // aligned to ndim
    std::vector<int64_t>      out_strides;  // row-major for output
    int64_t                   in_offset  = 0;
    int64_t                   out_offset = 0; // new outputs start at 0
};

// Primary signature (use this going forward)                           // UPDATED
TensorIter1 make_iter1_for_unary(const Tensor& x,
                                 const std::vector<int64_t>& out_sizes);

// Compatibility wrapper for legacy int-sized callers (optional)        // NEW
inline TensorIter1 make_iter1_for_unary(const Tensor& x,
                                        const std::vector<int>& out_sizes32) {
    std::vector<int64_t> out_sizes(out_sizes32.begin(), out_sizes32.end());
    return make_iter1_for_unary(x, out_sizes);
}

} // namespace znet
