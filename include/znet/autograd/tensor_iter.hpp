#pragma once
#include <vector>
#include <cstdint>
#include "znet/autograd/tensor.hpp"

namespace znet {

struct TensorIter2 {
    int ndim = 0;
    std::vector<int> sizes;          // output sizes [ndim]
    std::vector<int64_t> a_strides;  // aligned to ndim
    std::vector<int64_t> b_strides;  // aligned to ndim
    std::vector<int64_t> out_strides;// row-major for output
    int64_t a_offset = 0;
    int64_t b_offset = 0;
    int64_t out_offset = 0;          // (new outputs we allocate usually start at 0)
};

// Build iterator for two inputs. Throws on incompatible shapes.
// Notes: ranks are left-padded; size-1 dims broadcast with stride 0.
TensorIter2 make_iter2_for_binary(const Tensor& a, const Tensor& b, const std::vector<int>& out_sizes);



// --- Unary iterator for elementwise ops like ReLU ---
struct TensorIter1 {
    int ndim = 0;
    std::vector<int>     sizes;       // output (same as input for ReLU)
    std::vector<int64_t> in_strides;  // aligned to ndim
    std::vector<int64_t> out_strides; // row-major for output
    int64_t in_offset  = 0;
    int64_t out_offset = 0;           // new outputs usually start at 0
};

// Build iterator for one input. Out sizes must equal x.shape() for ReLU.
TensorIter1 make_iter1_for_unary(const Tensor& x, const std::vector<int>& out_sizes);

} // namespace znet
