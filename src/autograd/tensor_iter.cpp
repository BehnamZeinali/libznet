#include "znet/autograd/tensor_iter.hpp"
#include <stdexcept>
#include <cstdint>     // NEW
#include <vector>      // NEW
#include <limits>      // NEW (for optional overflow guards)

namespace znet {

// Pad sizes on the left with 1s up to ndim
static std::vector<int64_t> left_pad_sizes(const std::vector<int64_t>& v, int ndim) { // UPDATED
    if (ndim < 0) throw std::invalid_argument("left_pad_sizes: ndim < 0");            // NEW
    std::vector<int64_t> r(static_cast<size_t>(ndim) - v.size(), 1);                  // UPDATED
    r.insert(r.end(), v.begin(), v.end());
    return r;
}

// Pad strides on the left with 0s up to ndim (correct for broadcasted leading dims)
static std::vector<int64_t> left_pad_strides(const std::vector<int64_t>& v, int ndim) { // UPDATED
    if (ndim < 0) throw std::invalid_argument("left_pad_strides: ndim < 0");             // NEW
    std::vector<int64_t> r(static_cast<size_t>(ndim) - v.size(), 0);                     // UPDATED
    r.insert(r.end(), v.begin(), v.end());
    return r;
}

// Row-major contiguous strides for an output shape (int64_t)
static std::vector<int64_t> rowmajor_strides(const std::vector<int64_t>& sizes) { // UPDATED
    std::vector<int64_t> s(sizes.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
        s[static_cast<size_t>(i)] = stride;
        // Optional: overflow guard in debug
#ifndef NDEBUG
        if (sizes[static_cast<size_t>(i)] > 0 &&
            stride > (std::numeric_limits<int64_t>::max)() / sizes[static_cast<size_t>(i)]) {
            throw std::overflow_error("rowmajor_strides: stride overflow");
        }
#endif
        stride *= sizes[static_cast<size_t>(i)];
    }
    return s;
}

// === Binary iterator: broadcasting-aware (a, b -> out) =================
TensorIter2 make_iter2_for_binary(const Tensor& a,
                                  const Tensor& b,
                                  const std::vector<int64_t>& out_sizes) { // UPDATED
    TensorIter2 it;
    it.ndim  = static_cast<int>(out_sizes.size());
    it.sizes = out_sizes; // expect TensorIter2::sizes to be vector<int64_t>  // UPDATED

    // Align input sizes/strides to output rank
    auto a_sizes = left_pad_sizes(a.shape(), it.ndim);              // UPDATED
    auto b_sizes = left_pad_sizes(b.shape(), it.ndim);              // UPDATED
    auto a_str   = left_pad_strides(a.stride(), it.ndim);           // UPDATED
    auto b_str   = left_pad_strides(b.stride(), it.ndim);           // UPDATED

    // Broadcasting checks + stride zeroing along broadcasted dims
    for (int d = 0; d < it.ndim; ++d) {
        const int64_t os = out_sizes[static_cast<size_t>(d)];
        const int64_t as = a_sizes [static_cast<size_t>(d)];
        const int64_t bs = b_sizes [static_cast<size_t>(d)];

        // Output size must be the "max" per broadcast rules
        const int64_t expect = (as == bs) ? as : (as == 1 ? bs : (bs == 1 ? as : -1));
        if (expect != os || expect < 0) {
            throw std::runtime_error("make_iter2_for_binary: incompatible sizes for broadcasting");
        }

        if (as == 1) a_str[static_cast<size_t>(d)] = 0;  // broadcast a along this dim
        if (bs == 1) b_str[static_cast<size_t>(d)] = 0;  // broadcast b along this dim
    }

    it.a_strides   = std::move(a_str);                       // UPDATED
    it.b_strides   = std::move(b_str);                       // UPDATED
    it.out_strides = rowmajor_strides(out_sizes);            // UPDATED

    it.a_offset  = a.storage_offset();   // int64_t           // UPDATED
    it.b_offset  = b.storage_offset();   // int64_t           // UPDATED
    it.out_offset = 0;                   // new outputs start at 0

    return it;
}

// === Unary iterator: exact-shape (e.g., ReLU) ===========================
TensorIter1 make_iter1_for_unary(const Tensor& x,
                                 const std::vector<int64_t>& out_sizes) { // UPDATED
    TensorIter1 it;
    it.ndim  = static_cast<int>(out_sizes.size());
    it.sizes = out_sizes;  // expect exact shape match for unary ops       // UPDATED

    if (x.shape() != out_sizes) {
        throw std::runtime_error("make_iter1_for_unary: out_sizes must equal input shape");
    }

    auto x_str = left_pad_strides(x.stride(), it.ndim);  // no-op when ranks match // UPDATED
    it.in_strides  = std::move(x_str);                   // UPDATED
    it.out_strides = rowmajor_strides(out_sizes);        // UPDATED

    it.in_offset  = x.storage_offset();  // int64_t       // UPDATED
    it.out_offset = 0;
    return it;
}

} // namespace znet
