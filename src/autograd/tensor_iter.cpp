#include "znet/autograd/tensor_iter.hpp"
#include <stdexcept>

namespace znet {

static std::vector<int> left_pad(const std::vector<int>& v, int ndim) {
    std::vector<int> r(ndim - static_cast<int>(v.size()), 1);
    r.insert(r.end(), v.begin(), v.end());
    return r;
}
static std::vector<int64_t> left_pad64(const std::vector<int>& v, int ndim) {
    std::vector<int64_t> r(ndim - static_cast<int>(v.size()), 0);
    r.insert(r.end(), v.begin(), v.end());
    return r;
}
static std::vector<int64_t> rowmajor_strides(const std::vector<int>& sizes) {
    std::vector<int64_t> s(sizes.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
        s[i] = stride;
        stride *= sizes[i];
    }
    return s;
}

TensorIter2 make_iter2_for_binary(const Tensor& a, const Tensor& b, const std::vector<int>& out_sizes) {
    TensorIter2 it;
    it.ndim = static_cast<int>(out_sizes.size());
    it.sizes = out_sizes;

    // align a/b sizes & strides to output rank
    auto a_sizes = left_pad(a.shape(), it.ndim);
    auto b_sizes = left_pad(b.shape(), it.ndim);

    auto a_str   = left_pad64(a.stride(), it.ndim);
    auto b_str   = left_pad64(b.stride(), it.ndim);

    // broadcasting check + stride fixups
    for (int d = 0; d < it.ndim; ++d) {
        int os = out_sizes[d];
        int as = a_sizes[d];
        int bs = b_sizes[d];

        // out dim must be the max of inputs’ dims (standard rule)
        int expect = (as == bs) ? as : (as == 1 ? bs : (bs == 1 ? as : -1));
        if (expect != os || expect < 0) {
            throw std::runtime_error("make_iter2_for_binary: incompatible sizes for broadcasting");
        }

        if (as == 1) a_str[d] = 0;  // broadcast along this dim
        if (bs == 1) b_str[d] = 0;
    }

    it.a_strides  = std::move(a_str);
    it.b_strides  = std::move(b_str);
    it.out_strides = rowmajor_strides(out_sizes);

    it.a_offset = a.storage_offset();
    it.b_offset = b.storage_offset();
    it.out_offset = 0; // new outputs usually start at 0

    return it;
}

static std::vector<int64_t> left_pad64_unary(const std::vector<int>& v, int ndim) {
    std::vector<int64_t> r(ndim - static_cast<int>(v.size()), 0);
    r.insert(r.end(), v.begin(), v.end());
    return r;
}

TensorIter1 make_iter1_for_unary(const Tensor& x, const std::vector<int>& out_sizes) {
    TensorIter1 it;
    it.ndim  = static_cast<int>(out_sizes.size());
    it.sizes = out_sizes;

    // For ReLU we expect exact shape match.
    if (x.shape() != out_sizes) {
        throw std::runtime_error("make_iter1_for_unary: out_sizes must equal input shape for ReLU");
    }

    auto x_str = left_pad64_unary(x.stride(), it.ndim);
    it.in_strides  = std::move(x_str);
    it.out_strides = rowmajor_strides(out_sizes);

    it.in_offset  = x.storage_offset();
    it.out_offset = 0;
    return it;
}

} // namespace znet
