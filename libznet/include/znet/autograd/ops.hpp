#pragma once
#include <znet/autograd/tensor.hpp>
#include <znet/autograd/ops_impl.hpp>

namespace znet {

// ======== ADD ========
inline Tensor add(const Tensor& a, const Tensor& b) {
    return add_impl(a, b);
}

// ======== MATMUL ========
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    return matmul_impl(a, b);
}

// ======== RELU ========
inline Tensor relu(const Tensor& a) {
    return relu_impl(a);
}

// ======== CROSS ENTROPY ========
inline Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    return cross_entropy_impl(logits, targets);
}

// ======== MUL ========
inline Tensor mul(const Tensor& a, const Tensor& b) {
    return mul_impl(a, b);
}

// ======== SOFTMAX ========
inline Tensor softmax(const Tensor& a) {
    return softmax_impl(a);
}

} // namespace znet
