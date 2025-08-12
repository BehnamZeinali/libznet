// include/znet/autograd/ops_impl.hpp
#pragma once
#include <znet/autograd/tensor.hpp>

namespace znet {

// Backend implementations using intrusive pointer-based Tensor
Tensor add_impl(const Tensor& a, const Tensor& b);
Tensor matmul_impl(const Tensor& a, const Tensor& b);
Tensor matmul_impl_(const Tensor& a, const Tensor& b);
Tensor relu_impl(const Tensor& input);
Tensor cross_entropy_impl(const Tensor& logits, const Tensor& targets);
Tensor mul_impl(const Tensor& a, const Tensor& b);
Tensor softmax_impl(const Tensor& input);

} // namespace znet
