#pragma once
#include <znet/autograd/tensor.hpp>

namespace znet {

// Pure math. Never attaches grad_fn. Always returns requires_grad = false.
Tensor add_kernel(const Tensor& a, const Tensor& b);
Tensor matmul_kernel(const Tensor& a, const Tensor& b);
Tensor relu_kernel(const Tensor& input);
Tensor cross_entropy_kernel(const Tensor& logits, const Tensor& targets);

// C = A^T @ B
// A: [M, K], B: [M, N]  ->  C: [K, N]
Tensor matmul_AT_B_kernel(const Tensor& A, const Tensor& B);

// C = A @ B^T
// A: [M, K], B: [N, K]  ->  C: [M, N]
Tensor matmul_A_BT_kernel(const Tensor& A, const Tensor& B);

} // namespace znet
