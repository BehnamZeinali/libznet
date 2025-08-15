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

void add_kernel_strided(Tensor& out, const Tensor& a, const Tensor& b);

void relu_kernel_strided(Tensor& out, const Tensor& x);
void relu_backward_kernel_accum(Tensor& grad_in, const Tensor& x, const Tensor& grad_out);


void matmul_strided_batched_kernel(const Tensor& A, const Tensor& B, Tensor& C,
                                          bool A_logical_trans, bool B_logical_trans);
                                          
std::vector<int> broadcast_leading(const std::vector<int>& a, const std::vector<int>& b);
inline int64_t prod(const std::vector<int>& v);

std::vector<int> compute_matmul_out_shape_view(const Tensor& A, const Tensor& B);
std::vector<int> compute_mm_out_shape_flags(const Tensor& A, const Tensor& B,
                                                   bool transA, bool transB);

} // namespace znet
