// #pragma once

// #include <znet/autograd/functional.hpp>

// namespace znet {
// namespace autograd {

// class MatmulFunction : public Function {
// public:
//     std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override {
//         auto A = saved_tensors[0];
//         auto B = saved_tensors[1];

//         // dL/dA = grad_output @ B^T
//         // dL/dB = A^T @ grad_output
//         auto dA = std::make_shared<Tensor>(grad_output->matmul(B->transpose(-2,-1)));
//         auto dB = std::make_shared<Tensor>(A->transpose(-2,-1).matmul(*grad_output));
//         return {dA, dB};
//     }
// };

// } // namespace autograd
// } // namespace znet
