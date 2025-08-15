#include "znet/autograd/ops_impl.hpp"
#include "znet/autograd/autograd_function.hpp"
#include "znet/autograd/grad_mode.hpp"
#include "znet/autograd/kernels.hpp"


namespace znet {

Tensor add_impl(const Tensor& a, const Tensor& b) {
   Tensor out = add_kernel(a, b);

    // 2) If grad mode is on AND any input requires grad, wire autograd
    if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        out.set_requires_grad(true);  // this output participates in autograd
        out.set_grad_fn(std::make_shared<AddFunction>(a, b)); // SavedTensor inside
    }
    return out;
}

Tensor matmul_impl(const Tensor& a, const Tensor& b) {
   Tensor out = matmul_kernel(a, b);

    // 2) If grad mode is on AND any input requires grad, wire autograd
    if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        out.set_requires_grad(true);  // this output participates in autograd
        out.set_grad_fn(std::make_shared<MatmulFunction>(a, b)); // SavedTensor inside
    }
    return out;
}

Tensor matmul_impl_(const Tensor& a, const Tensor& b) {
   Tensor out = matmul_A_BT_kernel(a, b);

    // 2) If grad mode is on AND any input requires grad, wire autograd
    if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        out.set_requires_grad(true);  // this output participates in autograd
        out.set_grad_fn(std::make_shared<MatmulFunction>(a, b)); // SavedTensor inside
    }
    return out;
}


Tensor relu_impl(const Tensor& input) {
    Tensor out = relu_kernel(input);
    if (GradMode::is_enabled() && (input.requires_grad() ))
        out.set_grad_fn(std::make_shared<ReLUFunction>(input));
        // out.set_grad_fn(std::make_shared<ReLUFunction>(input));
    return out;
}

Tensor cross_entropy_impl(const Tensor& logits, const Tensor& target) {
    Tensor out = cross_entropy_kernel(logits, target);
    if (GradMode::is_enabled() && logits.requires_grad()) 
        // out.set_grad_fn(std::make_shared<CrossEntropyFunction>(&logits, &target));
        out.set_grad_fn(std::make_shared<CrossEntropyFunction>(logits, target));
    return out;
}

// Tensor mul_impl(const Tensor& a, const Tensor& b) {
//     if (a.shape() != b.shape())
//         throw std::runtime_error("Shape mismatch in mul()");

//     std::vector<float> result(a.numel());
//     for (int i = 0; i < a.numel(); ++i)
//         result[i] = a.data()[i] * b.data()[i];

//     bool requires_grad = a.requires_grad() || b.requires_grad();
//     Tensor out(a.shape(), result, requires_grad);
//     if (requires_grad)
//         out.set_grad_fn(std::make_shared<MulFunction>(&a, &b));
//     return out;
// }

// Tensor softmax_impl(const Tensor& input) {
//     const auto& shape = input.shape();
//     if (shape.size() != 2)
//         throw std::runtime_error("softmax currently supports only 2D tensors (batch, classes).");

//     int batch = shape[0];
//     int classes = shape[1];
//     const auto& data = input.data();
//     std::vector<float> result(data.size());

//     for (int i = 0; i < batch; ++i) {
//         float max_logit = -std::numeric_limits<float>::infinity();
//         for (int j = 0; j < classes; ++j)
//             max_logit = std::max(max_logit, data[i * classes + j]);

//         float sum_exp = 0.0f;
//         for (int j = 0; j < classes; ++j) {
//             float val = std::exp(data[i * classes + j] - max_logit);
//             result[i * classes + j] = val;
//             sum_exp += val;
//         }

//         for (int j = 0; j < classes; ++j)
//             result[i * classes + j] /= sum_exp;
//     }

//     bool requires_grad = input.requires_grad();
//     Tensor out(shape, result, requires_grad);
//     if (requires_grad)
//         out.set_grad_fn(std::make_shared<SoftmaxFunction>(&input));
//     return out;
// }

} // namespace znet
