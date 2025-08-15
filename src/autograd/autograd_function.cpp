// #include <znet/autograd/autograd_function.hpp>
// #include <znet/autograd/ops.hpp>

// #include <limits>
// #include <cmath>
// #include <algorithm>

// namespace znet {

// // ====== Base Function ======
// Function::Function() = default;
// Function::~Function() = default;

// // ====== Add ======
// AddFunction::AddFunction(const Tensor* l, const Tensor* r)
//     : lhs(l), rhs(r) {}

// std::vector<Tensor*> AddFunction::inputs() const {
//     return { const_cast<Tensor*>(lhs), const_cast<Tensor*>(rhs) };
// }

// void AddFunction::backward(Tensor& grad_out) {
//     if (lhs && lhs->requires_grad()) {
//         const_cast<Tensor*>(lhs)->accumulate_grad(grad_out);
//     }

//     if (rhs && rhs->requires_grad()) {
//         const auto& rhs_shape = rhs->shape();
//         const auto& grad_shape = grad_out.shape();

//         if (rhs_shape == grad_shape) {
//             const_cast<Tensor*>(rhs)->accumulate_grad(grad_out);
//         } else if (grad_shape.size() == 2 && rhs_shape.size() == 1 && grad_shape[1] == rhs_shape[0]) {
//             int B = grad_shape[0];
//             int F = grad_shape[1];
//             std::vector<float> reduced(F, 0.0f);
//             for (int i = 0; i < B; ++i)
//                 for (int j = 0; j < F; ++j)
//                     reduced[j] += grad_out.data()[i * F + j];

//             Tensor reduced_tensor(rhs_shape, reduced, true);
//             const_cast<Tensor*>(rhs)->accumulate_grad(reduced_tensor);
//         } else {
//             throw std::runtime_error("Unsupported broadcast in AddFunction::backward()");
//         }
//     }
// }

// // ====== Matmul ======
// MatmulFunction::MatmulFunction(const Tensor* l, const Tensor* r)
//     : lhs(l), rhs(r) {}

// std::vector<Tensor*> MatmulFunction::inputs() const {
//     return { const_cast<Tensor*>(lhs), const_cast<Tensor*>(rhs) };
// }

// void MatmulFunction::backward(Tensor& grad_out) {
//     if (lhs->requires_grad()) {
//         Tensor rhs_T = rhs->transpose(0, 1);
//         Tensor grad_lhs = matmul_impl(grad_out, rhs_T);
//         const_cast<Tensor*>(lhs)->accumulate_grad(grad_lhs);
//     }

//     if (rhs->requires_grad()) {
//         Tensor lhs_T = lhs->transpose(0, 1);
//         Tensor grad_rhs = matmul_impl(lhs_T, grad_out);
//         const_cast<Tensor*>(rhs)->accumulate_grad(grad_rhs);
//     }
// }

// // ====== ReLU ======
// ReLUFunction::ReLUFunction(const Tensor* input)
//     : input(input) {}

// std::vector<Tensor*> ReLUFunction::inputs() const {
//     return { const_cast<Tensor*>(input) };
// }

// void ReLUFunction::backward(Tensor& grad_out) {
//     std::vector<float> grad_data(grad_out.numel());
//     for (int i = 0; i < grad_out.numel(); ++i)
//         grad_data[i] = input->data()[i] > 0.0f ? grad_out.data()[i] : 0.0f;

//     Tensor grad(input->shape(), grad_data);
//     if (input->requires_grad()) {
//         const_cast<Tensor*>(input)->accumulate_grad(grad);
//     }
// }

// // ====== CrossEntropy ======
// CrossEntropyFunction::CrossEntropyFunction(const Tensor* logits, const Tensor* target)
//     : logits(logits), target(target) {}

// std::vector<Tensor*> CrossEntropyFunction::inputs() const {
//     return { const_cast<Tensor*>(logits), const_cast<Tensor*>(target) };
// }

// void CrossEntropyFunction::backward(Tensor& grad_out) {
//     const auto& shape = logits->shape();
//     int batch = shape[0];
//     int classes = shape[1];
//     const auto& logits_data = logits->data();
//     const auto& targets = target->data();

//     std::vector<float> probs(batch * classes);
//     for (int i = 0; i < batch; ++i) {
//         float max_logit = -std::numeric_limits<float>::infinity();
//         for (int j = 0; j < classes; ++j)
//             max_logit = std::max(max_logit, logits_data[i * classes + j]);

//         float sum_exp = 0.0f;
//         for (int j = 0; j < classes; ++j) {
//             float exp_val = std::exp(logits_data[i * classes + j] - max_logit);
//             probs[i * classes + j] = exp_val;
//             sum_exp += exp_val;
//         }

//         for (int j = 0; j < classes; ++j)
//             probs[i * classes + j] /= sum_exp;
//     }

//     std::vector<float> grad_data(batch * classes);
//     for (int i = 0; i < batch; ++i) {
//         int true_class = static_cast<int>(targets[i]);
//         for (int j = 0; j < classes; ++j) {
//             float grad = probs[i * classes + j];
//             if (j == true_class) grad -= 1.0f;
//             grad /= static_cast<float>(batch);
//             grad_data[i * classes + j] = grad * grad_out.data()[0];
//         }
//     }

//     Tensor grad(shape, grad_data);
//     if (logits->requires_grad()) {
//         const_cast<Tensor*>(logits)->accumulate_grad(grad);
//     }
// }

// ====== Mul ======
// MulFunction::MulFunction(const Tensor* l, const Tensor* r)
//     : lhs(l), rhs(r) {}

// std::vector<Tensor*> MulFunction::inputs() const {
//     return { const_cast<Tensor*>(lhs), const_cast<Tensor*>(rhs) };
// }

// void MulFunction::backward(Tensor& grad_out) {
//     if (lhs->requires_grad()) {
//         Tensor grad = mul(grad_out, *rhs);
//         const_cast<Tensor*>(lhs)->accumulate_grad(grad);
//     }

//     if (rhs->requires_grad()) {
//         Tensor grad = mul(grad_out, *lhs);
//         const_cast<Tensor*>(rhs)->accumulate_grad(grad);
//     }
// }

// // ====== Softmax ======
// SoftmaxFunction::SoftmaxFunction(const Tensor* input)
//     : input(input) {}

// std::vector<Tensor*> SoftmaxFunction::inputs() const {
//     return { const_cast<Tensor*>(input) };
// }

// void SoftmaxFunction::backward(Tensor& grad_out) {
//     const auto& shape = input->shape();
//     int batch = shape[0];
//     int classes = shape[1];
//     const auto& y = input->data();
//     const auto& go = grad_out.data();

//     std::vector<float> grad_input(input->numel());
//     for (int i = 0; i < batch; ++i) {
//         for (int j = 0; j < classes; ++j) {
//             float sum = 0.0f;
//             for (int k = 0; k < classes; ++k) {
//                 float delta = (j == k) ? 1.0f : 0.0f;
//                 sum += go[i * classes + k] * y[i * classes + j] * (delta - y[i * classes + k]);
//             }
//             grad_input[i * classes + j] = sum;
//         }
//     }

//     Tensor grad(shape, grad_input);
//     if (input->requires_grad()) {
//         const_cast<Tensor*>(input)->accumulate_grad(grad);
//     }
// }


// AddFunction::AddFunction(const Tensor& l, const Tensor& r)
//     : lhs(l), rhs(r) {}

// std::vector<Tensor*> AddFunction::inputs() const {
//     return {  const_cast<Tensor*>(&lhs), const_cast<Tensor*>(&rhs) };
// }

// void AddFunction::backward(Tensor& grad_out) {
//     if (lhs.requires_grad()) {
//         lhs.accumulate_grad(grad_out);
//     }

//     if (rhs.requires_grad()) {
//         const auto& rhs_shape = rhs.shape();
//         const auto& grad_shape = grad_out.shape();

//         if (rhs_shape == grad_shape) {
//             rhs.accumulate_grad(grad_out);
//         } else if (grad_shape.size() == 2 && rhs_shape.size() == 1 && grad_shape[1] == rhs_shape[0]) {
//             int B = grad_shape[0];
//             int F = grad_shape[1];
//             std::vector<float> reduced(F, 0.0f);
//             for (int i = 0; i < B; ++i)
//                 for (int j = 0; j < F; ++j)
//                     reduced[j] += grad_out.data()[i * F + j];

//             Tensor reduced_tensor(rhs_shape, reduced, true);
//             rhs.accumulate_grad(reduced_tensor);
//         } else {
//             throw std::runtime_error("Unsupported broadcast in AddFunction::backward()");
//         }
//     }
// }

// // ====== Matmul ======
// MatmulFunction::MatmulFunction(const Tensor& l, const Tensor& r)
//     : lhs(l), rhs(r) {}

// std::vector<Tensor*> MatmulFunction::inputs() const {
//     return {  const_cast<Tensor*>(&lhs), const_cast<Tensor*>(&rhs) };
// }

// void MatmulFunction::backward(Tensor& grad_out) {
//     if (lhs.requires_grad()) {
//         Tensor rhs_T = rhs.transpose(0, 1);
//         Tensor grad_lhs = matmul_impl(grad_out, rhs_T);
//         lhs.accumulate_grad(grad_lhs);
//     }

//     if (rhs.requires_grad()) {
//         Tensor lhs_T = lhs.transpose(0, 1);
//         Tensor grad_rhs = matmul_impl(lhs_T, grad_out);
//         rhs.accumulate_grad(grad_rhs);
//     }
// }

// // ====== ReLU ======
// ReLUFunction::ReLUFunction(const Tensor& input)
//     : input(input) {}

// std::vector<Tensor*> ReLUFunction::inputs() const {
   
//     return { const_cast<Tensor*>(&input) };
// }

// void ReLUFunction::backward(Tensor& grad_out) {
//     std::vector<float> grad_data(grad_out.numel());
//     for (int i = 0; i < grad_out.numel(); ++i)
//         grad_data[i] = input.data()[i] > 0.0f ? grad_out.data()[i] : 0.0f;

//     Tensor grad(input.shape(), grad_data);
//     if (input.requires_grad()) {
//        input.accumulate_grad(grad);
//     }
// }

// // ====== CrossEntropy ======
// CrossEntropyFunction::CrossEntropyFunction(const Tensor& logits, const Tensor& target)
//     : logits(logits), target(target) {}

// std::vector<Tensor*> CrossEntropyFunction::inputs() const {
//     return { const_cast<Tensor*>(&logits), const_cast<Tensor*>(&target) };
// }

// void CrossEntropyFunction::backward(Tensor& grad_out) {
//     const auto& shape = logits.shape();
//     int batch = shape[0];
//     int classes = shape[1];
//     const auto& logits_data = logits.data();
//     const auto& targets = target.data();

//     std::vector<float> probs(batch * classes);
//     for (int i = 0; i < batch; ++i) {
//         float max_logit = -std::numeric_limits<float>::infinity();
//         for (int j = 0; j < classes; ++j)
//             max_logit = std::max(max_logit, logits_data[i * classes + j]);

//         float sum_exp = 0.0f;
//         for (int j = 0; j < classes; ++j) {
//             float exp_val = std::exp(logits_data[i * classes + j] - max_logit);
//             probs[i * classes + j] = exp_val;
//             sum_exp += exp_val;
//         }

//         for (int j = 0; j < classes; ++j)
//             probs[i * classes + j] /= sum_exp;
//     }

//     std::vector<float> grad_data(batch * classes);
//     for (int i = 0; i < batch; ++i) {
//         int true_class = static_cast<int>(targets[i]);
//         for (int j = 0; j < classes; ++j) {
//             float grad = probs[i * classes + j];
//             if (j == true_class) grad -= 1.0f;
//             grad /= static_cast<float>(batch);
//             grad_data[i * classes + j] = grad * grad_out.data()[0];
//         }
//     }

//     Tensor grad(shape, grad_data);
//     if (logits.requires_grad()) {
//         logits.accumulate_grad(grad);
//     }
// }


// } // namespace znet

// // ====== Mul ======
// MulFunction::MulFunction(const Tensor* l, const Tensor* r)
//     : lhs(l), rhs(r) {}

// std::vector<Tensor*> MulFunction::inputs() const {
//     return { const_cast<Tensor*>(lhs), const_cast<Tensor*>(rhs) };
// }

// void MulFunction::backward(Tensor& grad_out) {
//     if (lhs->requires_grad()) {
//         Tensor grad = mul(grad_out, *rhs);
//         const_cast<Tensor*>(lhs)->accumulate_grad(grad);
//     }

//     if (rhs->requires_grad()) {
//         Tensor grad = mul(grad_out, *lhs);
//         const_cast<Tensor*>(rhs)->accumulate_grad(grad);
//     }
// }

// // ====== Softmax ======
// SoftmaxFunction::SoftmaxFunction(const Tensor* input)
//     : input(input) {}

// std::vector<Tensor*> SoftmaxFunction::inputs() const {
//     return { const_cast<Tensor*>(input) };
// }

// void SoftmaxFunction::backward(Tensor& grad_out) {
//     const auto& shape = input->shape();
//     int batch = shape[0];
//     int classes = shape[1];
//     const auto& y = input->data();
//     const auto& go = grad_out.data();

//     std::vector<float> grad_input(input->numel());
//     for (int i = 0; i < batch; ++i) {
//         for (int j = 0; j < classes; ++j) {
//             float sum = 0.0f;
//             for (int k = 0; k < classes; ++k) {
//                 float delta = (j == k) ? 1.0f : 0.0f;
//                 sum += go[i * classes + k] * y[i * classes + j] * (delta - y[i * classes + k]);
//             }
//             grad_input[i * classes + j] = sum;
//         }
//     }

//     Tensor grad(shape, grad_input);
//     if (input->requires_grad()) {
//         const_cast<Tensor*>(input)->accumulate_grad(grad);
//     }

#include <znet/autograd/autograd_function.hpp>
#include <znet/autograd/ops.hpp>
#include <znet/autograd/kernels.hpp>   
#include <znet/autograd/shape_ops.hpp>
#include <limits>
#include <cmath>
#include <algorithm>

namespace znet {

// ====== Base Function ======
Function::Function() = default;
Function::~Function() = default;

// ==================== Add ====================
AddFunction::AddFunction(const Tensor& a, const Tensor& b) {
    a_.save(a);
    b_.save(b);
    a_view_ = a_.unpack();
    b_view_ = b_.unpack();
}

std::vector<Tensor*> AddFunction::inputs() const {
    return { const_cast<Tensor*>(&a_view_), const_cast<Tensor*>(&b_view_) };
}

void AddFunction::backward(Tensor& grad_out) {
    // // LHS: pass-through (your original behavior)
    // if (a_view_.requires_grad()) {
    //     a_view_.accumulate_grad(grad_out);
    // }

    // // RHS: your original special cases
    // if (b_view_.requires_grad()) {
    //     const auto& rhs_shape  = b_view_.shape();
    //     const auto& grad_shape = grad_out.shape();

    //     if (rhs_shape == grad_shape) {
    //         b_view_.accumulate_grad(grad_out);

    //     } else if (grad_shape.size() == 2 &&
    //                rhs_shape.size()  == 1 &&
    //                grad_shape[1]     == rhs_shape[0]) {
    //         // Reduce over batch for (B,F) + (F)
    //         int B = grad_shape[0];
    //         int F = grad_shape[1];
    //         std::vector<float> reduced(F, 0.0f);
    //         for (int i = 0; i < B; ++i)
    //             for (int j = 0; j < F; ++j)
    //                 reduced[j] += grad_out.data()[i * F + j];

    //         // Grad tensors must not require grad
    //         Tensor reduced_tensor(rhs_shape, reduced, /*requires_grad=*/false);
    //         b_view_.accumulate_grad(reduced_tensor);
    //     } else {
    //         throw std::runtime_error("Unsupported broadcast in AddFunction::backward()");
    //     }
    // }
    // lhs
    if (a_view_.requires_grad()) {
        a_view_.accumulate_grad(grad_out);  // shapes match grad_out by definition
    }

    // rhs
    if (b_view_.requires_grad()) {
        const auto& rhs_shape = b_view_.shape();
        const auto& gout_shape = grad_out.shape();

        if (rhs_shape == gout_shape) {
            b_view_.accumulate_grad(grad_out);
        } else {
            // NEW: general broadcast reduction
            Tensor reduced = sum_to_shape(grad_out, rhs_shape);
            b_view_.accumulate_grad(reduced);
        }
    }
}

// ==================== Matmul ====================
MatmulFunction::MatmulFunction(const Tensor& a, const Tensor& b) {
    a_.save(a);
    b_.save(b);
    a_view_ = a_.unpack();
    b_view_ = b_.unpack();
}

std::vector<Tensor*> MatmulFunction::inputs() const {
    return { const_cast<Tensor*>(&a_view_), const_cast<Tensor*>(&b_view_) };
}

void MatmulFunction::backward(Tensor& grad_out) {
    // Keep your original matmul grad logic (no broadcast reductions added)
    // if (a_view_.requires_grad()) {
    //     // Tensor bT = b_view_.transpose(0, 1);
    //     Tensor grad_a = matmul_impl(grad_out, b_view_);
    //     a_view_.accumulate_grad(grad_a);
    // }

    // if (b_view_.requires_grad()) {
    //     // Tensor gradT = grad_out.transpose(0, 1);
    //     Tensor grad_b = matmul_AT_B_kernel(grad_out,a_view_);
    //     b_view_.accumulate_grad(grad_b);
    // }
    // dA
    const Tensor& A = a_view_;
    const Tensor& B = b_view_;

    // dA = dY @ B^T   -> shapes: dY[... M,N], B[... K,N] -> dA_full[... M,K]
    if (A.requires_grad()) {
        const std::vector<int> dA_shape =
            compute_mm_out_shape_flags(/*A=*/grad_out, /*B=*/B,
                                       /*transA=*/false, /*transB=*/false);

        Tensor dA_full(dA_shape,
                       std::vector<float>(static_cast<size_t>(prod(dA_shape)), 0.0f),
                       /*requires_grad=*/false);

        matmul_strided_batched_kernel(/*A=*/grad_out, /*B=*/B, /*C=*/dA_full,
                                      /*A_logical_trans=*/false,
                                      /*B_logical_trans=*/false); // Bᵀ

        // If A was broadcast across leading dims during forward, reduce back:
        Tensor dA = (dA_full.shape() == A.shape())
                    ? dA_full
                    : sum_to_shape(dA_full, A.shape());

        a_view_.accumulate_grad(dA);
    }

    // dB = A^T @ dY   -> shapes: A[... M,K], dY[... M,N] -> dB_full[... K,N]
    if (B.requires_grad()) {
        const std::vector<int> dB_shape =
            compute_mm_out_shape_flags(/*A=*/A, /*B=*/grad_out,
                                       /*transA=*/true, /*transB=*/false);

        Tensor dB_full(dB_shape,
                       std::vector<float>(static_cast<size_t>(prod(dB_shape)), 0.0f),
                       /*requires_grad=*/false);

        matmul_strided_batched_kernel(/*A=*/A, /*B=*/grad_out, /*C=*/dB_full,
                                      /*A_logical_trans=*/true,   // Aᵀ
                                      /*B_logical_trans=*/false);

        Tensor dB = (dB_full.shape() == B.shape())
                    ? dB_full
                    : sum_to_shape(dB_full, B.shape());

        b_view_.accumulate_grad(dB);
    }
}

// ==================== ReLU ====================
ReLUFunction::ReLUFunction(const Tensor& x) {
    x_.save(x);
    x_view_ = x_.unpack();

    // // Save output to build a correct mask in backward
    // // (In your implementation, out is contiguous, which makes this trivial.)
    // y_.save(y);
    // y_view_ = y_.unpack();
}

std::vector<Tensor*> ReLUFunction::inputs() const {
    return { const_cast<Tensor*>(&x_view_) };
}

void ReLUFunction::backward(Tensor& grad_out) {
    // std::vector<float> grad_data(grad_out.numel());
    // for (int i = 0; i < grad_out.numel(); ++i) {
    //     grad_data[i] = x_view_.data()[i] > 0.0f ? grad_out.data()[i] : 0.0f;
    // }
    // Tensor gx(x_view_.shape(), grad_data, /*requires_grad=*/false);
    // if (x_view_.requires_grad()) {
    //     x_view_.accumulate_grad(gx);
    // }
    // Mask using y > 0  (since y = relu(x), y>0 <=> x>0 elementwise)
    // std::vector<float> grad_data(grad_out.numel());
    // const auto& ydata = y_view_.data();
    // const auto& gout  = grad_out.data();

    // for (int i = 0; i < grad_out.numel(); ++i) {
    //     grad_data[i] = (ydata[i] > 0.0f) ? gout[i] : 0.0f;
    // }

    // Tensor gx(x_view_.shape(), grad_data, /*requires_grad=*/false);
    // if (x_view_.requires_grad()) {
    //     x_view_.accumulate_grad(gx);
    // }
    // Ensure x.grad() buffer exists (zeros)
    if (!x_view_.grad()) {
        x_view_.set_grad(std::make_shared<Tensor>(
            x_view_.shape(), std::vector<float>(x_view_.numel(), 0.0f), /*requires_grad=*/false));
    }
    // Accumulate in-place: grad_x += mask(x>0) * grad_out   (stride-aware)
    relu_backward_kernel_accum(*x_view_.grad(), x_view_, grad_out);
}

// ==================== CrossEntropy ====================
CrossEntropyFunction::CrossEntropyFunction(const Tensor& l, const Tensor& t) {
    logits_.save(l);
    target_.save(t);
    logits_view_ = logits_.unpack();
    target_view_ = target_.unpack();
}

std::vector<Tensor*> CrossEntropyFunction::inputs() const {
    return { const_cast<Tensor*>(&logits_view_), const_cast<Tensor*>(&target_view_) };
}

void CrossEntropyFunction::backward(Tensor& grad_out) {
    const auto& shape = logits_view_.shape();   // [B, C]
    int batch   = shape[0];
    int classes = shape[1];
    const auto& logits_data = logits_view_.data();
    const auto& targets     = target_view_.data();

    // softmax (stable)
    std::vector<float> probs(batch * classes);
    for (int i = 0; i < batch; ++i) {
        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < classes; ++j)
            m = std::max(m, logits_data[i * classes + j]);

        float s = 0.0f;
        for (int j = 0; j < classes; ++j) {
            float e = std::exp(logits_data[i * classes + j] - m);
            probs[i * classes + j] = e;
            s += e;
        }

        for (int j = 0; j < classes; ++j)
            probs[i * classes + j] /= s;
    }

    // dL/dz = (softmax - one_hot) / B * grad_out_scalar
    float go = grad_out.numel() ? grad_out.data()[0] : 1.0f;
    std::vector<float> grad_data(batch * classes);
    for (int i = 0; i < batch; ++i) {
        int y = static_cast<int>(targets[i]);
        for (int j = 0; j < classes; ++j) {
            float g = probs[i * classes + j];
            if (j == y) g -= 1.0f;
            g /= static_cast<float>(batch);
            grad_data[i * classes + j] = g * go;
        }
    }

    Tensor grad(shape, grad_data, /*requires_grad=*/false);
    if (logits_view_.requires_grad()) {
        logits_view_.accumulate_grad(grad);
    }
}

} // namespace znet


// namespace znet {

// // ====== Base Function ======
// Function::Function() = default;
// Function::~Function() = default;

// // ---------- small helper: sum_to_shape (broadcast reduction) ----------
// namespace {
// Tensor sum_to_shape(const Tensor& g, const std::vector<int>& target) {
//     // Trivial fast path
//     if (g.shape() == target) {
//         // Return a no-grad copy/alias as your API requires
//         return Tensor::from_impl(g.impl()); // alias is fine if accumulate_grad copies
//     }

//     // Right-align shapes
//     const auto& gs = g.shape();
//     const int G = (int)gs.size();
//     const int T = (int)target.size();
//     std::vector<int> tgt = target;
//     if (T < G) {
//         tgt.insert(tgt.begin(), G - T, 1);
//     }

//     // Axes to reduce: dims where tgt == 1 and gs > 1, plus leading extras if any
//     std::vector<int> reduce_axes;
//     for (int i = 0; i < G; ++i) {
//         if ((i < G - (int)target.size()) || (tgt[i] == 1 && gs[i] > 1)) {
//             reduce_axes.push_back(i);
//         }
//     }

//     // Implement reduction by repeated sum over axes (from highest to lowest)
//     Tensor out = g;
//     for (int k = (int)reduce_axes.size() - 1; k >= 0; --k) {
//         out = sum_impl(out, reduce_axes[k]);        // <- implement sum over axis
//     }

//     // Finally, reshape to the exact target (remove leading 1s etc.)
//     if ((int)target.size() != (int)out.shape().size() || out.shape() != target) {
//         out = reshape_impl(out, target);
//     }
//     return out; // must be requires_grad = false (your sum/reshape impls should ensure that)
// }
// } // namespace


// } // namespace znet
